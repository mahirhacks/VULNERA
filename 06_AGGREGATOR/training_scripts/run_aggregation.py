"""
Window-level stack scoring with max-pool aggregation to function and file tiers.

Pipeline:
  window embedding -> 4 trees -> meta -> calibration -> window_prob
  function_score = max(window_prob per function)
  file_score     = max(function_score per file)  [via file_score.py at scan time]

Usage:
    python 06_AGGREGATOR/training_scripts/run_aggregation.py
    python 06_AGGREGATOR/training_scripts/run_aggregation.py --split test
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aggregator_common import (
    DEFAULT_CONFIG_PATH,
    AGGREGATOR_ROOT,
    apply_score_calibrator,
    build_function_records,
    function_meta_from_window_frame,
    import_meta_common,
    import_train_trees,
    load_config,
    resolve_path,
    summarize_breakdown,
)
from window_stack_common import attach_window_scores, function_scores_from_windows, score_window_embeddings


def load_window_frame(
    cfg: dict[str, Any],
    split: str,
    pool: str,
) -> pd.DataFrame:
    path = resolve_path(f"{cfg['embeddings_root']}/{pool}/{split}/{split}_window_embeddings.parquet")
    frame = pd.read_parquet(path)
    frame = frame.copy()
    frame["window_index"] = frame.groupby("function_group_id", sort=False).cumcount().astype(int)
    return frame


def load_window_stack_models(cfg: dict[str, Any], meta_settings: dict[str, Any]) -> dict[str, Any]:
    model_keys = {
        "xgb": "xgboost_model",
        "lightgbm": "lightgbm_model",
        "random_forest": "random_forest_model",
        "extra_trees": "extra_trees_model",
    }
    feature_columns = [str(col) for col in meta_settings["feature_columns"]]
    base_models: dict[str, Any] = {}
    for col in feature_columns:
        base_models[col] = joblib.load(resolve_path(str(meta_settings[model_keys[col]])))
    return {
        "base_models": base_models,
        "feature_columns": feature_columns,
        "meta_model": joblib.load(resolve_path(str(cfg["meta_model"]))),
        "calibrator_bundle": joblib.load(resolve_path(str(cfg["score_calibrator"]))),
    }


def score_all_windows(
    window_frame: pd.DataFrame,
    *,
    stack: dict[str, Any],
    embedding_column: str = "embedding",
) -> pd.DataFrame:
    embeddings = np.vstack(window_frame[embedding_column].to_numpy()).astype(np.float32)
    raw_scores, calibrated, _ = score_window_embeddings(
        embeddings,
        base_models=stack["base_models"],
        feature_columns=stack["feature_columns"],
        meta_model=stack["meta_model"],
        calibrator_bundle=stack["calibrator_bundle"],
    )
    return attach_window_scores(window_frame, calibrated_scores=calibrated, raw_scores=raw_scores)


def main() -> None:
    parser = ArgumentParser(description="Run window-stack aggregation analysis.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", type=str, default=None, choices=["valid", "test"])
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else AGGREGATOR_ROOT / args.config
    config = load_config(config_path)
    cfg = config.get("run_aggregator", {})
    meta_common = import_meta_common()
    train_trees = import_train_trees()

    split = args.split or str(cfg.get("split", "test"))
    scoring_mode = str(cfg.get("scoring_mode", "window_stack_aggregate")).lower()
    pool = str(cfg.get("pool", "max"))
    window_pool = str(cfg.get("window_pool", pool))
    label_column = str(cfg.get("label_column", "label"))
    function_group_column = "function_group_id"
    tolerance = float(cfg.get("max_pool_tolerance", 1e-6))

    meta_config = meta_common.load_config(resolve_path(str(cfg["meta_config"])))
    meta_settings = meta_config.get("train_meta", {})
    stack = load_window_stack_models(cfg, meta_settings)

    calibrated_deployment = json.loads(resolve_path(str(cfg["calibrated_deployment"])).read_text(encoding="utf-8"))
    deployment_threshold = float(calibrated_deployment["deployment_threshold_calibrated"])

    window_frame = load_window_frame(cfg, split, window_pool)
    window_frame = score_all_windows(window_frame, stack=stack)

    function_meta = function_meta_from_window_frame(
        window_frame,
        function_group_column=function_group_column,
        label_column=label_column,
    )
    function_frame = function_meta.copy()
    calibrated_scores = function_scores_from_windows(
        window_frame,
        function_meta,
        function_group_column=function_group_column,
        threshold=deployment_threshold,
        weight=float(cfg.get("spread_weight", 0.25)),
    )

    precision_cfg = {
        "function_threshold_triage": deployment_threshold,
        "window_threshold_triage": deployment_threshold,
        "window_threshold_confirmed": deployment_threshold,
    }
    window_threshold_confirmed = deployment_threshold

    records = build_function_records(
        function_frame=function_frame,
        function_group_column=function_group_column,
        label_column=label_column,
        calibrated_scores=calibrated_scores,
        function_threshold=deployment_threshold,
        window_frame=window_frame,
        window_threshold=deployment_threshold,
        tolerance=tolerance,
        precision_cfg=precision_cfg,
        scoring_mode=scoring_mode,
    )

    spread_weight = float(cfg.get("spread_weight", 0.25))
    window_probs = window_frame["window_prob"].to_numpy(dtype=np.float32)
    window_labels = window_frame[label_column].astype(int).to_numpy()
    function_labels = function_frame[label_column].astype(int).to_numpy()

    window_metrics = train_trees.compute_metrics(
        window_labels, window_probs, threshold=deployment_threshold
    )
    function_metrics = train_trees.compute_metrics(
        function_labels, calibrated_scores, threshold=deployment_threshold
    )

    breakdown = summarize_breakdown(records)
    breakdown["split"] = split
    breakdown["scoring_mode"] = scoring_mode
    breakdown["deployment_threshold_calibrated"] = deployment_threshold
    breakdown["window_threshold"] = deployment_threshold
    breakdown["spread_weight"] = spread_weight
    breakdown["layer_metrics"] = {
        "window": {
            "unit": "window",
            "threshold": deployment_threshold,
            "precision": window_metrics["precision"],
            "recall": window_metrics["recall"],
            "f1": window_metrics["f1"],
            "roc_auc": window_metrics["roc_auc"],
            "count": int(len(window_labels)),
        },
        "function": {
            "unit": "function",
            "threshold": deployment_threshold,
            "spread_weight": spread_weight,
            "pooling": "max_plus_mean_excess",
            "precision": function_metrics["precision"],
            "recall": function_metrics["recall"],
            "f1": function_metrics["f1"],
            "roc_auc": function_metrics["roc_auc"],
            "count": int(len(function_labels)),
        },
        "file": {
            "unit": "file",
            "pooling": "max_plus_mean_excess",
            "spread_weight": spread_weight,
            "threshold": deployment_threshold,
            "note": (
                "Offline eval is one function per sample; file risk is computed at scan time "
                "via build_file_score() over all functions in the uploaded file."
            ),
            "formula": (
                "file_risk = min(1, max(function_score) + "
                f"{spread_weight} * mean(max(0, other_function_scores - tau)))"
            ),
        },
    }
    from aggregator_common import summarize_precision_tiers  # noqa: PLC0415

    breakdown["precision_tiers"] = summarize_precision_tiers(records)
    breakdown["window_threshold_confirmed"] = window_threshold_confirmed
    breakdown["paper_note"] = (
        "Each window: 4 tree probabilities -> meta learner -> isotonic calibration -> window_prob. "
        f"Function risk = max(window_prob) + {spread_weight} * spread uplift across other windows (tau={deployment_threshold:.4f}). "
        f"File risk = max(function risk) + {spread_weight} * spread uplift across other functions in the uploaded file."
    )

    results_dir = resolve_path(str(cfg.get("output_dir", "06_AGGREGATOR/results")))
    artifacts_dir = resolve_path(str(cfg.get("artifacts_dir", "06_AGGREGATOR/artifacts")))
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    per_function_path = artifacts_dir / f"{split}_function_aggregation.parquet"
    per_function_json_path = artifacts_dir / f"{split}_function_aggregation.json"
    breakdown_path = results_dir / f"{split}_aggregation_breakdown.json"

    slim_records = [
        {
            "function_group_id": record["function_group_id"],
            "label": record["label"],
            "function_score_calibrated": record["function_score_calibrated"],
            "function_flagged": record["function_flagged"],
            "agreement_status": record["agreement_status"],
            "deployment_tier": record.get("deployment_tier"),
            "user_facing_vuln": record.get("user_facing_vuln"),
            "whole_function_vuln": record.get("whole_function_vuln"),
            "window_count": record["window_count"],
            "flagged_window_indices": record["flagged_window_indices"],
            "confirmed_window_indices": record.get("confirmed_window_indices", []),
            "flagged_window_ids": record["flagged_window_ids"],
            "contributing_window_indices": record["contributing_window_indices"],
            "contributing_window_ids": record["contributing_window_ids"],
            "max_window_prob": record["max_window_prob"],
        }
        for record in records
    ]
    pd.DataFrame(slim_records).to_parquet(per_function_path, index=False)
    per_function_json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    breakdown_path.write_text(json.dumps(breakdown, indent=2), encoding="utf-8")

    window_preds_path = artifacts_dir / f"{split}_window_predictions.parquet"
    export_cols = [c for c in window_frame.columns if c != "embedding"]
    window_frame[export_cols].to_parquet(window_preds_path, index=False)

    lines = [
        f"Vulnera — window-stack aggregation ({split})",
        "",
        f"Scoring mode: {scoring_mode}",
        f"Deployment threshold (window + function + file tau): {deployment_threshold:.4f}",
        f"Spread uplift weight: {spread_weight}",
        "",
        "Layer metrics @ deployment threshold:",
        (
            f"  Window:   P={window_metrics['precision']:.4f}  R={window_metrics['recall']:.4f}  "
            f"F1={window_metrics['f1']:.4f}  (n={len(window_labels):,})"
        ),
        (
            f"  Function: P={function_metrics['precision']:.4f}  R={function_metrics['recall']:.4f}  "
            f"F1={function_metrics['f1']:.4f}  (n={len(function_labels):,}, max+uplift)"
        ),
        "  File:     computed at scan time (max+uplift over functions in upload)",
        "",
        f"{'Status':<18} {'Count':>8} {'Pct':>8}  label_0  label_1",
        "-" * 56,
    ]
    for status, count in breakdown["counts"].items():
        pct = breakdown["percentages"][status] * 100.0
        labels = breakdown["by_label"][status]
        lines.append(
            f"{status:<18} {count:>8} {pct:>7.2f}%  "
            f"{labels['label_0']:>7} {labels['label_1']:>7}"
        )
    lines.extend(
        [
            "",
            f"Per-function artifact: {per_function_path}",
            f"Window predictions:  {window_preds_path}",
            f"Breakdown:           {breakdown_path}",
        ]
    )
    report = "\n".join(lines) + "\n"
    (results_dir / f"{split}_aggregation_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
