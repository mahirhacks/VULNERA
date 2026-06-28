"""
Tune calibrated deployment threshold for window-stack scoring.

Target policy (all strict inequalities on valid):
  f1 > 0.5, precision > 0.4, recall > 0.6

Among qualifying thresholds, pick highest F1, then precision, then recall.
If none qualify, minimize weighted deficit to those floors.

Usage:
    python 05_SCORE/training_scripts/tune_window_stack_deployment.py
    python 05_SCORE/training_scripts/tune_window_stack_deployment.py --min-f1 0.5 --min-precision 0.4 --min-recall 0.6
"""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

SCORE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCORE_ROOT.parent
AGGREGATOR_ROOT = PROJECT_ROOT / "06_AGGREGATOR"
DEFAULT_SCORE_CONFIG = SCORE_ROOT / "score_config.yaml"
DEFAULT_AGGREGATOR_CONFIG = AGGREGATOR_ROOT / "aggregator_config.yaml"

_AGGREGATOR_SCRIPTS = AGGREGATOR_ROOT / "training_scripts"
_SCORE_SCRIPTS = SCORE_ROOT / "training_scripts"
if str(_AGGREGATOR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_AGGREGATOR_SCRIPTS))
if str(_SCORE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCORE_SCRIPTS))

from aggregator_common import import_meta_common, load_config, resolve_path  # noqa: E402
from score_common import select_threshold_balanced  # noqa: E402
from window_stack_common import score_window_embeddings  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_trees", PROJECT_ROOT / "03_TREE" / "training_scripts" / "train_trees.py"
)
_train_trees = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train_trees)


def load_window_split(embeddings_root: str, pool: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = resolve_path(f"{embeddings_root}/{pool}/{split}/{split}_window_embeddings.parquet")
    frame = pd.read_parquet(path, columns=["label", "embedding"])
    x = np.vstack(frame["embedding"].to_numpy()).astype(np.float32)
    y = frame["label"].astype(int).to_numpy()
    return x, y


def load_stack(aggregator_cfg: dict, meta_settings: dict) -> dict:
    model_keys = {
        "xgb": "xgboost_model",
        "lightgbm": "lightgbm_model",
        "random_forest": "random_forest_model",
        "extra_trees": "extra_trees_model",
    }
    feature_columns = [str(col) for col in meta_settings["feature_columns"]]
    base_models = {
        col: joblib.load(resolve_path(str(meta_settings[model_keys[col]])))
        for col in feature_columns
    }
    return {
        "base_models": base_models,
        "feature_columns": feature_columns,
        "meta_model": joblib.load(resolve_path(str(aggregator_cfg["meta_model"]))),
        "calibrator_bundle": joblib.load(resolve_path(str(aggregator_cfg["score_calibrator"]))),
    }


def sync_deployment(
    deployment_path: Path,
    score_config_path: Path,
    *,
    threshold: float,
    method: str,
    valid_row: pd.Series,
    test_row: pd.Series,
    policy: dict[str, float],
) -> None:
    deployment = {
        "selected_method": method,
        "deployment_threshold_calibrated": round(float(threshold), 4),
        "threshold_policy": policy,
        "valid_at_deployment_threshold": {
            k: float(valid_row[k])
            for k in ["threshold", "precision", "recall", "f1", "roc_auc", "avg_precision"]
            if k in valid_row.index
        },
        "test_at_deployment_threshold": {
            k: float(test_row[k])
            for k in ["threshold", "precision", "recall", "f1", "roc_auc", "avg_precision"]
            if k in test_row.index
        },
    }
    deployment_path.write_text(json.dumps(deployment, indent=2), encoding="utf-8")

    with score_config_path.open(encoding="utf-8") as handle:
        score_cfg = yaml.safe_load(handle)
    block = score_cfg.setdefault("calibrate_scores", {})
    block["deployment_threshold_calibrated"] = round(float(threshold), 4)
    block["selected_method"] = method
    with score_config_path.open("w", encoding="utf-8") as handle:
        yaml.dump(score_cfg, handle, sort_keys=False, default_flow_style=False, allow_unicode=True)


def main() -> None:
    parser = ArgumentParser(description="Tune window-stack calibrated deployment threshold.")
    parser.add_argument("--score-config", type=Path, default=DEFAULT_SCORE_CONFIG)
    parser.add_argument("--aggregator-config", type=Path, default=DEFAULT_AGGREGATOR_CONFIG)
    parser.add_argument("--min-f1", type=float, default=0.5)
    parser.add_argument("--min-precision", type=float, default=0.4)
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    aggregator_path = (
        args.aggregator_config
        if args.aggregator_config.is_absolute()
        else AGGREGATOR_ROOT / args.aggregator_config
    )
    score_config_path = args.score_config if args.score_config.is_absolute() else SCORE_ROOT / args.score_config

    aggregator = load_config(aggregator_path).get("run_aggregator", {})
    score_cfg = load_config(score_config_path).get("calibrate_scores", {})
    meta_common = import_meta_common()
    meta_config = meta_common.load_config(resolve_path(str(aggregator["meta_config"])))
    meta_settings = meta_config.get("train_meta", {})

    pool = str(aggregator.get("window_pool", aggregator.get("pool", "max")))
    embeddings_root = str(aggregator.get("embeddings_root", "01_Data_Processing/data/embeddings"))
    step = float(args.step)

    stack = load_stack(aggregator, meta_settings)
    method = str(score_cfg.get("selected_method", stack["calibrator_bundle"].get("method", "isotonic")))

    x_valid, y_valid = load_window_split(embeddings_root, pool, "valid")
    x_test, y_test = load_window_split(embeddings_root, pool, "test")

    _, valid_cal, _ = score_window_embeddings(
        x_valid,
        base_models=stack["base_models"],
        feature_columns=stack["feature_columns"],
        meta_model=stack["meta_model"],
        calibrator_bundle=stack["calibrator_bundle"],
    )
    _, test_cal, _ = score_window_embeddings(
        x_test,
        base_models=stack["base_models"],
        feature_columns=stack["feature_columns"],
        meta_model=stack["meta_model"],
        calibrator_bundle=stack["calibrator_bundle"],
    )

    valid_curve = meta_common.sweep_threshold_curve(y_valid, valid_cal, step=step)
    test_curve = meta_common.sweep_threshold_curve(y_test, test_cal, step=step)

    policy = {
        "min_f1": args.min_f1,
        "min_precision": args.min_precision,
        "min_recall": args.min_recall,
    }
    deploy_threshold, valid_metrics = select_threshold_balanced(
        valid_curve,
        min_f1=args.min_f1,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
    )
    threshold = deploy_threshold
    valid_at = valid_curve.loc[valid_curve["threshold"] == threshold].iloc[0]
    test_at = test_curve.loc[test_curve["threshold"] == threshold].iloc[0]

    deployment_path = resolve_path(str(aggregator["calibrated_deployment"]))
    sync_deployment(
        deployment_path,
        score_config_path,
        threshold=threshold,
        method=method,
        valid_row=valid_at,
        test_row=test_at,
        policy=policy,
    )

    results_dir = resolve_path(str(score_cfg.get("output_dir", "05_SCORE/results/window_stack")))
    results_dir.mkdir(parents=True, exist_ok=True)
    valid_curve.to_csv(results_dir / "window_stack_threshold_sweep_valid.csv", index=False)
    test_curve.to_csv(results_dir / "window_stack_threshold_sweep_test.csv", index=False)

    qualified = valid_curve[
        (valid_curve["f1"] > args.min_f1)
        & (valid_curve["precision"] > args.min_precision)
        & (valid_curve["recall"] > args.min_recall)
    ]
    meets_policy = bool(valid_metrics.get("meets_policy", len(qualified) > 0))
    report = f"""Window-stack deployment threshold tuning
Policy: f1 > {args.min_f1:.2f}, precision > {args.min_precision:.2f}, recall > {args.min_recall:.2f}
Qualifying thresholds on valid: {len(qualified)}
Meets policy on valid: {meets_policy}

Selected threshold: {threshold:.4f}

VALID ({len(y_valid):,} windows)
  precision={valid_at.precision:.4f}  recall={valid_at.recall:.4f}  f1={valid_at.f1:.4f}

TEST ({len(y_test):,} windows)
  precision={test_at.precision:.4f}  recall={test_at.recall:.4f}  f1={test_at.f1:.4f}

Updated: {deployment_path}
Updated: {score_config_path}
"""
    (results_dir / "window_stack_threshold_tune_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
