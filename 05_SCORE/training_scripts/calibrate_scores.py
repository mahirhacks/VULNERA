"""
Probability calibration for locked logistic meta-learner scores.

Fits Isotonic and Platt calibrators on full validation raw meta probabilities.
Selects the winner by validation Brier (tie-break ECE).
Reports Brier/ECE on held-out test as primary calibration evidence.
Re-derives deployment threshold (recall >= min_recall, max precision) on calibrated valid scores.

Usage:
    python 05_SCORE/training_scripts/calibrate_scores.py
    python 05_SCORE/training_scripts/calibrate_scores.py --method isotonic
"""

from __future__ import annotations

import importlib.util
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from score_common import (
    DEFAULT_CONFIG_PATH,
    SCORE_ROOT,
    brier_score,
    expected_calibration_error,
    import_meta_common,
    load_config,
    reliability_bins,
    resolve_path,
    select_threshold_balanced,
    threshold_policy_from_config,
)

_spec = importlib.util.spec_from_file_location(
    "train_trees", SCORE_ROOT.parent / "03_TREE" / "training_scripts" / "train_trees.py"
)
_train_trees = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train_trees)


def predict_meta_prob(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1].astype(np.float32)


def load_meta_frame(meta_common: Any, settings: dict[str, Any], split: str, feature_columns: list[str], label_column: str):
    if split == "valid":
        input_path = meta_common.resolve_path(str(settings["input_table"]))
        if input_path.exists():
            return pd.read_parquet(input_path)
    return meta_common.build_split_table(settings, split, feature_columns, label_column)


def fit_isotonic(raw_valid: np.ndarray, y_valid: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_valid, y_valid)
    return calibrator


def fit_platt(raw_valid: np.ndarray, y_valid: np.ndarray, random_state: int) -> LogisticRegression:
    calibrator = LogisticRegression(max_iter=1000, random_state=random_state)
    calibrator.fit(raw_valid.reshape(-1, 1), y_valid)
    return calibrator


def apply_calibrator(method: str, calibrator: Any, raw_scores: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        return np.clip(calibrator.predict(raw_scores), 0.0, 1.0).astype(np.float32)
    return calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1].astype(np.float32)


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, ece_bins: int) -> dict[str, float]:
    return {
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=ece_bins),
    }


def method_summary(
    method: str,
    *,
    raw_valid: np.ndarray,
    raw_test: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    calibrator: Any,
    ece_bins: int,
    min_recall: float,
    step: float,
    meta_common: Any,
    threshold_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_cal = apply_calibrator(method, calibrator, raw_valid)
    test_cal = apply_calibrator(method, calibrator, raw_test)

    valid_raw_metrics = calibration_metrics(y_valid, raw_valid, ece_bins=ece_bins)
    test_raw_metrics = calibration_metrics(y_test, raw_test, ece_bins=ece_bins)
    valid_cal_metrics = calibration_metrics(y_valid, valid_cal, ece_bins=ece_bins)
    test_cal_metrics = calibration_metrics(y_test, test_cal, ece_bins=ece_bins)

    valid_curve = meta_common.sweep_threshold_curve(y_valid, valid_cal, step=step)
    if threshold_policy and threshold_policy.get("name") == "balanced":
        deploy_threshold, valid_at_deploy = select_threshold_balanced(
            valid_curve,
            min_f1=float(threshold_policy["min_f1"]),
            min_precision=float(threshold_policy["min_precision"]),
            min_recall=float(threshold_policy["min_recall"]),
        )
    else:
        deploy_threshold, valid_at_deploy = meta_common.select_threshold_max_precision_at_recall(
            valid_curve, min_recall
        )
    test_at_deploy = _train_trees.compute_metrics(y_test, test_cal, threshold=deploy_threshold)

    return {
        "method": method,
        "valid": {
            "raw": valid_raw_metrics,
            "calibrated": valid_cal_metrics,
            "reliability_raw": reliability_bins(y_valid, raw_valid, n_bins=ece_bins),
            "reliability_calibrated": reliability_bins(y_valid, valid_cal, n_bins=ece_bins),
        },
        "test": {
            "raw": test_raw_metrics,
            "calibrated": test_cal_metrics,
            "reliability_raw": reliability_bins(y_test, raw_test, n_bins=ece_bins),
            "reliability_calibrated": reliability_bins(y_test, test_cal, n_bins=ece_bins),
            "at_deployment_threshold": test_at_deploy,
        },
        "deployment_threshold_calibrated": deploy_threshold,
        "valid_at_deployment_threshold": valid_at_deploy,
        "calibrator": calibrator,
        "valid_calibrated_scores": valid_cal,
        "test_calibrated_scores": test_cal,
    }


def pick_winner(
    results: list[dict[str, Any]],
    *,
    threshold_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if threshold_policy and threshold_policy.get("name") == "balanced":
        min_f1 = float(threshold_policy["min_f1"])
        min_p = float(threshold_policy["min_precision"])
        min_r = float(threshold_policy["min_recall"])

        def sort_key(row: dict[str, Any]) -> tuple:
            valid_at = row["valid_at_deployment_threshold"]
            meets = bool(
                valid_at.get("f1", 0.0) > min_f1
                and valid_at.get("precision", 0.0) > min_p
                and valid_at.get("recall", 0.0) > min_r
            )
            deficit = (
                max(0.0, min_f1 - valid_at.get("f1", 0.0)) * 2.0
                + max(0.0, min_p - valid_at.get("precision", 0.0))
                + max(0.0, min_r - valid_at.get("recall", 0.0))
            )
            return (
                meets,
                valid_at.get("f1", 0.0),
                -row["valid"]["calibrated"]["brier"],
                -row["valid"]["calibrated"]["ece"],
                0 if row["method"] == "isotonic" else 1,
                -deficit,
            )

        return sorted(results, key=sort_key, reverse=True)[0]

    return sorted(
        results,
        key=lambda row: (
            row["valid"]["calibrated"]["brier"],
            row["valid"]["calibrated"]["ece"],
            0 if row["method"] == "isotonic" else 1,
        ),
    )[0]


def sync_score_config(config_path: Path, winner_method: str, deployment_threshold: float) -> None:
    """Update only runtime fields; preserve comments by editing known keys in place."""
    text = config_path.read_text(encoding="utf-8")
    if "selected_method:" in text:
        import re  # noqa: PLC0415

        thresh = round(float(deployment_threshold), 4)
        text = re.sub(r'(\n  selected_method:\s*).*$', rf'\1"{winner_method}"', text, count=1)
        text = re.sub(
            r'(\n  deployment_threshold_calibrated:\s*).*$',
            lambda m: f"{m.group(1)}{thresh}",
            text,
            count=1,
        )
        config_path.write_text(text, encoding="utf-8")
        return

    config = load_config(config_path)
    block = config.setdefault("calibrate_scores", {})
    block["selected_method"] = winner_method
    block["deployment_threshold_calibrated"] = round(float(deployment_threshold), 4)
    import yaml  # noqa: PLC0415

    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False, default_flow_style=False)


def main() -> None:
    parser = ArgumentParser(description="Calibrate meta-learner probability scores.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--method", type=str, default=None, choices=["isotonic", "platt"])
    args = parser.parse_args()

    meta_common = import_meta_common()
    config_path = args.config if args.config.is_absolute() else SCORE_ROOT / args.config
    config = load_config(config_path)
    cfg = config.get("calibrate_scores", {})

    meta_config_path = resolve_path(str(cfg.get("meta_config", "04_META/meta_config.yaml")))
    meta_config = meta_common.load_config(meta_config_path)
    settings = meta_config.get("train_meta", {})

    feature_columns = [str(col) for col in settings.get("feature_columns", ["xgb", "lightgbm"])]
    label_column = str(settings.get("label_column", "label"))
    min_recall = float(cfg.get("min_recall", 0.6))
    threshold_policy = threshold_policy_from_config(cfg)
    step = float(cfg.get("threshold_step", 0.01))
    ece_bins = int(cfg.get("ece_bins", 10))
    random_state = int(cfg.get("random_state", 42))
    methods = [args.method] if args.method else [str(m) for m in cfg.get("methods", ["isotonic", "platt"])]

    variant_dir = resolve_path(str(cfg.get("variant_dir", "05_SCORE/logistic")))
    results_dir = resolve_path(str(cfg.get("output_dir", "05_SCORE/results")))
    variant_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    meta_model_path = resolve_path(str(cfg.get("meta_model", "04_META/logistic/meta_model.joblib")))
    meta_model = joblib.load(meta_model_path)

    valid_frame = load_meta_frame(meta_common, settings, "valid", feature_columns, label_column)
    test_frame = meta_common.build_split_table(settings, "test", feature_columns, label_column)

    x_valid = valid_frame[feature_columns].to_numpy(dtype=np.float32)
    y_valid = valid_frame[label_column].astype(int).to_numpy()
    x_test = test_frame[feature_columns].to_numpy(dtype=np.float32)
    y_test = test_frame[label_column].astype(int).to_numpy()

    raw_valid = predict_meta_prob(meta_model, x_valid)
    raw_test = predict_meta_prob(meta_model, x_test)

    raw_deployment_path = resolve_path(str(cfg.get("raw_deployment", "04_META/logistic/deployment_threshold.json")))
    raw_deployment = json.loads(raw_deployment_path.read_text(encoding="utf-8")) if raw_deployment_path.exists() else {}

    method_results: list[dict[str, Any]] = []
    for method in methods:
        if method == "isotonic":
            calibrator = fit_isotonic(raw_valid, y_valid)
        elif method == "platt":
            calibrator = fit_platt(raw_valid, y_valid, random_state)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        method_results.append(
            method_summary(
                method,
                raw_valid=raw_valid,
                raw_test=raw_test,
                y_valid=y_valid,
                y_test=y_test,
                calibrator=calibrator,
                ece_bins=ece_bins,
                min_recall=min_recall,
                step=step,
                meta_common=meta_common,
                threshold_policy=threshold_policy,
            )
        )

    winner = pick_winner(method_results, threshold_policy=threshold_policy)
    winner_method = str(winner["method"])
    deploy_threshold = float(winner["deployment_threshold_calibrated"])

    calibrator_path = variant_dir / "score_calibrator.joblib"
    joblib.dump(
        {
            "method": winner_method,
            "calibrator": winner["calibrator"],
            "fit_split": "valid",
            "meta_model": str(meta_model_path),
        },
        calibrator_path,
    )

    pd.DataFrame({"raw_score": raw_valid, "calibrated_score": winner["valid_calibrated_scores"], "label": y_valid}).to_parquet(
        variant_dir / "valid_calibrated_scores.parquet", index=False
    )
    pd.DataFrame({"raw_score": raw_test, "calibrated_score": winner["test_calibrated_scores"], "label": y_test}).to_parquet(
        variant_dir / "test_calibrated_scores.parquet", index=False
    )

    comparison_rows = []
    for row in method_results:
        comparison_rows.append(
            {
                "method": row["method"],
                "valid_brier_raw": row["valid"]["raw"]["brier"],
                "valid_brier_calibrated": row["valid"]["calibrated"]["brier"],
                "valid_ece_raw": row["valid"]["raw"]["ece"],
                "valid_ece_calibrated": row["valid"]["calibrated"]["ece"],
                "test_brier_raw": row["test"]["raw"]["brier"],
                "test_brier_calibrated": row["test"]["calibrated"]["brier"],
                "test_ece_raw": row["test"]["raw"]["ece"],
                "test_ece_calibrated": row["test"]["calibrated"]["ece"],
                "deployment_threshold_calibrated": row["deployment_threshold_calibrated"],
                "test_precision": row["test"]["at_deployment_threshold"]["precision"],
                "test_recall": row["test"]["at_deployment_threshold"]["recall"],
                "test_f1": row["test"]["at_deployment_threshold"]["f1"],
            }
        )
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(results_dir / "calibration_comparison.csv", index=False)

    granularity = str(cfg.get("granularity", "function"))
    unit = "windows" if granularity == "window" else "functions"

    if threshold_policy:
        policy = threshold_policy
        threshold_note = (
            f"f1 > {policy['min_f1']:.2f}, precision > {policy['min_precision']:.2f}, "
            f"recall > {policy['min_recall']:.2f}"
        )
    else:
        threshold_note = f"recall >= {min_recall:.2f}, maximize precision"

    summary = {
        "meta_variant": str(cfg.get("meta_variant", "logistic")),
        "granularity": granularity,
        "fit_split": "valid",
        "evidence_split": "test",
        "selected_method": winner_method,
        "threshold_policy": threshold_policy or {"name": "max_precision_at_recall", "min_recall": min_recall},
        "min_recall_floor": min_recall,
        "raw_deployment_threshold": raw_deployment.get("deployment_threshold"),
        "raw_test_at_deployment": raw_deployment.get("test_at_deployment_threshold"),
        "deployment_threshold_calibrated": deploy_threshold,
        "valid_at_deployment_threshold": winner["valid_at_deployment_threshold"],
        "test_at_deployment_threshold": winner["test"]["at_deployment_threshold"],
        "methods": [
            {
                "method": row["method"],
                "valid": row["valid"],
                "test": {
                    "raw": row["test"]["raw"],
                    "calibrated": row["test"]["calibrated"],
                    "at_deployment_threshold": row["test"]["at_deployment_threshold"],
                },
                "deployment_threshold_calibrated": row["deployment_threshold_calibrated"],
            }
            for row in method_results
        ],
        "paper_statement": (
            "Probability scores from the locked logistic meta-learner are post-hoc calibrated "
            f"with {winner_method} regression fit on the full validation set (standard single-score practice). "
            "Calibration quality is evidenced by Brier score and expected calibration error on the held-out test set. "
            f"The deployment threshold is selected on calibrated validation {unit} using "
            f"({threshold_note})."
        ),
        "artifacts": {
            "score_calibrator": str(calibrator_path),
            "calibration_report": str(results_dir / "calibration_report.json"),
            "calibration_comparison": str(results_dir / "calibration_comparison.csv"),
            "valid_calibrated_scores": str(variant_dir / "valid_calibrated_scores.parquet"),
            "test_calibrated_scores": str(variant_dir / "test_calibrated_scores.parquet"),
        },
    }

    # Drop calibrator objects before JSON serialization in nested method summaries
    for row in summary["methods"]:
        row.pop("calibrator", None)

    report_path = results_dir / "calibration_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    deployment_path = variant_dir / "calibrated_deployment.json"
    deployment_payload: dict[str, Any] = {
        "selected_method": winner_method,
        "deployment_threshold_calibrated": deploy_threshold,
        "raw_deployment_threshold": raw_deployment.get("deployment_threshold"),
        "valid_at_deployment_threshold": winner["valid_at_deployment_threshold"],
        "test_at_deployment_threshold": winner["test"]["at_deployment_threshold"],
        "test_calibration_evidence": winner["test"]["calibrated"],
        "test_calibration_raw_baseline": winner["test"]["raw"],
    }
    if threshold_policy:
        deployment_payload["threshold_policy"] = threshold_policy
    deployment_path.write_text(json.dumps(deployment_payload, indent=2), encoding="utf-8")

    sync_score_config(config_path, winner_method, deploy_threshold)

    winner_row = comparison_frame.loc[comparison_frame["method"] == winner_method].iloc[0]
    report_lines = [
        "Vulnera — score calibration (05_SCORE)",
        "",
        summary["paper_statement"],
        "",
        f"Selected method: {winner_method}",
        f"Fit on: full valid ({len(y_valid):,} {unit})",
        f"Calibration evidence: test Brier/ECE ({len(y_test):,} {unit})",
        "",
        f"{'Method':<10} {'Test Brier raw':>14} {'Test Brier cal':>14} {'Test ECE raw':>12} {'Test ECE cal':>12}",
        "-" * 66,
    ]
    for _, comp in comparison_frame.iterrows():
        report_lines.append(
            f"{comp['method']:<10} {comp['test_brier_raw']:>14.4f} {comp['test_brier_calibrated']:>14.4f} "
            f"{comp['test_ece_raw']:>12.4f} {comp['test_ece_calibrated']:>12.4f}"
        )
    report_lines.extend(
        [
            "",
            f"Deployment threshold (calibrated valid): {deploy_threshold:.4f}",
            f"Raw deployment threshold (reference): {raw_deployment.get('deployment_threshold', 'n/a')}",
            "",
            "Test @ calibrated deployment threshold:",
            f"  precision={winner_row['test_precision']:.4f}  recall={winner_row['test_recall']:.4f}  "
            f"f1={winner_row['test_f1']:.4f}",
            "",
            f"Calibrator: {calibrator_path}",
            f"Report:     {report_path}",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    (results_dir / "calibration_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
