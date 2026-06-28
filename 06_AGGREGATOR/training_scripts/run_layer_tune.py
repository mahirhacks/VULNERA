"""
Layer-by-layer tuning: meta learners -> calibration (aggregator unchanged).

Phase 1: Compare meta learners on VALID (trees fixed); pick winner by AUC then AP then score_std.
Phase 2: Compare calibration methods on winner raw scores; pick by valid Brier then AUC preservation.
Phase 3: Save artifacts + update aggregator/score configs; run test aggregation summary.

Usage:
    python 06_AGGREGATOR/training_scripts/run_layer_tune.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPTS_ROOT = Path(__file__).resolve().parent
AGGREGATOR_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = AGGREGATOR_ROOT.parent
META_SCRIPTS = PROJECT_ROOT / "04_META" / "training_scripts"
SCORE_SCRIPTS = PROJECT_ROOT / "05_SCORE" / "training_scripts"

sys.path.insert(0, str(META_SCRIPTS))
sys.path.insert(0, str(SCORE_SCRIPTS))
sys.path.insert(0, str(SCRIPTS_ROOT))

from aggregator_common import import_meta_common, import_train_trees, load_config, resolve_path  # noqa: E402
from meta_learners import (  # noqa: E402
    IdentityCalibrator,
    meta_learner_catalog,
    save_meta_model,
)
from score_common import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    select_threshold_balanced,
    threshold_policy_from_config,
)

RESULTS_DIR = AGGREGATOR_ROOT / "results" / "layer_tune"
SELECTED_META_DIR = PROJECT_ROOT / "04_META" / "window_stack" / "selected"
SELECTED_SCORE_DIR = PROJECT_ROOT / "05_SCORE" / "window_stack" / "selected"

THRESHOLD_POLICY = {
    "name": "balanced",
    "min_f1": 0.5,
    "min_precision": 0.4,
    "min_recall": 0.6,
}
THRESHOLD_STEP = 0.01
ECE_BINS = 10


def _metrics(y: np.ndarray, scores: np.ndarray, *, threshold: float) -> dict[str, float]:
    train_trees = import_train_trees()
    base = train_trees.compute_metrics(y, scores, threshold=threshold)
    base["roc_auc"] = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
    base["avg_precision"] = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
    base["score_std"] = float(np.std(scores))
    base["score_min"] = float(np.min(scores))
    base["score_max"] = float(np.max(scores))
    base["brier"] = brier_score(y, scores)
    base["ece"] = expected_calibration_error(y, scores, n_bins=ECE_BINS)
    return base


def _threshold_metrics(meta_common: Any, y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    curve = meta_common.sweep_threshold_curve(y, scores, step=THRESHOLD_STEP)
    threshold, at_threshold = select_threshold_balanced(
        curve,
        min_f1=THRESHOLD_POLICY["min_f1"],
        min_precision=THRESHOLD_POLICY["min_precision"],
        min_recall=THRESHOLD_POLICY["min_recall"],
    )
    return {"threshold": threshold, "at_threshold": at_threshold}


def _predict_meta(model: Any, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    return model.predict_proba(x)[:, 1].astype(np.float32)


def tune_meta_learners(
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> tuple[pd.DataFrame, str, Any]:
    y_valid = valid_frame[label_column].astype(int).to_numpy()
    y_test = test_frame[label_column].astype(int).to_numpy()
    meta_common = import_meta_common()

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for name, desc, prototype in meta_learner_catalog(feature_columns):
        t0 = time.perf_counter()
        # fresh instance per candidate (avoid shared state)
        model = next(m for n, _, m in meta_learner_catalog(feature_columns) if n == name)
        model.fit(valid_frame, label_column=label_column)
        valid_scores = _predict_meta(model, valid_frame, feature_columns)
        test_scores = _predict_meta(model, test_frame, feature_columns)
        valid_thr = _threshold_metrics(meta_common, y_valid, valid_scores)
        test_thr = _threshold_metrics(meta_common, y_test, test_scores)
        elapsed = time.perf_counter() - t0

        valid_m = _metrics(y_valid, valid_scores, threshold=valid_thr["threshold"])
        test_m = _metrics(y_test, test_scores, threshold=test_thr["threshold"])

        rows.append(
            {
                "meta_name": name,
                "description": desc,
                "valid_auc": valid_m["roc_auc"],
                "valid_ap": valid_m["avg_precision"],
                "valid_f1": valid_thr["at_threshold"].get("f1", 0.0),
                "valid_precision": valid_thr["at_threshold"].get("precision", 0.0),
                "valid_recall": valid_thr["at_threshold"].get("recall", 0.0),
                "valid_meets_policy": valid_thr["at_threshold"].get("meets_policy", False),
                "valid_score_std": valid_m["score_std"],
                "valid_brier": valid_m["brier"],
                "test_auc": test_m["roc_auc"],
                "test_ap": test_m["avg_precision"],
                "test_f1": test_thr["at_threshold"].get("f1", 0.0),
                "test_precision": test_thr["at_threshold"].get("precision", 0.0),
                "test_recall": test_thr["at_threshold"].get("recall", 0.0),
                "test_score_std": test_m["score_std"],
                "seconds": elapsed,
            }
        )
        fitted[name] = model
        print(f"  [{name}] valid AUC={valid_m['roc_auc']:.4f} std={valid_m['score_std']:.4f} ({elapsed:.1f}s)")

    report = pd.DataFrame(rows).sort_values(
        ["valid_auc", "valid_ap", "valid_score_std", "valid_meets_policy"],
        ascending=[False, False, False, False],
    )
    winner = str(report.iloc[0]["meta_name"])
    print(f"\nMeta winner: {winner}")
    return report, winner, fitted[winner]


def fit_calibrator(method: str, raw_valid: np.ndarray, y_valid: np.ndarray) -> Any:
    if method == "none":
        return IdentityCalibrator()
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(raw_valid, y_valid)
        return cal
    if method == "platt":
        cal = LogisticRegression(max_iter=1000, random_state=42)
        cal.fit(raw_valid.reshape(-1, 1), y_valid)
        return cal
    raise ValueError(method)


def apply_cal(method: str, calibrator: Any, raw: np.ndarray) -> np.ndarray:
    if method == "none":
        return np.clip(calibrator.predict(raw), 0.0, 1.0).astype(np.float32)
    if method == "isotonic":
        return np.clip(calibrator.predict(raw), 0.0, 1.0).astype(np.float32)
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1].astype(np.float32)


def tune_calibration(
    meta_model: Any,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    y_valid = valid_frame[label_column].astype(int).to_numpy()
    y_test = test_frame[label_column].astype(int).to_numpy()
    meta_common = import_meta_common()

    raw_valid = _predict_meta(meta_model, valid_frame, feature_columns)
    raw_test = _predict_meta(meta_model, test_frame, feature_columns)

    methods = ["none", "platt", "isotonic"]
    rows: list[dict[str, Any]] = []
    bundles: dict[str, dict[str, Any]] = {}

    for method in methods:
        calibrator = fit_calibrator(method, raw_valid, y_valid)
        valid_cal = apply_cal(method, calibrator, raw_valid)
        test_cal = apply_cal(method, calibrator, raw_test)

        valid_thr = _threshold_metrics(meta_common, y_valid, valid_cal)
        test_thr = _threshold_metrics(meta_common, y_test, test_cal)
        valid_m = _metrics(y_valid, valid_cal, threshold=valid_thr["threshold"])
        test_m = _metrics(y_test, test_cal, threshold=test_thr["threshold"])

        rows.append(
            {
                "calibration": method,
                "valid_brier": valid_m["brier"],
                "valid_ece": valid_m["ece"],
                "valid_auc": valid_m["roc_auc"],
                "valid_score_std": valid_m["score_std"],
                "valid_meets_policy": valid_thr["at_threshold"].get("meets_policy", False),
                "valid_f1": valid_thr["at_threshold"].get("f1", 0.0),
                "valid_precision": valid_thr["at_threshold"].get("precision", 0.0),
                "valid_recall": valid_thr["at_threshold"].get("recall", 0.0),
                "deployment_threshold": valid_thr["threshold"],
                "test_auc": test_m["roc_auc"],
                "test_brier": test_m["brier"],
                "test_ece": test_m["ece"],
                "test_score_std": test_m["score_std"],
                "test_f1": test_thr["at_threshold"].get("f1", 0.0),
                "test_precision": test_thr["at_threshold"].get("precision", 0.0),
                "test_recall": test_thr["at_threshold"].get("recall", 0.0),
            }
        )
        bundles[method] = {
            "method": method,
            "calibrator": calibrator,
            "deployment_threshold_calibrated": valid_thr["threshold"],
            "valid_at_deployment_threshold": valid_thr["at_threshold"],
            "test_at_deployment_threshold": test_thr["at_threshold"],
            "threshold_policy": THRESHOLD_POLICY,
        }
        print(
            f"  [{method}] valid Brier={valid_m['brier']:.4f} AUC={valid_m['roc_auc']:.4f} "
            f"std={valid_m['score_std']:.4f} policy={valid_thr['at_threshold'].get('meets_policy')}"
        )

    report = pd.DataFrame(rows).sort_values(
        ["valid_meets_policy", "valid_brier", "valid_auc", "valid_score_std"],
        ascending=[False, True, False, False],
    )
    winner = str(report.iloc[0]["calibration"])
    print(f"\nCalibration winner: {winner}")
    return report, winner, bundles[winner]


def apply_artifacts(
    meta_name: str,
    meta_model: Any,
    cal_bundle: dict[str, Any],
    *,
    meta_report: pd.DataFrame,
    cal_report: pd.DataFrame,
) -> None:
    SELECTED_META_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_SCORE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = SELECTED_META_DIR / "meta_model.joblib"
    cal_path = SELECTED_SCORE_DIR / "score_calibrator.joblib"
    deploy_path = SELECTED_SCORE_DIR / "calibrated_deployment.json"

    save_meta_model(meta_model, meta_path, name=meta_name)
    joblib.dump(
        {"method": cal_bundle["method"], "calibrator": cal_bundle["calibrator"]},
        cal_path,
    )
    deploy_payload = {
        "selected_meta": meta_name,
        "selected_method": cal_bundle["method"],
        "deployment_threshold_calibrated": cal_bundle["deployment_threshold_calibrated"],
        "threshold_policy": cal_bundle["threshold_policy"],
        "valid_at_deployment_threshold": cal_bundle["valid_at_deployment_threshold"],
        "test_at_deployment_threshold": cal_bundle["test_at_deployment_threshold"],
    }
    deploy_path.write_text(json.dumps(deploy_payload, indent=2), encoding="utf-8")

    meta_report.to_csv(RESULTS_DIR / "meta_comparison.csv", index=False)
    cal_report.to_csv(RESULTS_DIR / "calibration_comparison.csv", index=False)
    summary = {
        "selected_meta": meta_name,
        "selected_calibration": cal_bundle["method"],
        "meta_model_path": str(meta_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "score_calibrator_path": str(cal_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "calibrated_deployment_path": str(deploy_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "meta_top3": meta_report.head(3).to_dict(orient="records"),
        "calibration_all": cal_report.to_dict(orient="records"),
    }
    (RESULTS_DIR / "layer_tune_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    agg_cfg_path = AGGREGATOR_ROOT / "aggregator_config.yaml"
    with agg_cfg_path.open(encoding="utf-8") as handle:
        agg_cfg = yaml.safe_load(handle)
    agg_cfg["run_aggregator"]["meta_model"] = "04_META/window_stack/selected/meta_model.joblib"
    agg_cfg["run_aggregator"]["score_calibrator"] = "05_SCORE/window_stack/selected/score_calibrator.joblib"
    agg_cfg["run_aggregator"]["calibrated_deployment"] = "05_SCORE/window_stack/selected/calibrated_deployment.json"
    with agg_cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(agg_cfg, handle, sort_keys=False)

    score_cfg_path = PROJECT_ROOT / "05_SCORE" / "score_config.yaml"
    with score_cfg_path.open(encoding="utf-8") as handle:
        score_cfg = yaml.safe_load(handle)
    block = score_cfg.setdefault("calibrate_scores", {})
    block["meta_model"] = "04_META/window_stack/selected/meta_model.joblib"
    block["variant_dir"] = "05_SCORE/window_stack/selected"
    block["selected_method"] = cal_bundle["method"]
    block["deployment_threshold_calibrated"] = float(cal_bundle["deployment_threshold_calibrated"])
    with score_cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(score_cfg, handle, sort_keys=False)

    print(f"\nApplied artifacts:")
    print(f"  meta     -> {meta_path}")
    print(f"  calibrator -> {cal_path}")
    print(f"  deployment -> {deploy_path}")
    print(f"  aggregator_config.yaml updated")


def main() -> None:
    meta_common = import_meta_common()
    meta_config = meta_common.load_config(PROJECT_ROOT / "04_META" / "meta_config.yaml")
    settings = meta_config["train_meta"]
    feature_columns = [str(c) for c in settings["feature_columns"]]
    label_column = str(settings.get("label_column", "label"))

    print("Loading window-level tree probability tables (trees fixed)...")
    valid_frame = meta_common.build_window_split_table(settings, "valid", feature_columns, label_column)
    test_frame = meta_common.build_window_split_table(settings, "test", feature_columns, label_column)
    print(f"  valid={len(valid_frame):,}  test={len(test_frame):,}  pos_rate={valid_frame[label_column].mean():.3f}")

    print("\n=== PHASE 1: META LEARNER COMPARISON (select on VALID) ===")
    meta_report, meta_winner, meta_model = tune_meta_learners(
        valid_frame, test_frame, feature_columns, label_column
    )

    print("\n=== PHASE 2: CALIBRATION COMPARISON (select on VALID) ===")
    cal_report, cal_winner, cal_bundle = tune_calibration(
        meta_model, valid_frame, test_frame, feature_columns, label_column
    )

    print("\n=== PHASE 3: SAVE & WIRE CONFIG ===")
    apply_artifacts(meta_winner, meta_model, cal_bundle, meta_report=meta_report, cal_report=cal_report)

    print("\n=== TOP META (valid) ===")
    print(meta_report.head(8).to_string(index=False))
    print("\n=== CALIBRATION ===")
    print(cal_report.to_string(index=False))
    print(f"\nDone. Reports in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
