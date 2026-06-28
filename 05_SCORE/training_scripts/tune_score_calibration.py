"""
Tune score-calibration deployment policy: min_recall floor + isotonic vs platt.

Selects the combination with best validation F1 at the deployment threshold rule,
then writes min_recall into meta/score configs and runs calibrate_scores.py.

Usage:
    python 05_SCORE/training_scripts/tune_score_calibration.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

SCORE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCORE_ROOT.parent
META_ROOT = PROJECT_ROOT / "04_META"
PYTHON = sys.executable

from score_common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    brier_score,
    expected_calibration_error,
    import_meta_common,
    load_config,
    resolve_path,
)

_spec = importlib.util.spec_from_file_location(
    "train_trees", PROJECT_ROOT / "03_TREE" / "training_scripts" / "train_trees.py"
)
_train_trees = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train_trees)


def apply_calibrator(method: str, calibrator: Any, raw_scores: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        return np.clip(calibrator.predict(raw_scores), 0.0, 1.0).astype(np.float32)
    return calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1].astype(np.float32)


def sync_min_recall(meta_config_path: Path, score_config_path: Path, min_recall: float) -> None:
    with meta_config_path.open(encoding="utf-8") as handle:
        meta_cfg = yaml.safe_load(handle)
    meta_cfg.setdefault("calibrate_threshold", {})["min_recall"] = float(min_recall)
    with meta_config_path.open("w", encoding="utf-8") as handle:
        yaml.dump(meta_cfg, handle, sort_keys=False, default_flow_style=False)

    with score_config_path.open(encoding="utf-8") as handle:
        score_cfg = yaml.safe_load(handle)
    score_cfg.setdefault("calibrate_scores", {})["min_recall"] = float(min_recall)
    with score_config_path.open("w", encoding="utf-8") as handle:
        yaml.dump(score_cfg, handle, sort_keys=False, default_flow_style=False)


def main() -> None:
    parser = ArgumentParser(description="Tune min_recall for calibrated deployment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    meta_common = import_meta_common()
    score_config_path = args.config if args.config.is_absolute() else SCORE_ROOT / args.config
    score_cfg = load_config(score_config_path).get("calibrate_scores", {})
    meta_config_path = resolve_path(str(score_cfg.get("meta_config", "04_META/meta_config.yaml")))
    meta_config = meta_common.load_config(meta_config_path)
    settings = meta_config.get("train_meta", {})
    feature_columns = [str(col) for col in settings.get("feature_columns", [])]
    label_column = str(settings.get("label_column", "label"))
    step = float(score_cfg.get("threshold_step", 0.01))
    ece_bins = int(score_cfg.get("ece_bins", 10))
    random_state = int(score_cfg.get("random_state", 42))

    valid_frame = meta_common.build_split_table(settings, "valid", feature_columns, label_column)
    x_valid = valid_frame[feature_columns].to_numpy(dtype=np.float32)
    y_valid = valid_frame[label_column].astype(int).to_numpy()

    meta_model = joblib.load(resolve_path(str(score_cfg.get("meta_model", "04_META/logistic/meta_model.joblib"))))
    raw_valid = meta_model.predict_proba(x_valid)[:, 1].astype(np.float32)

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_valid, y_valid)
    platt = LogisticRegression(max_iter=1000, random_state=random_state)
    platt.fit(raw_valid.reshape(-1, 1), y_valid)

    min_recall_grid = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for min_recall in min_recall_grid:
        for method, calibrator in [("isotonic", isotonic), ("platt", platt)]:
            valid_cal = apply_calibrator(method, calibrator, raw_valid)
            curve = meta_common.sweep_threshold_curve(y_valid, valid_cal, step=step)
            deploy_t, valid_at_deploy = meta_common.select_threshold_max_precision_at_recall(curve, min_recall)
            row = {
                "min_recall": min_recall,
                "method": method,
                "deployment_threshold": deploy_t,
                "valid_f1": valid_at_deploy["f1"],
                "valid_precision": valid_at_deploy["precision"],
                "valid_recall": valid_at_deploy["recall"],
                "brier": brier_score(y_valid, valid_cal),
                "ece": expected_calibration_error(y_valid, valid_cal, n_bins=ece_bins),
            }
            rows.append(row)
            if best is None or row["valid_f1"] > best["valid_f1"]:
                best = row

    assert best is not None
    out_dir = resolve_path(str(score_cfg.get("output_dir", "05_SCORE/results")))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"grid": rows, "selected": best}
    summary_path = out_dir / "score_calibration_tune.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sync_min_recall(meta_config_path, score_config_path, float(best["min_recall"]))
    with score_config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg.setdefault("calibrate_scores", {})["selected_method"] = str(best["method"])
    with score_config_path.open("w", encoding="utf-8") as handle:
        yaml.dump(cfg, handle, sort_keys=False, default_flow_style=False)

    print(f"Selected min_recall={best['min_recall']:.2f} method={best['method']} valid F1={best['valid_f1']:.4f}")
    print(f"Wrote {summary_path}")
    print("Running calibrate_scores.py ...")
    subprocess.run([PYTHON, "05_SCORE/training_scripts/calibrate_scores.py"], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
