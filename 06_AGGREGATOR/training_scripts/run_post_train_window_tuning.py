"""
Post full window-stack training: hyperparameter search, retrain, calibrate, threshold tune.

End state:
  - Window: calibrated stack score with balanced threshold (F1>0.5, P>0.4, R>0.6)
  - Function: max(window_prob) + spread uplift at tau
  - File: max(function_score) + spread uplift at tau (scan pipeline)

Steps:
  1. Tune xgboost + lightgbm on full train (144 trials each)
  2. Tune random_forest + extra_trees on 50k train rows (72 trials each)
  3. Apply tuned params to tree_config.yaml
  4. Retrain all four window-stack trees
  5. Retrain meta + calibrate window scores (balanced threshold policy)
  6. Re-tune deployment threshold on live stack scores (same policy)
  7. Aggregation reports on valid + test (window / function / file formula)

Usage:
    python 06_AGGREGATOR/training_scripts/run_post_train_window_tuning.py
    python 06_AGGREGATOR/training_scripts/run_post_train_window_tuning.py --smoke-test
    python 06_AGGREGATOR/training_scripts/run_post_train_window_tuning.py --skip-tune --skip-retrain
"""

from __future__ import annotations

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = ArgumentParser(description="Post-train window-stack tuning pipeline.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--skip-apply", action="store_true")
    parser.add_argument("--skip-retrain", action="store_true")
    parser.add_argument("--skip-meta", action="store_true")
    parser.add_argument("--skip-calibrate", action="store_true")
    parser.add_argument("--skip-threshold", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    smoke = ["--smoke-test"] if args.smoke_test else []
    xgb_trials = "6" if args.smoke_test else "144"
    lgbm_trials = "6" if args.smoke_test else "144"
    rf_trials = "4" if args.smoke_test else "72"
    et_trials = "4" if args.smoke_test else "72"
    xgb_sample = "low" if args.smoke_test else "full"
    lgbm_sample = "low" if args.smoke_test else "full"
    rf_sample = "low" if args.smoke_test else "med"
    et_sample = "low" if args.smoke_test else "med"

    if not args.skip_tune:
        run(
            [
                PYTHON,
                "03_TREE/training_scripts/tune_window_stack_model.py",
                "--model",
                "xgboost",
                "--trials",
                xgb_trials,
                "--train-sample",
                xgb_sample,
                *smoke,
            ]
        )
        run(
            [
                PYTHON,
                "03_TREE/training_scripts/tune_window_stack_model.py",
                "--model",
                "lightgbm",
                "--trials",
                lgbm_trials,
                "--train-sample",
                lgbm_sample,
                *smoke,
            ]
        )
        run(
            [
                PYTHON,
                "03_TREE/training_scripts/tune_window_stack_model.py",
                "--model",
                "random_forest",
                "--trials",
                rf_trials,
                "--train-sample",
                rf_sample,
                *smoke,
            ]
        )
        run(
            [
                PYTHON,
                "03_TREE/training_scripts/tune_window_stack_model.py",
                "--model",
                "extra_trees",
                "--trials",
                et_trials,
                "--train-sample",
                et_sample,
                *smoke,
            ]
        )

    if not args.skip_apply:
        run([PYTHON, "03_TREE/training_scripts/apply_window_stack_tree_tuning.py"])

    if not args.skip_retrain:
        run([PYTHON, "03_TREE/training_scripts/train_window_stack_trees.py", *smoke])

    if not args.skip_meta:
        run(
            [
                PYTHON,
                "04_META/training_scripts/train_meta.py",
                "--config",
                "meta_config.yaml",
                "--variant",
                "logistic",
            ]
        )

    if not args.skip_calibrate:
        run(
            [
                PYTHON,
                "05_SCORE/training_scripts/calibrate_scores.py",
                "--config",
                "score_config.yaml",
            ]
        )

    if not args.skip_threshold:
        run([PYTHON, "05_SCORE/training_scripts/tune_window_stack_deployment.py"])

    if not args.skip_report:
        run([PYTHON, "06_AGGREGATOR/training_scripts/run_aggregation.py", "--split", "valid"])
        run([PYTHON, "06_AGGREGATOR/training_scripts/run_aggregation.py", "--split", "test"])


if __name__ == "__main__":
    main()
