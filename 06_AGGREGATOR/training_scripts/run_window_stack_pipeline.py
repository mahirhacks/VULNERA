"""
Train and deploy the window-stack aggregation pipeline.

Steps:
  1. Train four tree models on window embeddings
  2. Train logistic meta-learner on window-level base predictions
  3. Calibrate window-level meta scores
  4. Run aggregation report (max-pool to function)

Usage:
    python 06_AGGREGATOR/training_scripts/run_window_stack_pipeline.py
    python 06_AGGREGATOR/training_scripts/run_window_stack_pipeline.py --smoke-test
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
    parser = ArgumentParser(description="Window-stack train + calibrate + aggregate pipeline.")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    smoke = ["--smoke-test"] if args.smoke_test else []

    run([PYTHON, "03_TREE/training_scripts/train_window_stack_trees.py", *smoke])
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
    run(
        [
            PYTHON,
            "05_SCORE/training_scripts/calibrate_scores.py",
            "--config",
            "score_config.yaml",
        ]
    )
    run([PYTHON, "06_AGGREGATOR/training_scripts/run_aggregation.py", "--split", "valid"])
    run([PYTHON, "06_AGGREGATOR/training_scripts/run_aggregation.py", "--split", "test"])


if __name__ == "__main__":
    main()
