"""
Random search for logistic meta-learner hyperparameters on stacked base probabilities.

Deployment selection per trial (valid):
  precision > 0.4, recall in (0.6, 0.7], f1 > 0.5

Usage:
    python 04_META/training_scripts/tune_meta_logistic.py --config meta_config.yaml --trials 72
    python 04_META/training_scripts/tune_meta_logistic.py --trials 36
"""

from __future__ import annotations

import importlib.util
import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

SCRIPTS_ROOT = Path(__file__).resolve().parent
META_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = META_ROOT.parent
DEFAULT_CONFIG_PATH = META_ROOT / "meta_config.yaml"
RANDOM_SEED = 42

_spec = importlib.util.spec_from_file_location("train_trees", PROJECT_ROOT / "03_TREE" / "training_scripts" / "train_trees.py")
_train_trees = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train_trees)

from meta_common import (  # noqa: E402
    META_ROOT as _META_ROOT,
    build_split_table,
    load_config,
    results_output_dir,
    select_threshold_precision_recall_f1,
    sweep_threshold_curve,
)

SEARCH_SPACE: dict[str, list[Any]] = {
    "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "class_weight": [None, "balanced"],
    "max_iter": [1000, 2000, 3000],
    "solver": ["lbfgs", "liblinear", "saga"],
}


def sample_params(rng: random.Random) -> dict[str, Any]:
    params = {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}
    if params["solver"] == "liblinear" and params["class_weight"] is None:
        params["class_weight"] = "balanced"
    return params


def build_model(params: dict[str, Any]) -> LogisticRegression:
    return LogisticRegression(
        C=float(params["C"]),
        class_weight=params["class_weight"],
        max_iter=int(params["max_iter"]),
        solver=str(params["solver"]),
        random_state=RANDOM_SEED,
    )


def evaluate_model(
    model: LogisticRegression,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    valid_prob = model.predict_proba(x_valid)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    return {
        "threshold": threshold,
        "valid": _train_trees.compute_metrics(y_valid, valid_prob, threshold=threshold),
        "test": _train_trees.compute_metrics(y_test, test_prob, threshold=threshold),
    }


def default_params(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "C": float(cfg.get("C", 1.0)),
        "class_weight": cfg.get("class_weight", "balanced"),
        "max_iter": int(cfg.get("max_iter", 1000)),
        "solver": str(cfg.get("solver", "lbfgs")),
    }


def trial_sort_key(row: dict[str, Any]) -> tuple:
    valid_at = row["valid_at_threshold"]
    return (
        row.get("meets_policy", False),
        valid_at.get("f1", 0.0),
        valid_at.get("precision", 0.0),
        -valid_at.get("deficit", 999.0) if "deficit" in valid_at else 0.0,
        -abs(valid_at.get("recall", 0.0) - 0.65),
    )


def pick_threshold_for_probs(
    y_valid: np.ndarray,
    valid_prob: np.ndarray,
    *,
    min_precision: float,
    min_recall: float,
    max_recall: float,
    min_f1: float,
    step: float,
) -> tuple[float, dict[str, float], bool]:
    curve = sweep_threshold_curve(y_valid, valid_prob, step=step)
    threshold, metrics, meets = select_threshold_precision_recall_f1(
        curve,
        min_precision=min_precision,
        min_recall=min_recall,
        max_recall=max_recall,
        min_f1=min_f1,
    )
    return threshold, metrics, meets


def main() -> None:
    parser = ArgumentParser(description="Tune logistic meta-learner on stacked base probs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--trials", type=int, default=36)
    parser.add_argument("--min-precision", type=float, default=0.4)
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--max-recall", type=float, default=0.7)
    parser.add_argument("--min-f1", type=float, default=0.5)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else _META_ROOT / args.config
    config = load_config(config_path)
    settings = config.get("train_meta", {})
    feature_columns = [str(col) for col in settings.get("feature_columns", [])]
    label_column = str(settings.get("label_column", "label"))
    lr_cfg = settings.get("logistic_regression", {})
    granularity = str(settings.get("granularity", "function"))

    print("\nLoading stacked base predictions (this may take a few minutes for window granularity)...", flush=True)
    valid_frame = build_split_table(settings, "valid", feature_columns, label_column)
    test_frame = build_split_table(settings, "test", feature_columns, label_column)
    x_valid = valid_frame[feature_columns].to_numpy(dtype=np.float32)
    y_valid = valid_frame[label_column].astype(int).to_numpy()
    x_test = test_frame[feature_columns].to_numpy(dtype=np.float32)
    y_test = test_frame[label_column].astype(int).to_numpy()

    trials = min(args.trials, 8) if args.smoke_test else args.trials
    if args.smoke_test:
        x_valid, y_valid = x_valid[:8000], y_valid[:8000]
        x_test, y_test = x_test[:8000], y_test[:8000]

    rng = random.Random(RANDOM_SEED)
    policy = {
        "min_precision": args.min_precision,
        "min_recall": args.min_recall,
        "max_recall": args.max_recall,
        "min_f1": args.min_f1,
    }

    print(
        f"\nLogistic meta tuning [{granularity}] | valid={len(y_valid):,} test={len(y_test):,} | trials={trials}",
        flush=True,
    )
    print(
        f"Policy: P>{args.min_precision}, {args.min_recall}<R<={args.max_recall}, F1>{args.min_f1}\n",
        flush=True,
    )

    baseline_params = default_params(lr_cfg)
    baseline_model = build_model(baseline_params)
    baseline_model.fit(x_valid, y_valid)
    baseline_valid_prob = baseline_model.predict_proba(x_valid)[:, 1]
    baseline_threshold, baseline_valid_at, baseline_meets = pick_threshold_for_probs(
        y_valid,
        baseline_valid_prob,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        max_recall=args.max_recall,
        min_f1=args.min_f1,
        step=args.threshold_step,
    )
    baseline_metrics = evaluate_model(
        baseline_model, x_valid, y_valid, x_test, y_test, threshold=baseline_threshold
    )
    print(
        f"Baseline valid P={baseline_valid_at.get('precision', baseline_metrics['valid']['precision']):.4f} "
        f"R={baseline_valid_at.get('recall', baseline_metrics['valid']['recall']):.4f} "
        f"F1={baseline_valid_at.get('f1', baseline_metrics['valid']['f1']):.4f} @ t={baseline_threshold:.2f} "
        f"meets={baseline_meets} | test F1={baseline_metrics['test']['f1']:.4f}",
        flush=True,
    )

    best_row: dict[str, Any] = {
        "params": baseline_params,
        "threshold": baseline_threshold,
        "valid_at_threshold": baseline_valid_at,
        "meets_policy": baseline_meets,
        "valid": baseline_metrics["valid"],
        "test": baseline_metrics["test"],
    }
    trial_rows: list[dict[str, Any]] = []

    for trial_idx in range(1, trials + 1):
        params = sample_params(rng)
        try:
            model = build_model(params)
            model.fit(x_valid, y_valid)
        except Exception as exc:
            print(f"Trial {trial_idx:>2}/{trials} | skipped ({exc})", flush=True)
            continue

        valid_prob = model.predict_proba(x_valid)[:, 1]
        threshold, valid_at, meets = pick_threshold_for_probs(
            y_valid,
            valid_prob,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
            max_recall=args.max_recall,
            min_f1=args.min_f1,
            step=args.threshold_step,
        )
        metrics = evaluate_model(model, x_valid, y_valid, x_test, y_test, threshold=threshold)
        row = {
            "trial": trial_idx,
            "params": params,
            "threshold": threshold,
            "valid_at_threshold": valid_at,
            "meets_policy": meets,
            "valid": metrics["valid"],
            "test": metrics["test"],
        }
        trial_rows.append(row)

        candidate_key = trial_sort_key(row)
        best_key = trial_sort_key(best_row)
        improved = candidate_key > best_key
        if improved:
            best_row = row

        flag = "  *best*" if improved else ""
        print(
            f"Trial {trial_idx:>2}/{trials} | valid P={valid_at.get('precision', 0):.4f} "
            f"R={valid_at.get('recall', 0):.4f} F1={valid_at.get('f1', 0):.4f} @ t={threshold:.2f} "
            f"meets={meets}{flag}",
            flush=True,
        )

    tuned_model = build_model(best_row["params"])
    tuned_model.fit(x_valid, y_valid)
    tuned_metrics = evaluate_model(
        tuned_model,
        x_valid,
        y_valid,
        x_test,
        y_test,
        threshold=float(best_row["threshold"]),
    )

    print(
        f"\nBest logistic meta: valid P={tuned_metrics['valid']['precision']:.4f} "
        f"R={tuned_metrics['valid']['recall']:.4f} F1={tuned_metrics['valid']['f1']:.4f} | "
        f"test P={tuned_metrics['test']['precision']:.4f} R={tuned_metrics['test']['recall']:.4f} "
        f"F1={tuned_metrics['test']['f1']:.4f} | threshold={best_row['threshold']:.2f} "
        f"meets_policy={best_row['meets_policy']}",
        flush=True,
    )
    print(f"Best params: {json.dumps(best_row['params'])}", flush=True)

    out_dir = results_output_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_name = "window_stack_meta_logistic_tune.json" if granularity == "window" else "meta_logistic_tune.json"
    summary = {
        "variant": "logistic",
        "granularity": granularity,
        "config": str(config_path.name),
        "trials": trials,
        "threshold_policy": policy,
        "baseline": {
            "params": baseline_params,
            "threshold": baseline_threshold,
            "meets_policy": baseline_meets,
            "valid_at_threshold": baseline_valid_at,
            "valid": baseline_metrics["valid"],
            "test": baseline_metrics["test"],
        },
        "tuned": {
            "params": best_row["params"],
            "threshold": float(best_row["threshold"]),
            "meets_policy": best_row["meets_policy"],
            "valid_at_threshold": best_row["valid_at_threshold"],
            "valid": tuned_metrics["valid"],
            "test": tuned_metrics["test"],
        },
        "all_trials": trial_rows,
    }
    summary_path = out_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
