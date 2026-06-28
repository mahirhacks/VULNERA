"""
Random hyperparameter search for window-stack tree models (per-window embeddings).

Usage:
    python 03_TREE/training_scripts/tune_window_stack_model.py --model xgboost --trials 144 --train-sample full
    python 03_TREE/training_scripts/tune_window_stack_model.py --model lightgbm --trials 144 --train-sample full
    python 03_TREE/training_scripts/tune_window_stack_model.py --model random_forest --trials 72 --train-sample med
    python 03_TREE/training_scripts/tune_window_stack_model.py --model extra_trees --trials 72 --train-sample med
"""

from __future__ import annotations

import json
import random
from argparse import ArgumentParser
from typing import Any, Callable

import numpy as np
import yaml
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from tune_window_stack_common import (
    MODEL_CHOICES,
    POOL,
    RANDOM_SEED,
    TRAIN_SAMPLE_LIMITS,
    TrainSample,
    _train_trees,
    best_f1_threshold,
    evaluate_at_threshold,
    load_window_splits,
    subsample_train_windows,
    tree_config_section,
    tune_summary_path,
)

TREE_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

XGB_SEARCH: dict[str, list[Any]] = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_lambda": [1.0, 5.0, 10.0],
    "min_child_weight": [1, 5, 10],
    "scale_pos_weight": [1.5, 2.0, 2.334, 3.0, 4.0],
    "gamma": [0.0, 0.1, 1.0],
}

LGBM_SEARCH: dict[str, list[Any]] = {
    "num_leaves": [31, 64, 127],
    "max_depth": [-1, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_lambda": [1.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "min_child_samples": [5, 20, 50],
    "scale_pos_weight": [1.5, 2.0, 2.334, 3.0, 4.0],
    "min_split_gain": [0.0, 0.1, 1.0],
}

RF_SEARCH: dict[str, list[Any]] = {
    "max_depth": [6, 8, 12, -1],
    "n_estimators": [300, 500, 800],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": ["balanced", "balanced_subsample"],
    "max_samples": [0.7, 0.8, 1.0],
}

ET_SEARCH: dict[str, list[Any]] = {
    "max_depth": [6, 8, 12, -1],
    "n_estimators": [300, 500, 800],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": ["balanced", "balanced_subsample"],
    "min_impurity_decrease": [0.0, 0.001, 0.01],
    "max_samples": [0.7, 0.8, 1.0],
}


def load_tree_config() -> dict[str, Any]:
    with (TREE_ROOT / "tree_config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sample_params(rng: random.Random, space: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in space.items()}


def resolve_max_depth(value: Any) -> int | None:
    return None if value in (None, -1, "null") else int(value)


def build_xgb(params: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=1000,
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        min_child_weight=int(params["min_child_weight"]),
        scale_pos_weight=float(params["scale_pos_weight"]),
        gamma=float(params["gamma"]),
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
    )


def build_lgbm(params: dict[str, Any]) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=1000,
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        min_child_samples=int(params["min_child_samples"]),
        min_split_gain=float(params["min_split_gain"]),
        scale_pos_weight=float(params["scale_pos_weight"]),
        objective="binary",
        device="gpu",
        random_state=RANDOM_SEED,
        verbose=-1,
    )


def build_random_forest(params: dict[str, Any]) -> RandomForestClassifier:
    max_samples = float(params["max_samples"])
    bootstrap = max_samples < 1.0
    kwargs: dict[str, Any] = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": resolve_max_depth(params["max_depth"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "min_samples_split": int(params["min_samples_split"]),
        "max_features": params["max_features"],
        "class_weight": str(params["class_weight"]),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    if bootstrap:
        kwargs["bootstrap"] = True
        kwargs["max_samples"] = max_samples
    return RandomForestClassifier(**kwargs)


def build_extra_trees(params: dict[str, Any]) -> ExtraTreesClassifier:
    max_samples = float(params["max_samples"])
    bootstrap = max_samples < 1.0
    kwargs: dict[str, Any] = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": resolve_max_depth(params["max_depth"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "min_samples_split": int(params["min_samples_split"]),
        "max_features": params["max_features"],
        "class_weight": str(params["class_weight"]),
        "min_impurity_decrease": float(params["min_impurity_decrease"]),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    if bootstrap:
        kwargs["bootstrap"] = True
        kwargs["max_samples"] = max_samples
    return ExtraTreesClassifier(**kwargs)


def default_xgb_params(cfg: dict[str, Any], auto_weight: float) -> dict[str, Any]:
    weight = cfg.get("scale_pos_weight")
    return {
        "max_depth": int(cfg.get("max_depth", 6)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "subsample": float(cfg.get("subsample", 0.8)),
        "colsample_bytree": float(cfg.get("colsample_bytree", 0.9)),
        "reg_lambda": float(cfg.get("reg_lambda", 5.0)),
        "min_child_weight": int(cfg.get("min_child_weight", 1)),
        "scale_pos_weight": float(weight if weight is not None else auto_weight),
        "gamma": float(cfg.get("gamma", 0.1)),
    }


def default_lgbm_params(cfg: dict[str, Any], auto_weight: float) -> dict[str, Any]:
    weight = cfg.get("scale_pos_weight")
    return {
        "num_leaves": int(cfg.get("num_leaves", 64)),
        "max_depth": int(cfg.get("max_depth", 6)),
        "learning_rate": float(cfg.get("learning_rate", 0.1)),
        "subsample": float(cfg.get("subsample", 0.8)),
        "colsample_bytree": float(cfg.get("colsample_bytree", 0.8)),
        "reg_lambda": float(cfg.get("reg_lambda", 10.0)),
        "reg_alpha": float(cfg.get("reg_alpha", 0.0)),
        "min_child_samples": int(cfg.get("min_child_samples", 50)),
        "min_split_gain": float(cfg.get("min_split_gain", 0.0)),
        "scale_pos_weight": float(weight if weight is not None else auto_weight),
    }


def default_rf_params(cfg: dict[str, Any]) -> dict[str, Any]:
    max_depth = cfg.get("max_depth")
    return {
        "max_depth": -1 if max_depth in (None, "null") else int(max_depth),
        "n_estimators": int(cfg.get("n_estimators", 500)),
        "min_samples_leaf": int(cfg.get("min_samples_leaf", 5)),
        "min_samples_split": int(cfg.get("min_samples_split", 5)),
        "max_features": cfg.get("max_features", 0.5),
        "class_weight": str(cfg.get("class_weight", "balanced_subsample")),
        "max_samples": float(cfg.get("max_samples", 0.7)),
    }


def default_et_params(cfg: dict[str, Any]) -> dict[str, Any]:
    max_depth = cfg.get("max_depth")
    return {
        "max_depth": -1 if max_depth in (None, "null") else int(max_depth),
        "n_estimators": int(cfg.get("n_estimators", 800)),
        "min_samples_leaf": int(cfg.get("min_samples_leaf", 5)),
        "min_samples_split": int(cfg.get("min_samples_split", 5)),
        "max_features": cfg.get("max_features", 0.5),
        "class_weight": str(cfg.get("class_weight", "balanced_subsample")),
        "min_impurity_decrease": float(cfg.get("min_impurity_decrease", 0.0)),
        "max_samples": float(cfg.get("max_samples", 0.7)),
    }


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "search": XGB_SEARCH,
        "device": "cuda",
        "build": build_xgb,
        "default": default_xgb_params,
        "fit": lambda m, xt, yt, xv, yv: m.fit(xt, yt, eval_set=[(xv, yv)], verbose=False),
    },
    "lightgbm": {
        "search": LGBM_SEARCH,
        "device": "gpu",
        "build": build_lgbm,
        "default": default_lgbm_params,
        "fit": lambda m, xt, yt, xv, yv: m.fit(
            xt,
            yt,
            eval_set=[(xv, yv)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
        ),
    },
    "random_forest": {
        "search": RF_SEARCH,
        "device": "cpu",
        "build": build_random_forest,
        "default": lambda cfg, _w: default_rf_params(cfg),
        "fit": lambda m, xt, yt, _xv, _yv: m.fit(xt, yt),
    },
    "extra_trees": {
        "search": ET_SEARCH,
        "device": "cpu",
        "build": build_extra_trees,
        "default": lambda cfg, _w: default_et_params(cfg),
        "fit": lambda m, xt, yt, _xv, _yv: m.fit(xt, yt),
    },
}


def run_trial(
    spec: dict[str, Any],
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    model = spec["build"](params)
    spec["fit"](model, x_train, y_train, x_valid, y_valid)
    valid_prob = model.predict_proba(x_valid)[:, 1]
    threshold, valid_f1 = best_f1_threshold(y_valid, valid_prob)
    return valid_f1, threshold, valid_prob


def main() -> None:
    parser = ArgumentParser(description="Tune a window-stack tree model on window embeddings.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CHOICES))
    parser.add_argument("--trials", type=int, default=36)
    parser.add_argument(
        "--train-sample",
        type=str,
        default="med",
        choices=list(TRAIN_SAMPLE_LIMITS),
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    model_name: str = args.model
    spec = MODEL_SPECS[model_name]
    config = load_tree_config()
    section = config.get(tree_config_section(model_name), {})

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_window_splits()
    train_sample: TrainSample = args.train_sample
    train_cap_used: int | None = TRAIN_SAMPLE_LIMITS[train_sample]

    if args.smoke_test:
        x_train, y_train = x_train[:8000], y_train[:8000]
        x_valid, y_valid = x_valid[:3000], y_valid[:3000]
        x_test, y_test = x_test[:3000], y_test[:3000]
        trials = min(args.trials, 6)
        train_cap_used = min(8000, x_train.shape[0])
    else:
        trials = args.trials
        train_rows_before = x_train.shape[0]
        x_train, y_train, train_cap_used = subsample_train_windows(
            x_train, y_train, train_sample=train_sample
        )
        if x_train.shape[0] < train_rows_before:
            print(
                f"Subsampled train windows ({train_sample}): {x_train.shape[0]:,} "
                f"(cap {train_cap_used:,})",
                flush=True,
            )
        elif train_sample == "full":
            print(f"Using full train windows: {x_train.shape[0]:,}", flush=True)

    auto_weight = _train_trees.auto_scale_pos_weight(y_train)
    rng = random.Random(RANDOM_SEED)
    default_fn: Callable[..., dict[str, Any]] = spec["default"]
    baseline_params = default_fn(section, auto_weight)

    print(
        f"\nWindow-stack {model_name} tuning [pool={POOL} | device={spec['device']} | "
        f"train_sample={train_sample}]",
        flush=True,
    )
    print(
        f"Train: {x_train.shape[0]:,} | Valid: {x_valid.shape[0]:,} | Test: {x_test.shape[0]:,}",
        flush=True,
    )
    print(f"Trials: {trials}\n", flush=True)

    baseline_model = spec["build"](baseline_params)
    spec["fit"](baseline_model, x_train, y_train, x_valid, y_valid)
    baseline_valid_prob = baseline_model.predict_proba(x_valid)[:, 1]
    baseline_threshold, _ = best_f1_threshold(y_valid, baseline_valid_prob)
    baseline_metrics = evaluate_at_threshold(
        baseline_model, x_valid, y_valid, x_test, y_test, threshold=baseline_threshold
    )

    print("--- Baseline (tree_config.yaml) ---", flush=True)
    print(
        f"Valid F1: {baseline_metrics['valid']['f1']:.4f} @ t={baseline_threshold:.2f} | "
        f"Test F1: {baseline_metrics['test']['f1']:.4f} | "
        f"Test AUC: {baseline_metrics['test']['roc_auc']:.4f}",
        flush=True,
    )

    best_valid_f1 = baseline_metrics["valid"]["f1"]
    best_params = dict(baseline_params)
    best_threshold = baseline_threshold
    trial_rows: list[dict[str, Any]] = []

    for trial_idx in range(1, trials + 1):
        params = sample_params(rng, spec["search"])
        tuned_valid_f1, threshold, _ = run_trial(spec, params, x_train, y_train, x_valid, y_valid)
        trial_rows.append(
            {"trial": trial_idx, "valid_f1": tuned_valid_f1, "threshold": threshold, "params": params}
        )
        improved = tuned_valid_f1 > best_valid_f1
        if improved:
            best_valid_f1 = tuned_valid_f1
            best_params = dict(params)
            best_threshold = threshold
        print(
            f"Trial {trial_idx:>3}/{trials} | valid F1={tuned_valid_f1:.4f} @ t={threshold:.2f}"
            + ("  *best*" if improved else ""),
            flush=True,
        )

    tuned_model = spec["build"](best_params)
    spec["fit"](tuned_model, x_train, y_train, x_valid, y_valid)
    tuned_metrics = evaluate_at_threshold(
        tuned_model, x_valid, y_valid, x_test, y_test, threshold=best_threshold
    )

    print("\n--- Best tuned config ---", flush=True)
    print(json.dumps(best_params, indent=2), flush=True)
    print(
        f"Valid F1: {tuned_metrics['valid']['f1']:.4f} | Test F1: {tuned_metrics['test']['f1']:.4f} | "
        f"Test AUC: {tuned_metrics['test']['roc_auc']:.4f} | Threshold: {best_threshold:.2f}",
        flush=True,
    )

    out_dir = _train_trees.resolve_path("03_TREE/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model_name,
        "pool": POOL,
        "level": "window_stack",
        "train_sample": train_sample,
        "train_sample_cap": train_cap_used,
        "train_windows_used": int(x_train.shape[0]),
        "trials": trials,
        "baseline": {
            "params": baseline_params,
            "threshold": baseline_threshold,
            "valid": baseline_metrics["valid"],
            "test": baseline_metrics["test"],
        },
        "tuned": {
            "params": best_params,
            "threshold": best_threshold,
            "valid": tuned_metrics["valid"],
            "test": tuned_metrics["test"],
        },
        "all_trials": trial_rows,
    }
    summary_path = out_dir / tune_summary_path(model_name)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
