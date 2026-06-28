"""
Train meta-learners on stacked base-model probabilities.

Variants:
  logistic    — LogisticRegression (linear stack, 4 features)
  polynomial  — degree-2 interaction-only polynomial + LogisticRegression (10 features)
  xgboost     — shallow XGBoost (non-linear interactions)

Usage:
    python 04_META/training_scripts/train_meta.py --variant logistic
    python 04_META/training_scripts/train_meta.py --variant polynomial
    python 04_META/training_scripts/train_meta.py --variant xgboost
    python 04_META/training_scripts/train_meta.py --variant all
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier

from meta_common import (
    META_ROOT,
    best_f1_threshold,
    build_split_table,
    load_config,
    resolve_path,
    results_output_dir,
    variant_output_dir,
)

import importlib.util

_spec = importlib.util.spec_from_file_location("train_trees", META_ROOT.parent / "03_TREE" / "training_scripts" / "train_trees.py")
_train_trees = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train_trees)

DEFAULT_CONFIG_PATH = META_ROOT / "meta_config.yaml"


def build_logistic(cfg: dict[str, Any]) -> LogisticRegression:
    kwargs: dict[str, Any] = {
        "C": float(cfg.get("C", 1.0)),
        "max_iter": int(cfg.get("max_iter", 1000)),
        "class_weight": cfg.get("class_weight", "balanced"),
        "random_state": int(cfg.get("random_state", 42)),
    }
    if cfg.get("solver"):
        kwargs["solver"] = str(cfg["solver"])
    return LogisticRegression(**kwargs)


def build_polynomial_logistic(cfg: dict[str, Any], c_value: float) -> Pipeline:
    poly_cfg = cfg.get("polynomial_features", {})
    lr_cfg = cfg.get("logistic_regression", {})
    return Pipeline(
        [
            (
                "poly",
                PolynomialFeatures(
                    degree=int(poly_cfg.get("degree", 2)),
                    interaction_only=bool(poly_cfg.get("interaction_only", True)),
                    include_bias=bool(poly_cfg.get("include_bias", False)),
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=float(c_value),
                    max_iter=int(lr_cfg.get("max_iter", 1000)),
                    class_weight=lr_cfg.get("class_weight", "balanced"),
                    random_state=int(lr_cfg.get("random_state", 42)),
                ),
            ),
        ]
    )


def tune_polynomial_c(
    x_train: np.ndarray,
    y_train: np.ndarray,
    settings: dict[str, Any],
) -> float:
    poly_cfg = settings.get("meta_polynomial", {})
    configured_c = poly_cfg.get("C")
    if configured_c is not None:
        return float(configured_c)

    c_grid = [float(value) for value in poly_cfg.get("C_grid", [0.01, 0.1, 1.0, 10.0])]
    best_c = c_grid[0]
    best_valid_f1 = -1.0
    for c_value in c_grid:
        model = build_polynomial_logistic(poly_cfg, c_value)
        model.fit(x_train, y_train)
        valid_prob = model.predict_proba(x_train)[:, 1]
        _, valid_f1 = best_f1_threshold(y_train, valid_prob)
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_c = c_value
    return best_c


def pipeline_feature_names(model: Pipeline, feature_columns: list[str]) -> list[str]:
    return [str(name) for name in model.named_steps["poly"].get_feature_names_out(feature_columns)]


def build_meta_xgb(cfg: dict[str, Any]) -> XGBClassifier:
    device = str(cfg.get("device", "cuda"))
    params: dict[str, Any] = {
        "n_estimators": int(cfg.get("n_estimators", 100)),
        "max_depth": int(cfg.get("max_depth", 3)),
        "learning_rate": float(cfg.get("learning_rate", 0.1)),
        "subsample": float(cfg.get("subsample", 0.9)),
        "colsample_bytree": float(cfg.get("colsample_bytree", 1.0)),
        "reg_lambda": float(cfg.get("reg_lambda", 1.0)),
        "min_child_weight": float(cfg.get("min_child_weight", 1.0)),
        "gamma": float(cfg.get("gamma", 0.0)),
        "scale_pos_weight": float(cfg.get("scale_pos_weight", 1.0)),
        "tree_method": str(cfg.get("tree_method", "hist")),
        "device": device,
        "eval_metric": "logloss",
        "random_state": int(cfg.get("random_state", 42)),
    }
    if device == "cpu":
        params["n_jobs"] = -1
    return XGBClassifier(**params)


def fit_meta_model(
    variant: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    settings: dict[str, Any],
    *,
    feature_columns: list[str] | None = None,
) -> Any:
    if variant == "logistic":
        return build_logistic(settings.get("logistic_regression", {})).fit(x_train, y_train)
    if variant == "polynomial":
        poly_cfg = settings.get("meta_polynomial", {})
        c_value = tune_polynomial_c(x_train, y_train, settings)
        print(f"Selected C={c_value} (valid F1 grid search)")
        return build_polynomial_logistic(poly_cfg, c_value).fit(x_train, y_train)
    if variant == "xgboost":
        return build_meta_xgb(settings.get("meta_xgboost", {})).fit(x_train, y_train)
    raise ValueError(f"Unknown meta variant: {variant}")


def predict_meta_prob(model: Any, variant: str, x: np.ndarray) -> np.ndarray:
    if variant == "logistic":
        return model.predict_proba(x)[:, 1]
    return model.predict_proba(x)[:, 1]


def train_variant(
    variant: str,
    settings: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    results_dir: Path,
) -> dict[str, Any]:
    feature_columns = [str(col) for col in settings.get("feature_columns", ["xgb", "lightgbm"])]
    label_column = str(settings.get("label_column", "label"))

    x_train = train_frame[feature_columns].to_numpy(dtype=np.float32)
    y_train = train_frame[label_column].astype(int).to_numpy()
    x_test = test_frame[feature_columns].to_numpy(dtype=np.float32)
    y_test = test_frame[label_column].astype(int).to_numpy()

    model = fit_meta_model(variant, x_train, y_train, settings, feature_columns=feature_columns)
    valid_prob = predict_meta_prob(model, variant, x_train)

    threshold_cfg = settings.get("decision_thresholds", {}).get(variant)
    if threshold_cfg is None:
        threshold, _ = best_f1_threshold(y_train, valid_prob)
    else:
        threshold = float(threshold_cfg)

    valid_metrics = _train_trees.compute_metrics(y_train, valid_prob, threshold=threshold)
    test_prob = predict_meta_prob(model, variant, x_test)
    test_metrics = _train_trees.compute_metrics(y_test, test_prob, threshold=threshold)

    variant_dir = variant_output_dir(variant, settings)
    variant_dir.mkdir(parents=True, exist_ok=True)
    model_path = variant_dir / "meta_model.joblib"
    joblib.dump(model, model_path)

    pd.DataFrame({"prob": valid_prob, "label": y_train}).to_parquet(
        variant_dir / "meta_valid_predictions.parquet", index=False
    )
    pd.DataFrame({"prob": test_prob, "label": y_test}).to_parquet(
        variant_dir / "meta_test_predictions.parquet", index=False
    )

    summary: dict[str, Any] = {
        "variant": variant,
        "feature_columns": feature_columns,
        "label_column": label_column,
        "decision_threshold": threshold,
        "model_path": str(model_path),
        "valid": valid_metrics,
        "test": test_metrics,
    }
    if variant == "logistic" and hasattr(model, "coef_"):
        summary["coefficients"] = {
            feature_columns[i]: float(model.coef_[0][i]) for i in range(len(feature_columns))
        }
        summary["intercept"] = float(model.intercept_[0])
    if variant == "polynomial":
        poly_cfg = settings.get("meta_polynomial", {})
        clf = model.named_steps["clf"]
        names = pipeline_feature_names(model, feature_columns)
        summary["C"] = float(clf.C)
        summary["coefficients"] = {names[i]: float(clf.coef_[0][i]) for i in range(len(names))}
        summary["intercept"] = float(clf.intercept_[0])
        summary["polynomial_features"] = poly_cfg.get("polynomial_features", {})
    if variant == "xgboost":
        summary["hyperparameters"] = settings.get("meta_xgboost", {})

    summary_path = variant_dir / "meta_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_comparison(results: list[dict[str, Any]], output_dir: Path, feature_columns: list[str]) -> None:
    lines = [
        "Vulnera — meta-learner comparison",
        f"Base features: {', '.join(feature_columns)}",
        "",
        f"{'Variant':<12} {'Valid F1':>10} {'Test F1':>10} {'F1 gap':>10} {'Test AUC':>10} {'Threshold':>10}",
        "-" * 66,
    ]
    for result in results:
        gap = result["valid"]["f1"] - result["test"]["f1"]
        lines.append(
            f"{result['variant']:<12} "
            f"{result['valid']['f1']:>10.4f} "
            f"{result['test']['f1']:>10.4f} "
            f"{gap:>10.4f} "
            f"{result['test']['roc_auc']:>10.4f} "
            f"{result['decision_threshold']:>10.2f}"
        )
    report = "\n".join(lines) + "\n"
    (output_dir / "meta_comparison.txt").write_text(report, encoding="utf-8")
    (output_dir / "meta_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + report)


def main() -> None:
    parser = ArgumentParser(description="Train meta-learners on stacked base predictions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["logistic", "polynomial", "xgboost", "all"],
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else META_ROOT / args.config
    config = load_config(config_path)
    settings = config.get("train_meta", {})
    feature_columns = [str(col) for col in settings.get("feature_columns", ["xgb", "lightgbm"])]
    label_column = str(settings.get("label_column", "label"))
    input_path = resolve_path(str(settings["input_table"]))
    output_dir = results_output_dir(settings)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.exists():
        train_frame = pd.read_parquet(input_path)
    else:
        train_frame = build_split_table(settings, "valid", feature_columns, label_column)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        train_frame.to_parquet(input_path, index=False)

    test_frame = build_split_table(settings, "test", feature_columns, label_column)
    variants = ["logistic", "polynomial", "xgboost"] if args.variant == "all" else [args.variant]

    results: list[dict[str, Any]] = []
    for variant in variants:
        print(f"\n--- Training meta-learner: {variant} ---")
        result = train_variant(variant, settings, train_frame, test_frame, output_dir)
        results.append(result)
        print(
            f"Valid F1: {result['valid']['f1']:.4f} @ t={result['decision_threshold']:.2f} | "
            f"Test F1: {result['test']['f1']:.4f} | Test AUC: {result['test']['roc_auc']:.4f}"
        )

    if len(results) > 1:
        write_comparison(results, output_dir, feature_columns)


if __name__ == "__main__":
    main()
