"""
Shared tree-model training utilities for window-stack classifiers.

Used by train_window_stack_trees.py, meta/score calibration, and aggregator eval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier, early_stopping
except ImportError as exc:
    raise ImportError("lightgbm is required. Install with: pip install lightgbm") from exc

SCRIPTS_ROOT = Path(__file__).resolve().parent
TREE_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = TREE_ROOT.parent
DEFAULT_CONFIG_PATH = TREE_ROOT / "tree_config.yaml"
DEFAULT_EMBEDDINGS_ROOT = "01_Data_Processing/data/embeddings"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def embeddings_root_from_config(config: dict[str, Any]) -> str:
    train_cfg = config.get("train_trees", {})
    window_cfg = config.get("train_window_trees", {})
    return str(
        train_cfg.get("embeddings_root")
        or window_cfg.get("embeddings_root")
        or DEFAULT_EMBEDDINGS_ROOT
    )


def resolve_window_embedding_path(split: str, pool: str, embeddings_root: str | None = None) -> str:
    root = embeddings_root or DEFAULT_EMBEDDINGS_ROOT
    return f"{root}/{pool}/{split}/{split}_window_embeddings.parquet"


def load_embedding_split(path: Path, label_column: str, embedding_column: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path, columns=[label_column, embedding_column])
    labels = frame[label_column].astype(int).to_numpy()
    features = np.vstack(frame[embedding_column].to_numpy()).astype(np.float32)
    return features, labels


def load_window_split(
    path: Path,
    *,
    label_column: str,
    embedding_column: str,
    window_id_column: str,
    function_group_column: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = pd.read_parquet(
        path,
        columns=[label_column, embedding_column, window_id_column, function_group_column],
    )
    features = np.vstack(frame[embedding_column].to_numpy()).astype(np.float32)
    labels = frame[label_column].astype(int).to_numpy()
    return features, labels, frame[[window_id_column, function_group_column, label_column]].copy()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "avg_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def auto_scale_pos_weight(y_train: np.ndarray) -> float:
    positives = max(int((y_train == 1).sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    return negatives / positives


def build_xgb(cfg: dict[str, Any], scale_pos_weight: float) -> XGBClassifier:
    weight = cfg.get("scale_pos_weight")
    device = str(cfg.get("device", "cpu"))
    params: dict[str, Any] = {
        "n_estimators": int(cfg.get("n_estimators", 500)),
        "max_depth": int(cfg.get("max_depth", 6)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "subsample": float(cfg.get("subsample", 0.8)),
        "colsample_bytree": float(cfg.get("colsample_bytree", 0.8)),
        "reg_lambda": float(cfg.get("reg_lambda", 1.0)),
        "min_child_weight": float(cfg.get("min_child_weight", 1.0)),
        "gamma": float(cfg.get("gamma", 0.0)),
        "scale_pos_weight": float(weight if weight is not None else scale_pos_weight),
        "tree_method": str(cfg.get("tree_method", "hist")),
        "device": device,
        "eval_metric": str(cfg.get("eval_metric", "logloss")),
        "early_stopping_rounds": int(cfg.get("early_stopping_rounds", 50)),
        "random_state": 42,
    }
    if device == "cpu":
        params["n_jobs"] = -1
    return XGBClassifier(**params)


def build_lgbm(cfg: dict[str, Any], scale_pos_weight: float) -> LGBMClassifier:
    weight = cfg.get("scale_pos_weight")
    device = str(cfg.get("device", "cpu"))
    params: dict[str, Any] = {
        "n_estimators": int(cfg.get("n_estimators", 500)),
        "max_depth": int(cfg.get("max_depth", -1)),
        "num_leaves": int(cfg.get("num_leaves", 64)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "subsample": float(cfg.get("subsample", 0.8)),
        "colsample_bytree": float(cfg.get("colsample_bytree", 0.8)),
        "reg_lambda": float(cfg.get("reg_lambda", 1.0)),
        "reg_alpha": float(cfg.get("reg_alpha", 0.0)),
        "min_child_samples": int(cfg.get("min_child_samples", 20)),
        "min_split_gain": float(cfg.get("min_split_gain", 0.0)),
        "objective": str(cfg.get("objective", "binary")),
        "random_state": 42,
        "verbose": -1,
    }
    if device != "cpu":
        params["device"] = device
    else:
        params["n_jobs"] = -1
    if weight is not None:
        params["scale_pos_weight"] = float(weight)
    else:
        params["scale_pos_weight"] = scale_pos_weight
    return LGBMClassifier(**params)


def build_random_forest(cfg: dict[str, Any]) -> RandomForestClassifier:
    max_depth = cfg.get("max_depth")
    bootstrap = bool(cfg.get("bootstrap", True))
    max_samples = cfg.get("max_samples")
    return RandomForestClassifier(
        n_estimators=int(cfg.get("n_estimators", 500)),
        max_depth=None if max_depth in (None, "null", -1) else int(max_depth),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
        min_samples_split=int(cfg.get("min_samples_split", 2)),
        max_features=cfg.get("max_features", "sqrt"),
        class_weight=str(cfg.get("class_weight", "balanced_subsample")),
        bootstrap=bootstrap,
        max_samples=float(max_samples) if bootstrap and max_samples is not None else None,
        random_state=42,
        n_jobs=int(cfg.get("n_jobs", -1)),
    )


def build_extra_trees(cfg: dict[str, Any]) -> ExtraTreesClassifier:
    max_depth = cfg.get("max_depth")
    max_samples = cfg.get("max_samples")
    bootstrap = bool(cfg.get("bootstrap", False))
    if max_samples is not None:
        max_samples = float(max_samples)
        bootstrap = bootstrap or max_samples < 1.0
    return ExtraTreesClassifier(
        n_estimators=int(cfg.get("n_estimators", 500)),
        max_depth=None if max_depth in (None, "null", -1) else int(max_depth),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
        min_samples_split=int(cfg.get("min_samples_split", 2)),
        max_features=cfg.get("max_features", "sqrt"),
        class_weight=str(cfg.get("class_weight", "balanced_subsample")),
        min_impurity_decrease=float(cfg.get("min_impurity_decrease", 0.0)),
        bootstrap=bootstrap,
        max_samples=max_samples if bootstrap and max_samples is not None else None,
        random_state=42,
        n_jobs=int(cfg.get("n_jobs", -1)),
    )


def model_output_dir(base_dir: Path, trained_subdir: str) -> Path:
    return base_dir / trained_subdir


def train_and_evaluate(
    name: str,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    base_dir: Path,
    *,
    trained_subdir: str,
    use_early_stopping: bool,
    decision_threshold: float = 0.5,
    early_stopping_rounds: int = 50,
) -> dict[str, Any]:
    output_dir = model_output_dir(base_dir, trained_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_early_stopping:
        if name == "lightgbm":
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                eval_metric="binary_logloss",
                callbacks=[early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
            )
        else:
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                verbose=False,
            )
    else:
        model.fit(x_train, y_train)

    valid_prob = model.predict_proba(x_valid)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    valid_metrics = compute_metrics(y_valid, valid_prob, threshold=decision_threshold)
    test_metrics = compute_metrics(y_test, test_prob, threshold=decision_threshold)

    model_path = output_dir / f"{name}_model.joblib"
    joblib.dump(model, model_path)

    pd.DataFrame({"prob": valid_prob, "label": y_valid}).to_parquet(
        output_dir / f"{name}_valid_predictions.parquet", index=False
    )
    pd.DataFrame({"prob": test_prob, "label": y_test}).to_parquet(
        output_dir / f"{name}_test_predictions.parquet", index=False
    )

    return {
        "name": name,
        "model_path": str(model_path),
        "valid_predictions": str(output_dir / f"{name}_valid_predictions.parquet"),
        "test_predictions": str(output_dir / f"{name}_test_predictions.parquet"),
        "decision_threshold": decision_threshold,
        "valid": valid_metrics,
        "test": test_metrics,
    }
