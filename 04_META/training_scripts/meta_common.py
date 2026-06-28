"""Shared helpers for meta-learner training."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

SCRIPTS_ROOT = Path(__file__).resolve().parent
META_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = META_ROOT.parent
DEFAULT_CONFIG_PATH = META_ROOT / "meta_config.yaml"


def variant_output_dir(variant: str, settings: dict[str, Any] | None = None) -> Path:
    if settings and settings.get("variant_root"):
        return META_ROOT / str(settings["variant_root"]) / variant
    return META_ROOT / variant


def results_output_dir(settings: dict[str, Any]) -> Path:
    return resolve_path(settings.get("output_dir", "04_META/results"))

_spec = importlib.util.spec_from_file_location("train_trees", PROJECT_ROOT / "03_TREE" / "training_scripts" / "train_trees.py")
_train_trees = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train_trees)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for threshold in np.arange(0.20, 0.81, 0.02):
        f1 = _train_trees.f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(threshold)
    return best_t, best_f1


def sweep_threshold_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    step: float = 0.01,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in np.arange(0.0, 1.0 + step / 2, step):
        t = float(round(threshold, 4))
        metrics = _train_trees.compute_metrics(y_true, y_prob, threshold=t)
        rows.append({"threshold": t, **metrics})
    return pd.DataFrame(rows)


def select_threshold_max_precision_at_recall(
    curve: pd.DataFrame,
    min_recall: float,
) -> tuple[float, dict[str, float]]:
    eligible = curve[curve["recall"] >= min_recall]
    if eligible.empty:
        raise ValueError(f"No threshold achieves recall >= {min_recall}")
    best_row = eligible.sort_values(["precision", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
    threshold = float(best_row["threshold"])
    return threshold, {key: float(best_row[key]) for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]}


def select_threshold_precision_recall_f1(
    curve: pd.DataFrame,
    *,
    min_precision: float = 0.4,
    min_recall: float = 0.6,
    max_recall: float = 0.7,
    min_f1: float = 0.5,
    step: float = 0.01,
) -> tuple[float, dict[str, float], bool]:
    """
    Deployment rule: precision > min_precision, min_recall < recall <= max_recall, f1 > min_f1.
    Returns (threshold, metrics_at_threshold, meets_policy).
    """
    qualified = curve[
        (curve["precision"] > min_precision)
        & (curve["recall"] > min_recall)
        & (curve["recall"] <= max_recall)
        & (curve["f1"] > min_f1)
    ]
    if not qualified.empty:
        best = qualified.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
        metrics = {key: float(best[key]) for key in best.index if key in curve.columns}
        metrics["threshold"] = float(best["threshold"])
        return float(best["threshold"]), metrics, True

    scored = curve.assign(
        deficit=(
            np.maximum(0.0, min_f1 - curve["f1"]) * 2.0
            + np.maximum(0.0, min_precision - curve["precision"])
            + np.maximum(0.0, min_recall - curve["recall"])
            + np.maximum(0.0, curve["recall"] - max_recall)
        ),
        recall_mid_gap=(curve["recall"] - (min_recall + max_recall) / 2.0).abs(),
    )
    best = scored.sort_values(
        ["deficit", "f1", "precision", "recall_mid_gap"],
        ascending=[True, False, False, True],
    ).iloc[0]
    metrics = {key: float(best[key]) for key in best.index if key in curve.columns}
    metrics["threshold"] = float(best["threshold"])
    return float(best["threshold"]), metrics, False


def predict_positive_proba(model: Any, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1].astype(np.float32)


def load_base_models(settings: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for key in ("xgboost_model", "lightgbm_model", "random_forest_model", "extra_trees_model"):
        if key in settings:
            name = key.replace("_model", "")
            models[name] = joblib.load(resolve_path(str(settings[key])))
    return models


def resolve_window_embedding_path(embeddings_root: str, split: str, pool: str) -> Path:
    return resolve_path(f"{embeddings_root}/{pool}/{split}/{split}_window_embeddings.parquet")


def build_window_split_table(
    settings: dict[str, Any],
    split: str,
    feature_columns: list[str],
    label_column: str = "label",
) -> pd.DataFrame:
    pool = str(settings.get("pool", "max"))
    embeddings_root = str(settings.get("embeddings_root", "01_Data_Processing/data/embeddings"))
    embedding_column = str(settings.get("embedding_column", "embedding"))
    model_keys = {
        "xgb": "xgboost_model",
        "lightgbm": "lightgbm_model",
        "random_forest": "random_forest_model",
        "extra_trees": "extra_trees_model",
    }
    embedding_path = resolve_window_embedding_path(embeddings_root, split, pool)
    features, labels = _train_trees.load_embedding_split(embedding_path, label_column, embedding_column)

    frame: dict[str, Any] = {}
    for col in feature_columns:
        model_path_key = model_keys[col]
        if model_path_key not in settings:
            raise KeyError(f"Missing model path for feature column {col!r}")
        model = joblib.load(resolve_path(str(settings[model_path_key])))
        frame[col] = predict_positive_proba(model, features)
    frame[label_column] = labels.astype(int)
    return pd.DataFrame(frame)


def build_split_table(
    settings: dict[str, Any],
    split: str,
    feature_columns: list[str],
    label_column: str = "label",
) -> pd.DataFrame:
    if str(settings.get("granularity", "function")).lower() == "window":
        return build_window_split_table(settings, split, feature_columns, label_column)

    raise RuntimeError(
        "Function-granularity meta tables used pooled function embeddings, which were removed. "
        "Set train_meta.granularity: window (see meta_config.yaml)."
    )
