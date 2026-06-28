"""Shared helpers for window-stack tree hyperparameter search."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

TREE_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = TREE_ROOT / "training_scripts"

_spec = importlib.util.spec_from_file_location("train_trees", SCRIPTS_ROOT / "train_trees.py")
_train_trees = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train_trees)

POOL = "max"
RANDOM_SEED = 42

TRAIN_SAMPLE_LIMITS: dict[str, int | None] = {
    "low": 20_000,
    "med": 50_000,
    "high": 100_000,
    "full": None,
}
TrainSample = Literal["low", "med", "high", "full"]

MODEL_CHOICES = ("xgboost", "lightgbm", "random_forest", "extra_trees")


def resolve_window_path(split: str) -> Path:
    tree_cfg = _train_trees.load_config(TREE_ROOT / "tree_config.yaml")
    root = tree_cfg.get("train_trees", {}).get("embeddings_root", _train_trees.DEFAULT_EMBEDDINGS_ROOT)
    rel = f"{root}/{POOL}/{split}/{split}_window_embeddings.parquet"
    return _train_trees.resolve_path(rel)


def load_window_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path, columns=["label", "embedding"])
    x = np.vstack(frame["embedding"].to_numpy()).astype(np.float32)
    y = frame["label"].astype(int).to_numpy()
    return x, y


def load_window_splits() -> tuple[np.ndarray, ...]:
    train_path = resolve_window_path("train")
    valid_path = resolve_window_path("valid")
    test_path = resolve_window_path("test")
    x_train, y_train = load_window_split(train_path)
    x_valid, y_valid = load_window_split(valid_path)
    x_test, y_test = load_window_split(test_path)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def subsample_train_windows(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    train_sample: TrainSample,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    cap = TRAIN_SAMPLE_LIMITS[train_sample]
    if cap is None or x_train.shape[0] <= cap:
        return x_train, y_train, cap
    rng_sub = np.random.RandomState(seed)
    keep = rng_sub.choice(x_train.shape[0], size=cap, replace=False)
    return x_train[keep], y_train[keep], cap


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for threshold in np.arange(0.20, 0.81, 0.02):
        f1 = _train_trees.f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(threshold)
    return best_t, best_f1


def evaluate_at_threshold(
    model: Any,
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


def tune_summary_path(model: str) -> str:
    return f"window_stack_{model}_tune_max.json"


def tree_config_section(model: str) -> str:
    return model
