"""Shared helpers for score calibration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPTS_ROOT = Path(__file__).resolve().parent
SCORE_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = SCORE_ROOT.parent
META_SCRIPTS_ROOT = PROJECT_ROOT / "04_META" / "training_scripts"
DEFAULT_CONFIG_PATH = SCORE_ROOT / "score_config.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def import_meta_common():
    if str(META_SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(META_SCRIPTS_ROOT))
    import meta_common  # noqa: PLC0415

    return meta_common


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss  # noqa: PLC0415

    return float(brier_score_loss(y_true, y_prob))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    if len(y_true) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        low = bin_edges[index]
        high = bin_edges[index + 1]
        if index == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if not np.any(mask):
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += float(mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for index in range(n_bins):
        low = bin_edges[index]
        high = bin_edges[index + 1]
        if index == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        count = int(mask.sum())
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "count": count,
                "mean_predicted": float(y_prob[mask].mean()) if count else 0.0,
                "fraction_positive": float(y_true[mask].mean()) if count else 0.0,
            }
        )
    return rows


def select_threshold_balanced(
    curve: Any,
    *,
    min_f1: float,
    min_precision: float,
    min_recall: float,
) -> tuple[float, dict[str, float]]:
    """Pick threshold with f1/precision/recall strictly above floors; else minimize deficit."""
    import pandas as pd  # noqa: PLC0415

    frame = curve if isinstance(curve, pd.DataFrame) else pd.DataFrame(curve)
    qualified = frame[
        (frame["f1"] > min_f1)
        & (frame["precision"] > min_precision)
        & (frame["recall"] > min_recall)
    ]
    if not qualified.empty:
        best = qualified.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    else:
        scored = frame.assign(
            deficit=(
                np.maximum(0.0, min_f1 - frame["f1"]) * 2.0
                + np.maximum(0.0, min_precision - frame["precision"])
                + np.maximum(0.0, min_recall - frame["recall"])
            )
        )
        best = scored.sort_values(["deficit", "f1", "precision"], ascending=[True, False, False]).iloc[0]

    threshold = float(best["threshold"])
    metrics = {
        key: float(best[key])
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
        if key in best.index
    }
    metrics["threshold"] = threshold
    metrics["meets_policy"] = bool(
        metrics.get("f1", 0.0) > min_f1
        and metrics.get("precision", 0.0) > min_precision
        and metrics.get("recall", 0.0) > min_recall
    )
    return threshold, metrics


def threshold_policy_from_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    policy_name = str(cfg.get("threshold_policy", "")).lower()
    if policy_name != "balanced":
        return None
    return {
        "name": "balanced",
        "min_f1": float(cfg.get("min_f1", 0.5)),
        "min_precision": float(cfg.get("min_precision", 0.4)),
        "min_recall": float(cfg.get("min_recall", 0.6)),
    }
