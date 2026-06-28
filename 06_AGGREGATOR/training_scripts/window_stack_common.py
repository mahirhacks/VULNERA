"""Window-level stack scoring (4 trees → meta → calibration) and max-pool aggregation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aggregator_common import apply_score_calibrator, predict_meta_raw

_SCORE_DIR = Path(__file__).resolve().parents[2] / "05_SCORE"
if str(_SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCORE_DIR))
from file_score import DEFAULT_SPREAD_WEIGHT, composite_pool_risk  # noqa: E402


def predict_base_probs(
    embeddings: np.ndarray,
    *,
    base_models: dict[str, Any],
    feature_columns: list[str],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for col in feature_columns:
        model = base_models[col]
        columns.append(model.predict_proba(embeddings)[:, 1].astype(np.float32))
    return np.column_stack(columns).astype(np.float32)


def score_window_embeddings(
    embeddings: np.ndarray,
    *,
    base_models: dict[str, Any],
    feature_columns: list[str],
    meta_model: Any,
    calibrator_bundle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-window stack:
      embedding -> 4 tree probabilities -> meta learner -> calibration -> window_prob
    """
    if embeddings.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, np.empty((0, len(feature_columns)), dtype=np.float32)

    base_matrix = predict_base_probs(embeddings, base_models=base_models, feature_columns=feature_columns)
    raw_scores = predict_meta_raw(meta_model, base_matrix)
    calibrated = apply_score_calibrator(calibrator_bundle, raw_scores)
    return raw_scores, calibrated, base_matrix


def pool_window_probs_for_function(
    window_probs: np.ndarray | list[float],
    *,
    threshold: float,
    weight: float = DEFAULT_SPREAD_WEIGHT,
) -> dict[str, float]:
    """Max-pool window scores with spread uplift; single-window functions keep the window score."""
    scores = [float(prob) for prob in np.asarray(window_probs).reshape(-1)]
    return composite_pool_risk(scores, threshold=threshold, weight=weight)


def attach_window_scores(
    window_frame: pd.DataFrame,
    *,
    calibrated_scores: np.ndarray,
    raw_scores: np.ndarray | None = None,
    prob_column: str = "window_prob",
) -> pd.DataFrame:
    frame = window_frame.copy()
    frame[prob_column] = calibrated_scores.astype(np.float32)
    if raw_scores is not None:
        frame["window_raw_score"] = raw_scores.astype(np.float32)
    return frame


def function_scores_from_windows(
    window_frame: pd.DataFrame,
    function_meta: pd.DataFrame,
    *,
    function_group_column: str = "function_group_id",
    prob_column: str = "window_prob",
    threshold: float,
    weight: float = DEFAULT_SPREAD_WEIGHT,
) -> np.ndarray:
    pooled_by_function: dict[str, float] = {}
    for group_id, group in window_frame.groupby(function_group_column, sort=False):
        pooled = pool_window_probs_for_function(
            group[prob_column].to_numpy(dtype=np.float32),
            threshold=threshold,
            weight=weight,
        )
        pooled_by_function[str(group_id)] = float(pooled["pooled_risk"])

    return (
        function_meta[function_group_column]
        .astype(str)
        .map(pooled_by_function)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
