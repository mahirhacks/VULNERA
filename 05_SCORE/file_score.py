"""File-level risk from function calibrated scores with max floor + spread uplift."""

from __future__ import annotations

from typing import Any

POOLING_METHOD = "max_plus_mean_excess"
POOLING_METHOD_MAX = "window_prob_max"
DEFAULT_FUNCTION_THRESHOLD = 0.29
DEFAULT_SPREAD_WEIGHT = 0.25
MAX_POOL_TOLERANCE = 1e-9

STATUS_LABELS: dict[str, str] = {
    "agree_positive": "Vulnerable",
    "review_suggested": "Needs review",
    "diffuse_risk": "Diffuse risk",
    "agree_negative": "Safe",
}


def function_calibrated_score(function: dict[str, Any]) -> float:
    try:
        return float(function.get("function_score_calibrated", 0.0))
    except (TypeError, ValueError):
        return 0.0


def mean_excess_above_threshold(scores: list[float], threshold: float) -> float:
    if not scores:
        return 0.0
    tau = float(threshold)
    return sum(max(0.0, float(score) - tau) for score in scores) / len(scores)


def other_scores_below_peak(scores: list[float]) -> list[float]:
    """Scores strictly below the peak (excludes tied max item(s))."""
    if not scores:
        return []
    base = max(float(score) for score in scores)
    return [float(score) for score in scores if float(score) < base - MAX_POOL_TOLERANCE]


def other_function_scores(scores: list[float]) -> list[float]:
    """Alias for file-level spread (other functions below the peak)."""
    return other_scores_below_peak(scores)


def composite_pool_risk(
    scores: list[float],
    *,
    threshold: float = DEFAULT_FUNCTION_THRESHOLD,
    weight: float = DEFAULT_SPREAD_WEIGHT,
) -> dict[str, float]:
    """
    pooled = min(1, max(r_i) + w * mean(max(0, r_j - tau))) over *other* items j.

    A single-item list returns that score unchanged (no uplift).
  Used for windows -> function and functions -> file.
    """
    if not scores:
        return {
            "pooled_risk": 0.0,
            "base_max_risk": 0.0,
            "mean_excess_above_threshold": 0.0,
            "spread_uplift": 0.0,
            "other_count": 0,
        }

    base = max(float(score) for score in scores)
    others = other_scores_below_peak(scores)
    mean_excess = mean_excess_above_threshold(others, threshold) if others else 0.0
    uplift = float(weight) * mean_excess
    return {
        "pooled_risk": min(1.0, base + uplift),
        "base_max_risk": base,
        "mean_excess_above_threshold": mean_excess,
        "spread_uplift": uplift,
        "other_count": len(others),
    }


def composite_file_risk(
    scores: list[float],
    *,
    threshold: float = DEFAULT_FUNCTION_THRESHOLD,
    weight: float = DEFAULT_SPREAD_WEIGHT,
    pooling: str = POOLING_METHOD,
) -> dict[str, float]:
    """
    file_risk = min(1, max(r_i) + w * mean(max(0, r_j - tau)))  [max_plus_mean_excess]

    window_prob_max: file_risk = max(r_i) over functions (pure max-pool).
    """
    if pooling == POOLING_METHOD_MAX:
        if not scores:
            return {
                "file_risk_calibrated": 0.0,
                "base_max_risk": 0.0,
                "mean_excess_above_threshold": 0.0,
                "spread_uplift": 0.0,
                "other_function_count": 0,
            }
        base = max(float(score) for score in scores)
        return {
            "file_risk_calibrated": base,
            "base_max_risk": base,
            "mean_excess_above_threshold": 0.0,
            "spread_uplift": 0.0,
            "other_function_count": max(0, len(scores) - 1),
        }

    pooled = composite_pool_risk(scores, threshold=threshold, weight=weight)
    return {
        "file_risk_calibrated": pooled["pooled_risk"],
        "base_max_risk": pooled["base_max_risk"],
        "mean_excess_above_threshold": pooled["mean_excess_above_threshold"],
        "spread_uplift": pooled["spread_uplift"],
        "other_function_count": pooled["other_count"],
    }


def _status_label(function: dict[str, Any]) -> str:
    display = function.get("status_display") or {}
    label = display.get("label")
    if label:
        return str(label)
    status = str(function.get("agreement_status", "agree_negative"))
    return STATUS_LABELS.get(status, status)


def build_file_score(
    functions: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_FUNCTION_THRESHOLD,
    weight: float = DEFAULT_SPREAD_WEIGHT,
    pooling: str | None = None,
) -> dict[str, Any]:
    """
    Aggregate function-level risks into a single file prediction.

    Default: max floor + spread uplift across other functions.
    window_prob_max: pure max over function risks (window-max-pool stack).
    """
    resolved_pooling = pooling or POOLING_METHOD
    if resolved_pooling == "window_max_pool":
        resolved_pooling = POOLING_METHOD_MAX
    entries: list[dict[str, Any]] = []
    for function in functions:
        calibrated = function_calibrated_score(function)
        entries.append(
            {
                "function_id": function.get("function_group_id"),
                "name": function.get("name") or "function",
                "calibrated_risk": calibrated,
                "status": str(function.get("agreement_status", "agree_negative")),
                "status_label": _status_label(function),
                "function_flagged": bool(function.get("function_flagged")),
                "max_window_prob": float(function.get("max_window_prob") or 0.0),
            }
        )

    scores = [entry["calibrated_risk"] for entry in entries]
    composite = composite_file_risk(
        scores,
        threshold=threshold,
        weight=weight,
        pooling=resolved_pooling,
    )
    base_max = composite["base_max_risk"]
    contributor_ids = [
        entry["function_id"]
        for entry in entries
        if entry["function_id"] is not None
        and abs(float(entry["calibrated_risk"]) - base_max) <= MAX_POOL_TOLERANCE
    ]

    return {
        "pooling": resolved_pooling,
        "file_risk_calibrated": composite["file_risk_calibrated"],
        "base_max_risk": composite["base_max_risk"],
        "mean_excess_above_threshold": composite["mean_excess_above_threshold"],
        "spread_uplift": composite["spread_uplift"],
        "other_function_count": composite["other_function_count"],
        "function_threshold": float(threshold),
        "spread_weight": float(weight),
        "function_count": len(entries),
        "functions": entries,
        "max_pool_contributor_ids": contributor_ids,
    }
