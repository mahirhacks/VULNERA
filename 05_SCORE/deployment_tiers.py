"""Tiered deployment labels from calibrated function risk."""

from __future__ import annotations

from typing import Any

DEFAULT_VULN_THRESHOLD = 0.32
DEFAULT_REVIEW_THRESHOLD = 0.26


def classify_function_deployment_tier(
    score: float,
    *,
    vuln_threshold: float = DEFAULT_VULN_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """
    Map function risk to deployment tier:
      >= vuln_threshold (32%)  -> Vulnerable
      >= review_threshold (26%) -> Needs review
      < review_threshold        -> Safe
    """
    risk = max(0.0, min(1.0, float(score)))
    vuln_tau = float(vuln_threshold)
    review_tau = float(review_threshold)

    if risk >= vuln_tau:
        return {
            "agreement_status": "agree_positive",
            "deployment_tier": "vuln",
            "function_flagged": True,
            "function_needs_review": True,
            "user_facing_vuln": True,
            "whole_function_vuln": True,
        }
    if risk >= review_tau:
        return {
            "agreement_status": "review_suggested",
            "deployment_tier": "needs_review",
            "function_flagged": False,
            "function_needs_review": True,
            "user_facing_vuln": True,
            "whole_function_vuln": False,
        }
    return {
        "agreement_status": "agree_negative",
        "deployment_tier": "safe",
        "function_flagged": False,
        "function_needs_review": False,
        "user_facing_vuln": False,
        "whole_function_vuln": False,
    }


def apply_function_deployment_tier(
    record: dict[str, Any],
    *,
    vuln_threshold: float | None = None,
    review_threshold: float | None = None,
) -> dict[str, Any]:
    """Apply tiered deployment fields from function_score_calibrated."""
    thresholds = record.get("thresholds") or {}
    vuln_tau = float(
        vuln_threshold
        if vuln_threshold is not None
        else thresholds.get("function", DEFAULT_VULN_THRESHOLD)
    )
    review_tau = float(
        review_threshold
        if review_threshold is not None
        else thresholds.get("function_review", DEFAULT_REVIEW_THRESHOLD)
    )
    score = float(record.get("function_score_calibrated") or 0.0)
    tier = classify_function_deployment_tier(
        score,
        vuln_threshold=vuln_tau,
        review_threshold=review_tau,
    )
    updated = dict(record)
    updated.update(tier)
    return updated
