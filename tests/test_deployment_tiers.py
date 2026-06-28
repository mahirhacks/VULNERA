"""Tests for tiered deployment thresholds."""

from __future__ import annotations

import sys
from pathlib import Path

SCORE_DIR = Path(__file__).resolve().parents[1] / "05_SCORE"
if str(SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(SCORE_DIR))

from deployment_tiers import classify_function_deployment_tier  # noqa: E402


def test_tier_vuln_at_32_percent() -> None:
    tier = classify_function_deployment_tier(0.32)
    assert tier["deployment_tier"] == "vuln"
    assert tier["agreement_status"] == "agree_positive"
    assert tier["function_flagged"] is True


def test_tier_needs_review_band() -> None:
    tier = classify_function_deployment_tier(0.28)
    assert tier["deployment_tier"] == "needs_review"
    assert tier["agreement_status"] == "review_suggested"
    assert tier["function_flagged"] is False
    assert tier["user_facing_vuln"] is True


def test_tier_safe_below_26_percent() -> None:
    tier = classify_function_deployment_tier(0.25)
    assert tier["deployment_tier"] == "safe"
    assert tier["agreement_status"] == "agree_negative"
    assert tier["user_facing_vuln"] is False
