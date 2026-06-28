"""Unit tests for tri-layer signature / novelty engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACK_END = Path(__file__).resolve().parents[1] / "09_WEB" / "back_end"
if str(BACK_END) not in sys.path:
    sys.path.insert(0, str(BACK_END))

from pipeline.signature_engine import (  # noqa: E402
    analyze_code_segment,
    classify_novelty,
    clear_catalog_cache,
    extract_metadata_hints,
)
from pipeline.signature_runtime import (  # noqa: E402
    attach_signature_attribution,
    compute_graduated_boost_score,
    compute_signature_plateau_score,
    compute_smooth_signature_boost_score,
    corroboration_allows,
    match_signatures,
    ml_support_score,
)

TEST_DIR = Path(__file__).resolve().parents[1] / "10_TEST"


def _read_test(name: str) -> str:
    return (TEST_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "filename,expected_cwe",
    [
        ("test_1.c", "CWE-125"),
        ("test_2.c", "CWE-78"),
        ("test_3.c", "CWE-362"),
        ("test_4.c", "CWE-22"),
        ("test_6.c", "CWE-190"),
    ],
)
def test_known_cwe_detected_in_test_corpus(filename: str, expected_cwe: str) -> None:
    code = _read_test(filename)
    hints = extract_metadata_hints(code)
    assert expected_cwe in hints.cwes

    hits, _ = analyze_code_segment(code)
    cwes = {hit.cwe for hit in hits}
    assert expected_cwe in cwes, f"{filename}: expected {expected_cwe}, got {cwes}"


def test_ml_flagged_novel_when_no_signature() -> None:
    code = "int foo(int x) { return x + 1; }"
    hits, _ = analyze_code_segment(code)
    category, _ = classify_novelty(ml_flagged=True, hits=hits)
    assert category == "novel_pattern"


def test_ml_flagged_known_when_signature_matches() -> None:
    code = _read_test("test_2.c")
    hits, _ = analyze_code_segment(code)
    category, novelty = classify_novelty(ml_flagged=True, hits=hits)
    assert category == "known_signature"
    assert novelty < 0.6


def test_attach_signature_attribution_enriches_record() -> None:
    code = _read_test("test_1.c")
    record = {
        "agreement_status": "agree_positive",
        "function_score_calibrated": 0.40,
        "max_window_prob": 0.35,
        "full_code": code,
        "flagged_windows": [{"window_index": 0, "code": code}],
        "flagged_window_indices": [0],
    }
    enriched = attach_signature_attribution(record, function_threshold=0.32)
    attr = enriched["pattern_attribution"]
    assert attr["category"] == "known_signature"
    assert attr["is_known_pattern"] is True
    assert attr["engine"] == "tri_layer_v2"
    assert any(m["cwe"] == "CWE-125" for m in attr["signature_matches"])
    assert enriched.get("signature_risk_boosted") is True
    assert enriched["function_flagged"] is True


def test_match_signatures_legacy_api() -> None:
    hits = match_signatures('system(user_input);')
    assert any(h["cwe"] == "CWE-78" for h in hits)


def test_temporal_filter_strips_post_2019_comment_hints() -> None:
    clear_catalog_cache()
    code = _read_test("test_16.c")
    hints = extract_metadata_hints(code)
    assert "CWE-190" not in hints.cwes
    assert not any(cve.startswith("CVE-2022") for cve in hints.cves)


def test_temporal_filter_keeps_pre_2019_comment_hints() -> None:
    clear_catalog_cache()
    code = _read_test("test_1.c")
    hints = extract_metadata_hints(code)
    assert "CWE-125" in hints.cwes
    assert any(cve.startswith("CVE-2014") for cve in hints.cves)


def test_smooth_signature_boost_formula() -> None:
    assert compute_smooth_signature_boost_score(current_score=0.10, alpha=0.4) == pytest.approx(0.46)
    assert compute_smooth_signature_boost_score(current_score=0.20, alpha=0.4) == pytest.approx(0.52)
    assert compute_smooth_signature_boost_score(current_score=0.70, alpha=0.4) == pytest.approx(0.82)
    assert compute_smooth_signature_boost_score(current_score=1.0, alpha=0.4) == pytest.approx(1.0)


def test_smooth_boost_increases_with_ml_score() -> None:
    low = compute_smooth_signature_boost_score(current_score=0.12, alpha=0.4)
    high = compute_smooth_signature_boost_score(current_score=0.36, alpha=0.4)
    assert low < high
    assert low == pytest.approx(0.472)
    assert high == pytest.approx(0.616)


def test_signature_boost_flags_signature_only_with_smooth_boost() -> None:
    clear_catalog_cache()
    code = _read_test("test_2.c")
    record = {
        "agreement_status": "agree_negative",
        "function_score_calibrated": 0.20,
        "max_window_prob": 0.18,
        "function_flagged": False,
        "user_facing_vuln": False,
        "whole_function_vuln": False,
        "full_code": code,
        "flagged_windows": [],
        "contributing_windows": [{"window_index": 0, "code": code}],
        "thresholds": {"function": 0.32, "window": 0.5, "window_confirmed": 0.6},
    }
    enriched = attach_signature_attribution(record, function_threshold=0.32)
    attr = enriched["pattern_attribution"]
    assert attr["category"] == "signature_only"
    assert attr.get("boost_mode") == "smooth_signature_only"
    assert enriched.get("signature_risk_boosted") is True
    assert enriched["function_score_calibrated"] < 0.846
    assert enriched["function_flagged"] is True
