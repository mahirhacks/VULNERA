"""Tests for grounded LLM explanation prompts."""

from __future__ import annotations

import sys
from pathlib import Path

XAI_SCRIPTS = Path(__file__).resolve().parents[1] / "07_XAI" / "training_scripts"
if str(XAI_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(XAI_SCRIPTS))

from grounded_explain import (  # noqa: E402
    build_grounded_analysis_prompt,
    build_verification_prompt,
    extract_top_suspicious_tokens,
    parse_verification_response,
    resolve_detection_context,
)


def test_extract_top_tokens_from_dangerous_apis() -> None:
    code = "void f() { char buf[8]; scanf(\"%s\", buf); }"
    tokens = extract_top_suspicious_tokens(code, n=5)
    assert "scanf" in tokens


def test_resolve_detection_context_novel_pattern() -> None:
    function = {
        "name": "foo",
        "function_score_calibrated": 0.35,
        "function_flagged": True,
        "pattern_attribution": {"category": "novel_pattern", "signature_matches": []},
        "prompt_windows": [
            {"window_index": 0, "code": "int x = 1;", "window_prob": 0.34},
        ],
    }
    ctx = resolve_detection_context(function, window_index=0)
    assert ctx["detected_cwe"] == "NOVEL"
    assert "NOVEL" in build_grounded_analysis_prompt(ctx)


def test_build_prompt_includes_cwe_and_tokens() -> None:
    ctx = {
        "function_name": "parse_input",
        "window_index": 0,
        "window_prob": 0.45,
        "function_score": 0.40,
        "pattern_category": "known_signature",
        "pattern_name": "Format string",
        "detected_cwe": "CWE-134",
        "cwe_definition": "Format string: user input as format specifier.",
        "top_tokens": ["printf", "%s"],
        "window_code": 'printf(user_input);',
    }
    prompt = build_grounded_analysis_prompt(ctx)
    assert "CWE-134" in prompt
    assert "printf" in prompt
    assert "Do NOT invent" in prompt


def test_parse_verification_yes() -> None:
    parsed = parse_verification_response("YES\nThis line uses scanf without bounds.")
    assert parsed["verified"] is True
    assert "scanf" in parsed["explanation"]


def test_parse_verification_no_correction() -> None:
    parsed = parse_verification_response("NO\nCorrected: only strcpy is present, not scanf.")
    assert parsed["verified"] is False
    assert "strcpy" in parsed["explanation"]


def test_verification_prompt_shape() -> None:
    prompt = build_verification_prompt(window_code="int a;", proposed_explanation="Bad malloc.")
    assert "YES or NO" in prompt
    assert "int a;" in prompt
