"""Tests for SHAP token ranking and grounded context integration."""

from __future__ import annotations

import sys
from pathlib import Path

XAI_SCRIPTS = Path(__file__).resolve().parents[1] / "07_XAI" / "training_scripts"
if str(XAI_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(XAI_SCRIPTS))

from grounded_explain import build_grounded_analysis_prompt, resolve_detection_context  # noqa: E402
from shap_token_attribution import rank_shap_tokens  # noqa: E402


def test_rank_shap_tokens_merges_subwords_and_filters_special() -> None:
    tokens = ["[CLS]", "scan", "##f", "%", "##s", "[SEP]"]
    values = [0.0, 0.8, 0.3, 0.5, 0.4, 0.0]
    result = rank_shap_tokens(tokens, values, n=3)
    assert "scanf" in result.top_tokens
    assert "[CLS]" not in result.top_tokens
    assert result.source == "shap"


def test_rank_shap_tokens_prefers_high_magnitude() -> None:
    tokens = ["strcpy", "buf", "malloc"]
    values = [0.9, 0.1, -0.85]
    result = rank_shap_tokens(tokens, values, n=2)
    assert result.top_tokens[0] in {"strcpy", "malloc"}


def test_resolve_detection_context_prefers_shap_tokens() -> None:
    function = {
        "name": "parse",
        "pattern_attribution": {"category": "known_signature", "signature_matches": []},
        "prompt_windows": [
            {
                "window_index": 0,
                "code": 'scanf("%s", buf);',
                "window_prob": 0.5,
                "shap_top_tokens": ["scanf", "%s", "buf"],
                "token_attribution_source": "shap",
            },
        ],
    }
    ctx = resolve_detection_context(function, window_index=0)
    assert ctx["top_tokens"] == ["scanf", "%s", "buf"]
    assert ctx["token_attribution_source"] == "shap"
    prompt = build_grounded_analysis_prompt(ctx)
    assert "SHAP attribution" in prompt
