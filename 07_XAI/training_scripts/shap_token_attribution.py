"""SHAP token attribution for window-level vulnerability scores (GraphCodeBERT + stack)."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_ROOT.parents[1]
AGGREGATOR_SCRIPTS = PROJECT_ROOT / "06_AGGREGATOR" / "training_scripts"

_SPECIAL_TOKEN_RE = re.compile(r"^\[.+\]$")


@dataclass
class ShapTokenResult:
    top_tokens: list[str] = field(default_factory=list)
    token_scores: list[dict[str, Any]] = field(default_factory=list)
    source: str = "shap"
    baseline_prob: float | None = None
    error: str | None = None


def _ensure_aggregator_path() -> None:
    if str(AGGREGATOR_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(AGGREGATOR_SCRIPTS))


def _is_special_token(token: str) -> bool:
    text = str(token).strip()
    if not text:
        return True
    if text in {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}:
        return True
    return bool(_SPECIAL_TOKEN_RE.match(text))


def _merge_subword_pieces(
    tokens: list[str],
    values: list[float] | np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Merge BPE/WordPiece fragments (##, Ġ) before ranking."""
    merged_tokens: list[str] = []
    merged_values: list[float] = []
    current = ""
    current_val = 0.0

    def flush() -> None:
        nonlocal current, current_val
        if current:
            merged_tokens.append(current)
            merged_values.append(current_val)
            current = ""
            current_val = 0.0

    for token, value in zip(tokens, np.asarray(values, dtype=np.float64).reshape(-1)):
        piece = str(token).strip()
        if _is_special_token(piece):
            flush()
            continue
        if piece.startswith("##"):
            current += piece[2:]
            current_val += float(value)
            continue
        if piece.startswith("Ġ"):
            flush()
            current = piece[1:]
            current_val = float(value)
            continue
        flush()
        current = piece
        current_val = float(value)

    flush()
    return merged_tokens, np.asarray(merged_values, dtype=np.float64)


def rank_shap_tokens(
    tokens: list[str],
    values: list[float] | np.ndarray,
    *,
    n: int = 5,
) -> ShapTokenResult:
    """Rank tokenizer pieces by |SHAP| after merging subword fragments."""
    merged_tokens, merged_values = _merge_subword_pieces(tokens, values)
    scores: dict[str, float] = {}
    order: list[str] = []

    for token, value in zip(merged_tokens, merged_values):
        display = str(token).strip()
        if not display or display.isspace():
            continue
        key = display.lower()
        scores[key] = scores.get(key, 0.0) + float(value)
        if key not in {item.lower() for item in order}:
            order.append(display)

    ranked = sorted(scores.items(), key=lambda item: (-abs(item[1]), item[0]))
    top_tokens: list[str] = []
    seen: set[str] = set()
    token_scores: list[dict[str, Any]] = []
    for key, score in ranked:
        if key in seen:
            continue
        seen.add(key)
        display = next((t for t in order if t.lower() == key), key)
        token_scores.append({"token": display, "shap": round(float(score), 6)})
        top_tokens.append(display)
        if len(top_tokens) >= n:
            break

    return ShapTokenResult(top_tokens=top_tokens, token_scores=token_scores, source="shap")


def _default_shap_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    grounded = (settings or {}).get("grounded_explanation") or {}
    shap_cfg = grounded.get("shap_attribution") or {}
    return {
        "enabled": bool(shap_cfg.get("enabled", True)),
        "top_tokens": int(shap_cfg.get("top_tokens", grounded.get("top_tokens", 5))),
        "max_evals": int(shap_cfg.get("max_evals", 80)),
        "batch_size": int(shap_cfg.get("batch_size", 4)),
        "fallback_heuristic": bool(shap_cfg.get("fallback_heuristic", True)),
    }


@lru_cache(maxsize=1)
def _import_shap():
    import shap  # noqa: PLC0415

    return shap


def explain_window_code_shap(
    code: str,
    *,
    encoder: Any,
    bundle: Any,
    n: int = 5,
    max_evals: int = 80,
    batch_size: int = 4,
) -> ShapTokenResult:
    """
    Permutation SHAP over tokenizer pieces: mask tokens -> re-embed -> window stack prob.
    """
    text = str(code or "").strip()
    if not text:
        return ShapTokenResult(source="shap", error="empty code")

    _ensure_aggregator_path()
    from window_stack_common import score_window_embeddings  # noqa: PLC0415

    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        return ShapTokenResult(source="shap", error="encoder has no tokenizer")

    def predict_window_probs(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float64)
        embs = encoder.encode(texts, batch_size=min(batch_size, len(texts)))
        _, probs, _ = score_window_embeddings(
            embs,
            base_models=bundle.base_models,
            feature_columns=bundle.feature_columns,
            meta_model=bundle.meta_model,
            calibrator_bundle=bundle.calibrator_bundle,
        )
        return np.asarray(probs, dtype=np.float64)

    try:
        baseline = float(predict_window_probs([text])[0])
        shap = _import_shap()
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(
            predict_window_probs,
            masker,
            algorithm="permutation",
            output_names=["window_prob"],
        )
        explanation = explainer([text], max_evals=max_evals)
        row = explanation[0]
        tokens = list(getattr(row, "data", []) or [])
        values = np.asarray(getattr(row, "values", []) or [], dtype=np.float64)
        if values.ndim > 1:
            values = values[:, 0]
        ranked = rank_shap_tokens(tokens, values, n=n)
        ranked.baseline_prob = baseline
        return ranked
    except Exception as exc:
        return ShapTokenResult(source="shap", error=str(exc))


def attach_shap_tokens_to_functions(
    functions: list[dict[str, Any]],
    *,
    encoder: Any,
    bundle: Any,
    settings: dict[str, Any] | None = None,
    on_step_complete: Callable[[str], None] | None = None,
) -> int:
    """
    Compute SHAP top tokens for flagged / contributing windows and attach to prompt_windows.
    Returns count of windows attributed.
    """
    cfg = _default_shap_settings(settings)
    if not cfg["enabled"]:
        return 0

    attributed = 0
    n_tokens = int(cfg["top_tokens"])
    max_evals = int(cfg["max_evals"])
    batch_size = int(cfg["batch_size"])

    for function in functions:
        name = str(function.get("name") or "function")
        flagged = {int(i) for i in function.get("flagged_window_indices") or []}
        contributing = {int(i) for i in function.get("contributing_window_indices") or []}
        target_indices = flagged | contributing
        if not target_indices:
            continue

        windows_by_index = {
            int(w.get("window_index", -1)): w
            for w in (function.get("prompt_windows") or [])
        }

        for window_index in sorted(target_indices):
            window = windows_by_index.get(window_index)
            if window is None:
                continue
            code = str(window.get("code") or function.get("full_code") or "").strip()
            if not code:
                continue

            result = explain_window_code_shap(
                code,
                encoder=encoder,
                bundle=bundle,
                n=n_tokens,
                max_evals=max_evals,
                batch_size=batch_size,
            )
            if result.top_tokens:
                window["shap_top_tokens"] = result.top_tokens
                window["shap_token_scores"] = result.token_scores
                window["token_attribution_source"] = "shap"
                if result.baseline_prob is not None:
                    window["shap_baseline_prob"] = result.baseline_prob
                attributed += 1
                if on_step_complete is not None:
                    preview = ", ".join(result.top_tokens[:3])
                    on_step_complete(f"SHAP tokens · window {window_index} · {name} ({preview})")
            elif result.error and on_step_complete is not None:
                on_step_complete(f"SHAP skipped · window {window_index} · {name}")

    return attributed
