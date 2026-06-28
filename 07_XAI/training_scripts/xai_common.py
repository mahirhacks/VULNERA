"""Shared helpers for XAI prompts and webapp payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

SCRIPTS_ROOT = Path(__file__).resolve().parent
XAI_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = XAI_ROOT.parent
DEFAULT_CONFIG_PATH = XAI_ROOT / "xai_config.yaml"

STATUS_DISPLAY: dict[str, dict[str, str]] = {
    "agree_negative": {
        "label": "Safe",
        "badge": "agree_negative",
        "emoji": "✅",
        "color": "#16a34a",
        "summary": "Function-level and window-level models agree: not flagged.",
    },
    "agree_positive": {
        "label": "Vulnerable",
        "badge": "agree_positive",
        "emoji": "🔴",
        "color": "#dc2626",
        "summary": "Function-level and window-level models agree: likely vulnerable.",
    },
    "review_suggested": {
        "label": "Needs review",
        "badge": "review_suggested",
        "emoji": "🟡",
        "color": "#ca8a04",
        "summary": "Function risk is in the review band (≥26% and below the vulnerable threshold).",
    },
    "diffuse_risk": {
        "label": "Flagged (cross-window risk)",
        "badge": "diffuse_risk",
        "emoji": "🟠",
        "color": "#ea580c",
        "summary": "Function-level score is high but no single window crossed the window threshold.",
    },
}

TIER_DISPLAY: dict[str, dict[str, str]] = {
    "vuln": {
        "label": "Vulnerable",
        "badge": "vuln",
        "emoji": "🔴",
        "color": "#dc2626",
        "summary": "Function risk is at or above the vulnerable threshold (≥32%).",
    },
    "needs_review": {
        "label": "Needs review",
        "badge": "needs_review",
        "emoji": "🟡",
        "color": "#ca8a04",
        "summary": "Function risk is in the review band (≥26% and below 32%).",
    },
    "confirmed": {
        "label": "Confirmed vulnerable",
        "badge": "confirmed",
        "emoji": "🔴",
        "color": "#b91c1c",
        "summary": "Function triage and a high-confidence window agree — mark for remediation.",
    },
    "investigate": {
        "label": "Investigate (localized)",
        "badge": "investigate",
        "emoji": "🟡",
        "color": "#ca8a04",
        "summary": "A high-confidence window hit without whole-function triage — review localized slice only.",
    },
    "soft_review": {
        "label": "Soft review (function only)",
        "badge": "soft_review",
        "emoji": "🟠",
        "color": "#ea580c",
        "summary": "Function score is elevated but no window reached confirmed threshold — do not flag whole function.",
    },
    "safe": STATUS_DISPLAY["agree_negative"],
}

XAI_CONTEXT_BLOCK = """## VULNERA triage alert
Function ID: {function_group_id}
Pipeline status: {status_label} ({agreement_status})
Alert summary: {status_summary}
Calibrated function risk: {function_score:.1%} | function flagged: {function_flagged}
Max window probability: {max_window_prob:.3f} | windows in function: {window_count}
Affected window index(es): {window_indices}
Window-level scores: {window_scores}
"""

PROMPT_TEMPLATES: dict[str, str] = {
    "agree_positive": """{xai_context}
Both detectors agree this C/C++ function is likely vulnerable (agree_positive).

Code windows to review:
{code_blocks}

Explain why the highlighted window(s) are likely vulnerable. Reference the specific unsafe operation or pattern in the code. Be concise (3-5 sentences).""",
    "review_suggested": """{xai_context}
Disagreement detected (review_suggested): at least one window crossed the window threshold while the pooled function-level score did not.

Flagged window code:
{code_blocks}

Explain why the flagged window(s) may contain a vulnerability even though the function-level calibrated score stayed below deployment threshold. Point to the affected window index explicitly. Be concise (3-5 sentences).""",
    "diffuse_risk": """{xai_context}
Disagreement detected (diffuse_risk): function-level score is high, but no single window crossed the window threshold.

Contributing window code (max-pool contributors; may look safe in isolation):
{code_blocks}

No single segment may look unsafe alone, but pooled risk is elevated. Explain possible cross-window interaction or max-pooling effect across these contributors. Be concise (3-5 sentences).""",
    "agree_negative": """{xai_context}
Both function-level and window-level models agree this function is not flagged (agree_negative).

Provide a one-sentence confirmation that no review is required unless the developer has external reason to suspect a bug.""",
}


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def status_meta(status: str) -> dict[str, str]:
    if status not in STATUS_DISPLAY:
        raise KeyError(f"Unknown agreement status: {status}")
    return STATUS_DISPLAY[status]


def tier_status_meta(record: dict[str, Any]) -> dict[str, str]:
    tier = record.get("deployment_tier")
    if tier and tier in TIER_DISPLAY:
        display = dict(TIER_DISPLAY[str(tier)])
        display["agreement_status"] = str(record.get("agreement_status", ""))
        display["deployment_tier"] = str(tier)
        return display
    return status_meta(str(record["agreement_status"]))


def format_code_blocks(windows: list[dict[str, Any]]) -> str:
    if not windows:
        return "(no window code available)"
    blocks: list[str] = []
    for window in windows:
        cwe = window.get("cwe") or "n/a"
        cve = window.get("cve") or "n/a"
        blocks.append(
            f"--- Window {window.get('window_index')} "
            f"(id={window.get('window_id')}, prob={window.get('window_prob', 0):.3f}, CWE={cwe}, CVE={cve}) ---\n"
            f"{window.get('code', '').strip()}"
        )
    return "\n\n".join(blocks)


def build_xai_context_block(record: dict[str, Any]) -> str:
    status = str(record["agreement_status"])
    display = status_meta(status)
    target_windows = record.get("prompt_windows") or []
    highlight = record.get("highlight_window_indices") or []
    if not target_windows and highlight:
        window_indices = ", ".join(str(index) for index in highlight)
        window_scores = "n/a"
    else:
        window_indices = ", ".join(str(window.get("window_index")) for window in target_windows) or "n/a"
        window_scores = ", ".join(f"{window.get('window_prob', 0):.3f}" for window in target_windows) or "n/a"

    return XAI_CONTEXT_BLOCK.format(
        function_group_id=str(record.get("function_group_id", "unknown")),
        status_label=display["label"],
        agreement_status=status,
        status_summary=display["summary"],
        function_score=float(record.get("function_score_calibrated", 0.0)),
        function_flagged=bool(record.get("function_flagged", False)),
        max_window_prob=float(record.get("max_window_prob", 0.0)),
        window_count=int(record.get("window_count", 0)),
        window_indices=window_indices,
        window_scores=window_scores,
    )


def build_prompt(record: dict[str, Any]) -> str:
    status = str(record["agreement_status"])
    if status not in PROMPT_TEMPLATES:
        raise KeyError(f"No prompt template for status: {status}")

    target_windows = record.get("prompt_windows") or []
    highlight = record.get("highlight_window_indices") or []
    if highlight:
        window_indices = ", ".join(str(index) for index in highlight)
    else:
        window_indices = ", ".join(str(window.get("window_index")) for window in target_windows) or "n/a"
    window_scores = ", ".join(f"{window.get('window_prob', 0):.3f}" for window in target_windows) or "n/a"
    return PROMPT_TEMPLATES[status].format(
        xai_context=build_xai_context_block(record),
        function_score=float(record.get("function_score_calibrated", 0.0)),
        window_indices=window_indices,
        window_scores=window_scores,
        code_blocks=format_code_blocks(target_windows),
    )


def build_single_window_prompt(record: dict[str, Any], window_index: int) -> str:
    """Build a prompt that includes exactly one window code block."""
    pools = (
        record.get("prompt_windows") or [],
        record.get("flagged_windows") or [],
        record.get("contributing_windows") or [],
    )
    selected: dict[str, Any] | None = None
    for pool in pools:
        for window in pool:
            if int(window.get("window_index", -1)) == int(window_index):
                selected = dict(window)
                break
        if selected is not None:
            break
    if selected is None:
        raise KeyError(f"Window {window_index} not found for function {record.get('function_group_id')}")

    subset = dict(record)
    subset["prompt_windows"] = [selected]
    subset["highlight_window_indices"] = [int(window_index)]
    return build_prompt(subset)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def json_sanitize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_sanitize(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {key: json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_sanitize(item) for item in value]
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_sanitize(row), ensure_ascii=False) + "\n")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def reconstruct_window_lists(
    record: dict[str, Any],
    window_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def build_windows(id_key: str, index_key: str) -> list[dict[str, Any]]:
        window_ids = _as_list(record.get(id_key))
        window_indices = _as_list(record.get(index_key))
        windows: list[dict[str, Any]] = []
        for window_id, window_index in zip(window_ids, window_indices, strict=False):
            meta = window_lookup.get(str(window_id), {})
            windows.append(
                {
                    "window_id": str(window_id),
                    "window_index": int(window_index),
                    "window_prob": float(meta.get("window_prob", 0.0)),
                }
            )
        return windows

    enriched = dict(record)
    enriched["flagged_windows"] = build_windows("flagged_window_ids", "flagged_window_indices")
    enriched["contributing_windows"] = build_windows("contributing_window_ids", "contributing_window_indices")
    return enriched


def attach_window_code(
    record: dict[str, Any],
    code_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def enrich(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for window in windows:
            meta = code_index.get(str(window.get("window_id")), {})
            enriched.append(
                {
                    **window,
                    "code": meta.get("code", ""),
                    "cwe": meta.get("cwe"),
                    "cve": meta.get("cve"),
                }
            )
        return enriched

    status = str(record["agreement_status"])
    flagged = enrich(record.get("flagged_windows") or [])
    contributing = enrich(record.get("contributing_windows") or [])

    if status in {"agree_positive", "review_suggested"}:
        prompt_windows = flagged or contributing
    elif status == "diffuse_risk":
        prompt_windows = contributing
    else:
        prompt_windows = []

    payload = {
        **record,
        "flagged_windows": flagged,
        "contributing_windows": contributing,
        "prompt_windows": prompt_windows,
        "status_display": status_meta(status),
    }
    payload["highlight_window_indices"] = [
        window["window_index"] for window in (flagged if status != "diffuse_risk" else contributing)
    ]
    return payload
