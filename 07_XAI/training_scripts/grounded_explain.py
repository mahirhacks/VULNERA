"""Grounded LLM prompts: CWE context, suspicious tokens, verification pass."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_ROOT = Path(__file__).resolve().parent
XAI_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = XAI_ROOT.parent
CATALOG_PATH = PROJECT_ROOT / "09_WEB" / "back_end" / "pipeline" / "data" / "signature_catalog.yaml"

_IDENTIFIER_RE = re.compile(
    r"\b("
    r"strcpy|strcat|gets|sprintf|snprintf|printf|fprintf|scanf|sscanf|fscanf|"
    r"memcpy|memmove|memset|malloc|calloc|realloc|free|"
    r"system|popen|execve|execl|"
    r"read|write|recv|send|"
    r"sizeof|strlen|strncpy|strncat"
    r")\b"
)


@lru_cache(maxsize=1)
def load_cwe_definitions() -> dict[str, str]:
    """CWE id -> short definition (from signature catalog + novel fallback)."""
    definitions: dict[str, str] = {
        "NOVEL": (
            "Unattributed ML risk: the ensemble flagged this window but no CWE signature "
            "reached confidence threshold. Explain only concrete unsafe operations visible in the code."
        ),
        "NONE": "No CWE attributed. Explain only patterns directly visible in the code.",
    }
    if not CATALOG_PATH.exists():
        return definitions

    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    for entry in (catalog.get("signatures") or {}).values():
        cwe = str(entry.get("cwe", "")).strip()
        if not cwe:
            continue
        name = str(entry.get("name", cwe))
        desc = str(entry.get("description", "")).strip()
        definitions[cwe] = f"{name}: {desc}" if desc else name
    return definitions


def _window_from_record(record: dict[str, Any], window_index: int) -> dict[str, Any] | None:
    for pool_key in ("prompt_windows", "flagged_windows", "contributing_windows"):
        for window in record.get(pool_key) or []:
            if int(window.get("window_index", -1)) == int(window_index):
                return dict(window)
    return None


def _collect_signature_matches(function: dict[str, Any], window: dict[str, Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in window.get("signature_matches") or []:
        key = str(hit.get("rule_id") or hit.get("matched_on") or hit.get("cwe"))
        if key not in seen:
            seen.add(key)
            matches.append(hit)
    attr = function.get("pattern_attribution") or {}
    for hit in attr.get("signature_matches") or []:
        key = str(hit.get("rule_id") or hit.get("matched_on") or hit.get("cwe"))
        if key not in seen:
            seen.add(key)
            matches.append(hit)
    primary = attr.get("primary_signature")
    if isinstance(primary, dict):
        key = str(primary.get("rule_id") or primary.get("cwe"))
        if key not in seen:
            matches.append(primary)
    return matches


def _tokens_from_signatures(matches: list[dict[str, Any]], code: str) -> list[str]:
    tokens: list[str] = []
    code_lower = code.lower()
    for hit in matches:
        for key in ("matched_on", "name", "rule_id"):
            value = str(hit.get(key) or "").strip()
            if value and value.lower() in code_lower:
                tokens.append(value)
        cwe = str(hit.get("cwe") or "")
        if cwe:
            tokens.append(cwe)
    return tokens


def _tokens_from_catalog_keywords(code: str) -> list[str]:
    code_lower = code.lower()
    found: list[str] = []
    if not CATALOG_PATH.exists():
        return found
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    for entry in (catalog.get("signatures") or {}).values():
        for keyword in entry.get("keywords") or []:
            kw = str(keyword).strip()
            if kw and kw.lower() in code_lower:
                found.append(kw)
        for pattern in entry.get("patterns") or []:
            pat = str(pattern).strip().rstrip("(")
            if pat and pat.lower() in code_lower:
                found.append(pat)
    return found


def _tokens_from_identifiers(code: str) -> list[str]:
    return [m.group(1) for m in _IDENTIFIER_RE.finditer(code)]


def extract_top_suspicious_tokens(
    code: str,
    *,
    signature_matches: list[dict[str, Any]] | None = None,
    n: int = 5,
) -> list[str]:
    """
    Lightweight token saliency (signature + catalog keywords + dangerous APIs).
    Stand-in for SHAP token attribution at scan time without extra model passes.
    """
    if not code.strip():
        return []

    scores: dict[str, float] = {}
    matches = signature_matches or []

    def bump(token: str, weight: float) -> None:
        token = token.strip()
        if not token or len(token) < 2:
            return
        scores[token] = scores.get(token, 0.0) + weight

    for token in _tokens_from_signatures(matches, code):
        bump(token, 3.0)
    for token in _tokens_from_catalog_keywords(code):
        bump(token, 2.0)
    for token in _tokens_from_identifiers(code):
        bump(token, 1.0)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    result: list[str] = []
    seen_lower: set[str] = set()
    for token, _ in ranked:
        key = token.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        result.append(token)
        if len(result) >= n:
            break
    return result


def resolve_detection_context(
    function: dict[str, Any],
    *,
    window_index: int,
    top_token_count: int = 5,
) -> dict[str, Any]:
    """Structured grounding payload for one window."""
    window = _window_from_record(function, window_index) or {}
    code = str(window.get("code") or function.get("full_code") or "").strip()
    attr = function.get("pattern_attribution") or {}
    category = str(attr.get("category") or "no_signal")
    matches = _collect_signature_matches(function, window)

    primary = window.get("primary_signature") or attr.get("primary_signature")
    cwe_id = ""
    pattern_name = ""
    if isinstance(primary, dict):
        cwe_id = str(primary.get("cwe") or "")
        pattern_name = str(primary.get("name") or "")

    if not cwe_id and matches:
        cwe_id = str(matches[0].get("cwe") or "")
        pattern_name = str(matches[0].get("name") or pattern_name)

    if category == "novel_pattern":
        cwe_id = "NOVEL"
        pattern_name = "Novel-pattern vulnerability"
    elif not cwe_id:
        cwe_id = "NONE"

    definitions = load_cwe_definitions()
    cwe_definition = definitions.get(cwe_id, definitions["NONE"])

    token_source = str(window.get("token_attribution_source") or "heuristic")
    shap_tokens = window.get("shap_top_tokens")
    if isinstance(shap_tokens, list) and shap_tokens:
        top_tokens = [str(t) for t in shap_tokens[:top_token_count]]
        token_source = "shap"
    else:
        top_tokens = extract_top_suspicious_tokens(code, signature_matches=matches, n=top_token_count)
        token_source = "heuristic"

    window_prob = float(window.get("window_prob") or function.get("max_window_prob") or 0.0)

    return {
        "function_name": str(function.get("name") or function.get("function_group_id") or "unknown"),
        "window_index": int(window_index),
        "window_code": code,
        "window_prob": window_prob,
        "function_score": float(function.get("function_score_calibrated") or 0.0),
        "function_flagged": bool(function.get("function_flagged")),
        "pattern_category": category,
        "detected_cwe": cwe_id,
        "pattern_name": pattern_name,
        "cwe_definition": cwe_definition,
        "top_tokens": top_tokens,
        "token_attribution_source": token_source,
        "shap_token_scores": window.get("shap_token_scores") or [],
        "signature_confidence": float(attr.get("signature_confidence") or 0.0),
        "signature_matches": matches,
    }


def build_grounded_analysis_prompt(
    context: dict[str, Any],
    *,
    chain_of_thought: bool = False,
) -> str:
    """Suggestion 1 + 2 + 4: structured ML/signature context before code."""
    tokens = context.get("top_tokens") or []
    token_line = ", ".join(f'"{t}"' for t in tokens) if tokens else "(none extracted — cite only code-visible evidence)"
    token_label = (
        "Suspicious tokens (SHAP attribution on window stack)"
        if str(context.get("token_attribution_source")) == "shap"
        else "Suspicious tokens (signature / catalog / API saliency)"
    )

    cot_prefix = (
        "Think step by step about the evidence below, then give your final explanation.\n\n"
        if chain_of_thought
        else ""
    )

    return (
        f"{cot_prefix}"
        "You are a C/C++ vulnerability analyst for the VULNERA triage pipeline.\n\n"
        "The ML ensemble flagged this code window as vulnerable.\n"
        f"- Function: {context.get('function_name')}\n"
        f"- Window index: {context.get('window_index')}\n"
        f"- Window risk (calibrated): {float(context.get('window_prob', 0)):.1%}\n"
        f"- Function risk (calibrated): {float(context.get('function_score', 0)):.1%}\n"
        f"- Pattern category: {context.get('pattern_category')}\n"
        f"- Detected pattern: {context.get('pattern_name') or 'n/a'}\n"
        f"- CWE attribution: {context.get('detected_cwe')}\n"
        f"- CWE definition: {context.get('cwe_definition')}\n"
        f"- {token_label}: [{token_line}]\n\n"
        "Explain specifically how the suspicious tokens and code lines create the detected "
        f"vulnerability pattern ({context.get('detected_cwe')}).\n"
        "Rules:\n"
        "- Reference exact identifiers or lines from the window code.\n"
        "- Do NOT invent CVEs, CWEs, or vulnerabilities not supported by the tokens/code above.\n"
        "- If evidence for the stated CWE is weak, say so explicitly.\n"
        "- Be concise (3–5 sentences).\n\n"
        "Window code:\n"
        f"{context.get('window_code', '').strip()}"
    )


def build_verification_prompt(*, window_code: str, proposed_explanation: str) -> str:
    """Suggestion 3: self-verification pass."""
    return (
        "You are verifying a vulnerability explanation against source code.\n\n"
        "Original window code:\n"
        f"{window_code.strip()}\n\n"
        "Proposed explanation:\n"
        f"{proposed_explanation.strip()}\n\n"
        "Does the explanation correctly identify specific line(s) or tokens in the code?\n"
        "Answer on the first line with YES or NO only.\n"
        "If YES: repeat the explanation unchanged on the following lines.\n"
        "If NO: give a corrected explanation (3–5 sentences) that cites only code that exists."
    )


def parse_verification_response(text: str) -> dict[str, Any]:
    """Parse YES/NO verification output."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return {"verified": False, "explanation": text.strip(), "raw": text}

    first = lines[0].upper()
    verified = first.startswith("YES")
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text.strip()

    if first.startswith("NO") and len(lines) > 1:
        body = "\n".join(lines[1:]).strip()
    elif first.startswith("YES") and not body:
        body = text.strip()

    return {"verified": verified, "explanation": body or text.strip(), "raw": text}


def build_grounded_window_prompt(
    record: dict[str, Any],
    window_index: int,
    *,
    chain_of_thought: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Build grounded prompt + context dict for one window."""
    context = resolve_detection_context(record, window_index=window_index)
    prompt = build_grounded_analysis_prompt(context, chain_of_thought=chain_of_thought)
    return prompt, context
