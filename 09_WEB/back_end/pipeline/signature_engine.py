"""
Tri-layer vulnerability signature engine for known vs novel pattern attribution.

Research basis
--------------
* **VulSim** (USENIX Security 2024): fuse syntactic, semantic, and contextual code
  properties; nearest-neighbor style similarity to known vulnerability classes.
* **CLeVeR** (ACL 2025 Findings): align code with vulnerability *descriptions* —
  we proxy this with CWE keyword profiles + comment metadata extraction.
* **Vulnerability2Vec** (2025): CWE taxonomy as discrete classification targets.

Layers
------
1. **Structural** — high-precision API sinks, regexes, and token patterns (CodeQL-style).
2. **Semantic** — CWE description keyword overlap (lightweight description alignment).
3. **Contextual** — CWE/CVE hints in comments + optional embedding kNN to prototypes.

Fusion ranks hits by confidence; ``known_signature`` requires confidence >= floor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "signature_catalog.yaml"

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}|\.\./|->|==|!=|<=|>=")


@dataclass(frozen=True)
class SignatureHit:
    rule_id: str
    cwe: str
    name: str
    confidence: float
    layer: str
    matched_on: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "cwe": self.cwe,
            "name": self.name,
            "confidence": round(float(self.confidence), 4),
            "layer": self.layer,
            "matched_on": self.matched_on,
            "description": self.description,
        }


@dataclass
class MetadataHints:
    cwes: list[str] = field(default_factory=list)
    cves: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    with _CATALOG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clear_catalog_cache() -> None:
    load_catalog.cache_clear()


def _cve_year(cve: str) -> int | None:
    match = re.match(r"CVE-(\d{4})-", cve or "", flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _temporal_config(catalog: dict[str, Any]) -> tuple[int, bool]:
    temporal = catalog.get("temporal", {})
    max_year = int(temporal.get("max_disclosure_year", 2019))
    strip_on_leak = bool(temporal.get("strip_comment_hints_when_post_2019_cve", True))
    return max_year, strip_on_leak


def eligible_signatures(catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return signature rules allowed under the catalog temporal cutoff."""
    catalog = catalog or load_catalog()
    max_year, _ = _temporal_config(catalog)
    signatures: dict[str, Any] = catalog.get("signatures", {})
    eligible: dict[str, Any] = {}
    for rule_id, rule in signatures.items():
        rule_year = int(rule.get("max_disclosure_year", max_year))
        if rule_year <= max_year:
            eligible[rule_id] = rule
    return eligible


def _code_blob(code: str) -> str:
    return (code or "").lower()


def _tokenize(code: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(code or "")}


def extract_metadata_hints(code: str, catalog: dict[str, Any] | None = None) -> MetadataHints:
    """Parse CWE/CVE references from comments (contextual layer).

    Comment hints are filtered to disclosures on or before ``max_disclosure_year``.
    When a post-cutoff CVE appears in the file, CWE hints are stripped to avoid
    temporal-split leakage from test-era headers.
    """
    catalog = catalog or load_catalog()
    meta = catalog.get("metadata_patterns", {})
    max_year, strip_on_leak = _temporal_config(catalog)
    cwes: list[str] = []
    cves: list[str] = []

    cwe_re = meta.get("cwe_comment", {}).get("regex", r"CWE[-\s]?(\d{1,4})")
    cve_re = meta.get("cve_comment", {}).get("regex", r"CVE[-\s]?(\d{4}-\d{4,7})")

    for match in re.finditer(cwe_re, code or "", flags=re.IGNORECASE):
        cwes.append(f"CWE-{int(match.group(1))}")
    for match in re.finditer(cve_re, code or "", flags=re.IGNORECASE):
        cves.append(f"CVE-{match.group(1)}")

    # Deduplicate preserving order
    seen_cwe: set[str] = set()
    unique_cwes: list[str] = []
    for cwe in cwes:
        if cwe not in seen_cwe:
            seen_cwe.add(cwe)
            unique_cwes.append(cwe)

    seen_cve: set[str] = set()
    unique_cves: list[str] = []
    for cve in cves:
        if cve not in seen_cve:
            seen_cve.add(cve)
            unique_cves.append(cve)

    cve_years = [_cve_year(cve) for cve in unique_cves]
    has_post_cutoff_cve = any(year is not None and year > max_year for year in cve_years)
    filtered_cves = [
        cve
        for cve, year in zip(unique_cves, cve_years)
        if year is None or year <= max_year
    ]

    if strip_on_leak and has_post_cutoff_cve:
        unique_cwes = []

    return MetadataHints(cwes=unique_cwes, cves=filtered_cves)


def _rule_structural_match(code: str, rule: dict[str, Any]) -> bool:
    text = _code_blob(code)
    patterns: tuple[str, ...] = tuple(rule.get("patterns") or ())
    regexes: tuple[str, ...] = tuple(rule.get("regexes") or ())

    if not patterns and not regexes:
        return False

    pattern_ok = True
    if patterns:
        pattern_ok = any(token in text for token in patterns)

    regex_ok = False
    for pattern in regexes:
        if re.search(pattern, code or "", flags=re.IGNORECASE | re.MULTILINE | re.DOTALL):
            regex_ok = True
            break

    if patterns and regexes:
        structural = pattern_ok and regex_ok
    elif patterns:
        structural = pattern_ok
    else:
        structural = regex_ok

    if not structural:
        return False

    require_any: tuple[str, ...] = tuple(rule.get("require_any") or ())
    if require_any and not any(token in text for token in require_any):
        return False

    return True


def _semantic_score(code: str, rule: dict[str, Any]) -> float:
    keywords: list[str] = [str(k).lower() for k in rule.get("keywords") or []]
    if not keywords:
        return 0.0

    text = _code_blob(code)
    tokens = _tokenize(code)
    hits = 0.0
    for keyword in keywords:
        kw = keyword.lower()
        if " " in kw:
            if kw in text:
                hits += 1.0
        elif kw in tokens or kw in text:
            hits += 1.0
    return hits / len(keywords)


def _cwe_number(cwe: str) -> str:
    digits = re.sub(r"\D", "", cwe or "")
    return digits or ""


def _metadata_hits(
    hints: MetadataHints,
    signatures: dict[str, Any],
    catalog: dict[str, Any],
) -> list[SignatureHit]:
    if not hints.cwes:
        return []

    conf = float(catalog.get("metadata_confidence", 0.88))
    hits: list[SignatureHit] = []
    hint_numbers = {_cwe_number(cwe) for cwe in hints.cwes}

    for rule_id, rule in signatures.items():
        rule_num = _cwe_number(str(rule.get("cwe", "")))
        if rule_num and rule_num in hint_numbers:
            hits.append(
                SignatureHit(
                    rule_id=rule_id,
                    cwe=str(rule["cwe"]),
                    name=str(rule["name"]),
                    confidence=conf,
                    layer="contextual",
                    matched_on="comment_cwe",
                    description=str(rule.get("description", "")),
                )
            )
    return hits


def _structural_hits(code: str, signatures: dict[str, Any], catalog: dict[str, Any]) -> list[SignatureHit]:
    conf = float(catalog.get("structural_confidence", 0.92))
    hits: list[SignatureHit] = []
    for rule_id, rule in signatures.items():
        if _rule_structural_match(code, rule):
            hits.append(
                SignatureHit(
                    rule_id=rule_id,
                    cwe=str(rule["cwe"]),
                    name=str(rule["name"]),
                    confidence=conf,
                    layer="structural",
                    matched_on=rule_id,
                    description=str(rule.get("description", "")),
                )
            )
    return hits


def _semantic_hits(code: str, signatures: dict[str, Any], catalog: dict[str, Any]) -> list[SignatureHit]:
    threshold = float(catalog.get("semantic_threshold", 0.38))
    hits: list[SignatureHit] = []
    for rule_id, rule in signatures.items():
        score = _semantic_score(code, rule)
        if score < threshold:
            continue
        # Scale semantic confidence into [threshold, 0.85]
        confidence = min(0.85, 0.45 + score * 0.5)
        hits.append(
            SignatureHit(
                rule_id=rule_id,
                cwe=str(rule["cwe"]),
                name=str(rule["name"]),
                confidence=confidence,
                layer="semantic",
                matched_on=f"keywords:{score:.2f}",
                description=str(rule.get("description", "")),
            )
        )
    return hits


def _embedding_hits(
    embedding: np.ndarray,
    prototypes: dict[str, np.ndarray],
    signatures: dict[str, Any],
    *,
    similarity_floor: float = 0.72,
) -> list[SignatureHit]:
    """VulSim-style kNN: match code embedding to CWE prototype vectors."""
    if embedding is None or embedding.size == 0 or not prototypes:
        return []

    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return []

    hits: list[SignatureHit] = []
    for rule_id, proto in prototypes.items():
        rule = signatures.get(rule_id)
        if rule is None:
            continue
        p = np.asarray(proto, dtype=np.float32).reshape(-1)
        pnorm = float(np.linalg.norm(p))
        if pnorm < 1e-9:
            continue
        sim = float(np.dot(vec, p) / (norm * pnorm))
        if sim < similarity_floor:
            continue
        hits.append(
            SignatureHit(
                rule_id=rule_id,
                cwe=str(rule["cwe"]),
                name=str(rule["name"]),
                confidence=min(0.9, 0.55 + sim * 0.35),
                layer="embedding",
                matched_on=f"cosine:{sim:.3f}",
                description=str(rule.get("description", "")),
            )
        )
    return hits


def _merge_hits(hit_lists: list[list[SignatureHit]]) -> list[SignatureHit]:
    best: dict[str, SignatureHit] = {}
    for hits in hit_lists:
        for hit in hits:
            current = best.get(hit.rule_id)
            if current is None or hit.confidence > current.confidence:
                best[hit.rule_id] = hit
    merged = sorted(best.values(), key=lambda h: h.confidence, reverse=True)
    return merged


def analyze_code_segment(
    code: str,
    *,
    embedding: np.ndarray | None = None,
    prototypes: dict[str, np.ndarray] | None = None,
    catalog: dict[str, Any] | None = None,
) -> tuple[list[SignatureHit], MetadataHints]:
    catalog = catalog or load_catalog()
    signatures: dict[str, Any] = eligible_signatures(catalog)

    hints = extract_metadata_hints(code, catalog)
    layers = [
        _metadata_hits(hints, signatures, catalog),
        _structural_hits(code, signatures, catalog),
        _semantic_hits(code, signatures, catalog),
    ]
    if embedding is not None and prototypes:
        layers.append(_embedding_hits(embedding, prototypes, signatures))

    return _merge_hits(layers), hints


def max_confidence(hits: list[SignatureHit]) -> float:
    if not hits:
        return 0.0
    return max(hit.confidence for hit in hits)


def is_known_pattern(hits: list[SignatureHit], catalog: dict[str, Any] | None = None) -> bool:
    catalog = catalog or load_catalog()
    floor = float(catalog.get("known_confidence_floor", 0.45))
    return max_confidence(hits) >= floor


def classify_novelty(
    *,
    ml_flagged: bool,
    hits: list[SignatureHit],
    catalog: dict[str, Any] | None = None,
) -> tuple[str, float]:
    """
    Cross ML detector with fused signature confidence.

    Returns (category, novelty_score) where novelty_score in [0,1] —
    higher means more likely a novel / undocumented pattern.
    """
    catalog = catalog or load_catalog()
    known = is_known_pattern(hits, catalog)
    confidence = max_confidence(hits)
    novelty_score = round(1.0 - confidence, 4)

    if ml_flagged and not known:
        return "novel_pattern", novelty_score
    if ml_flagged and known:
        return "known_signature", novelty_score
    if not ml_flagged and known:
        return "signature_only", novelty_score
    return "no_signal", novelty_score
