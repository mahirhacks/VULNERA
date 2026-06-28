"""Pattern attribution: fuse ML risk flags with multi-layer CWE signature engine."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.signature_engine import (
    MetadataHints,
    SignatureHit,
    analyze_code_segment,
    classify_novelty,
    is_known_pattern,
    load_catalog,
    max_confidence,
)

_CATEGORY_LABELS = {
    "novel_pattern": "Novel-pattern vulnerability",
    "known_signature": "Known signature vulnerability",
    "signature_only": "Signature match (ML not flagged)",
    "no_signal": "No signature match",
}


def _ml_flagged(function_record: dict[str, Any]) -> bool:
    status = str(function_record.get("agreement_status", "agree_negative"))
    return status != "agree_negative"


def _collect_code_segments(
    function_record: dict[str, Any],
    *,
    flagged_only: bool,
) -> list[tuple[str, int | None, np.ndarray | None]]:
    """Return (code, window_index, optional_embedding) tuples to scan."""
    segments: list[tuple[str, int | None, np.ndarray | None]] = []
    flagged_indices = {int(i) for i in function_record.get("flagged_window_indices") or []}
    contributing = {int(i) for i in function_record.get("contributing_window_indices") or []}
    target_indices = flagged_indices | contributing if flagged_only else flagged_indices | contributing

    embedding_by_index: dict[int, np.ndarray] = {}
    for pool_key in ("flagged_windows", "contributing_windows", "prompt_windows"):
        for window in function_record.get(pool_key) or []:
            idx = int(window.get("window_index", -1))
            emb = window.get("embedding")
            if emb is not None and idx >= 0:
                embedding_by_index[idx] = np.asarray(emb, dtype=np.float32)

    pools = (
        function_record.get("flagged_windows")
        or function_record.get("contributing_windows")
        or function_record.get("prompt_windows")
        or []
    )
    for window in pools:
        window_index = int(window.get("window_index", -1))
        if flagged_only and window_index not in target_indices and target_indices:
            continue
        code = str(window.get("code") or "")
        if code.strip():
            segments.append((code, window_index, embedding_by_index.get(window_index)))

    full_code = str(function_record.get("full_code") or "")
    if full_code.strip():
        segments.append((full_code, None, None))

    return segments


def _merge_hit_dicts(hit_lists: list[list[SignatureHit]]) -> list[dict[str, Any]]:
    best: dict[str, SignatureHit] = {}
    for hits in hit_lists:
        for hit in hits:
            current = best.get(hit.rule_id)
            if current is None or hit.confidence > current.confidence:
                best[hit.rule_id] = hit
    return [hit.to_dict() for hit in sorted(best.values(), key=lambda h: h.confidence, reverse=True)]


def _category_detail(
    category: str,
    *,
    signature_hits: list[dict[str, Any]],
    metadata: MetadataHints,
    novelty_score: float,
) -> str:
    if category == "novel_pattern":
        detail = (
            "Flagged by the ML stack but no CWE signature reached confidence threshold "
            f"(novelty score {novelty_score:.2f}). Pattern may be outside catalogued classes — "
            "manual review recommended."
        )
        if metadata.cwes:
            detail += f" Comment hints: {', '.join(metadata.cwes)}."
        return detail

    if category == "known_signature" and signature_hits:
        primary = signature_hits[0]
        layer = primary.get("layer", "structural")
        conf = primary.get("confidence", 0.0)
        return (
            f"{primary['name']} ({primary['cwe']}) via {layer} layer "
            f"(confidence {conf:.0%}). ML risk and pattern attribution agree."
        )

    if category == "signature_only" and signature_hits:
        primary = signature_hits[0]
        return (
            f"{primary['name']} ({primary['cwe']}) detected ({primary.get('layer', 'rule')}), "
            "but the ML stack did not flag this function. Review context for true positive."
        )

    return "No CWE signature detected and ML did not flag this function."


def _resolve_function_threshold(function_record: dict[str, Any], function_threshold: float | None) -> float:
    if function_threshold is not None:
        return float(function_threshold)
    thresholds = function_record.get("thresholds") or {}
    if thresholds.get("function") is not None:
        return float(thresholds["function"])
    return 0.32


def ml_support_score(function_record: dict[str, Any]) -> float:
    """Peak ML evidence used for signature corroboration (pre-boost function score vs windows)."""
    pre = function_record.get("function_score_calibrated_pre_signature")
    func = float(pre if pre is not None else function_record.get("function_score_calibrated") or 0.0)
    max_window = float(function_record.get("max_window_prob") or 0.0)
    return max(func, max_window)


def corroboration_allows(ml_support: float, *, threshold: float, omega: float) -> bool:
    return ml_support >= (float(threshold) - float(omega))


def compute_smooth_signature_boost_score(*, current_score: float, alpha: float) -> float:
    """
  Smooth corroboration boost: R' = R + α(1 − R).

  R is the pooled ML function risk (0–1). Always increases score, never exceeds 1.
  """
    r = max(0.0, min(1.0, float(current_score)))
    a = max(0.0, min(1.0, float(alpha)))
    return round(r + a * (1.0 - r), 6)


def compute_signature_plateau_score(
    *,
    current_score: float,
    threshold: float,
    confidence: float,
    boost_cfg: dict[str, Any],
) -> float:
    """Legacy hard-plateau target (upper bound for graduated blending)."""
    min_flag_score = float(boost_cfg.get("min_flag_score", 0.58))
    margin = float(boost_cfg.get("threshold_margin", 0.26))
    boosted = max(current_score, min_flag_score, threshold + margin, confidence * 0.92)
    return round(min(1.0, boosted), 6)


def compute_graduated_boost_score(
    *,
    current_score: float,
    support: float,
    threshold: float,
    confidence: float,
    boost_cfg: dict[str, Any],
) -> float:
    """
    Blend from current ML score toward the plateau target.

    Low ML support → small lift; strong ML + signature confidence → near plateau.
    """
    plateau = compute_signature_plateau_score(
        current_score=current_score,
        threshold=threshold,
        confidence=confidence,
        boost_cfg=boost_cfg,
    )
    grad = boost_cfg.get("graduated") or {}
    min_bump = float(grad.get("min_bump", 0.05))
    conf_ref = float(grad.get("confidence_reference", 0.88))
    support_ref = float(grad.get("support_reference", 1.0))

    conf_ratio = min(1.0, confidence / conf_ref) if conf_ref > 0 else 1.0
    support_ratio = min(1.0, support / (threshold * support_ref)) if threshold > 0 else 0.0
    blend = min(1.0, support_ratio * conf_ratio)

    floor_score = max(current_score, support + min_bump)
    blended = current_score + (plateau - current_score) * blend
    graduated = max(floor_score, blended)
    return round(min(plateau, graduated), 6)


def _refresh_status_display(function_record: dict[str, Any]) -> None:
    try:
        xai_root = Path(__file__).resolve().parents[3] / "07_XAI" / "training_scripts"
        if str(xai_root) not in sys.path:
            sys.path.insert(0, str(xai_root))
        from xai_common import tier_status_meta  # noqa: PLC0415

        function_record["status_display"] = tier_status_meta(function_record)
    except Exception:
        pass


def apply_signature_risk_boost(
    function_record: dict[str, Any],
    *,
    function_threshold: float | None = None,
) -> dict[str, Any]:
    """Raise function risk when a known CWE pattern is detected in the function body."""
    enriched = dict(function_record)
    attr = enriched.get("pattern_attribution") or {}
    category = str(attr.get("category", "no_signal"))
    if category not in ("known_signature", "signature_only"):
        return enriched
    if not attr.get("is_known_pattern"):
        return enriched

    catalog = load_catalog()
    boost_cfg = catalog.get("signature_risk_boost") or {}
    if not bool(boost_cfg.get("enabled", True)):
        return enriched

    confidence = float(attr.get("signature_confidence") or 0.0)
    floor = float(catalog.get("known_confidence_floor", 0.45))
    if confidence < floor:
        return enriched

    threshold = _resolve_function_threshold(enriched, function_threshold)
    corroboration_cfg = catalog.get("corroboration") or {}
    omega = float(corroboration_cfg.get("omega", 0.15))
    corroboration_enabled = bool(corroboration_cfg.get("enabled", True))
    use_smooth = bool(boost_cfg.get("smooth_enabled", True))
    ml_flagged = bool(attr.get("ml_flagged"))

    current_score = float(enriched.get("function_score_calibrated") or 0.0)
    support = ml_support_score(enriched)
    pattern_attr = dict(attr)

    if corroboration_enabled and not corroboration_allows(support, threshold=threshold, omega=omega):
        pattern_attr["corroboration_blocked"] = True
        pattern_attr["corroboration_omega"] = omega
        pattern_attr["ml_support_score"] = round(support, 4)
        pattern_attr["corroboration_floor"] = round(threshold - omega, 4)
        pattern_attr["boost_mode"] = "blocked"
        primary = pattern_attr.get("primary_signature") or {}
        pattern_attr["detail"] = (
            f"{primary.get('name', 'Pattern')} ({primary.get('cwe', 'CWE')}) detected "
            f"(confidence {confidence:.0%}), but ML support {support:.0%} is below the "
            f"corroboration floor {threshold - omega:.0%}. Pattern recorded as hint only."
        )
        enriched["pattern_attribution"] = pattern_attr
        return enriched

    plateau_target = compute_signature_plateau_score(
        current_score=current_score,
        threshold=threshold,
        confidence=confidence,
        boost_cfg=boost_cfg,
    )
    agreement_alpha = float(boost_cfg.get("agreement_alpha", 0.4))
    signature_only_alpha = float(
        boost_cfg.get("signature_only_alpha", agreement_alpha * 0.75)
    )

    if use_smooth and category == "known_signature" and ml_flagged:
        boosted_score = compute_smooth_signature_boost_score(
            current_score=current_score,
            alpha=agreement_alpha,
        )
        boost_mode = "smooth_agreement"
    elif use_smooth and category == "signature_only":
        boosted_score = compute_smooth_signature_boost_score(
            current_score=current_score,
            alpha=signature_only_alpha,
        )
        boost_mode = "smooth_signature_only"
    elif bool(boost_cfg.get("graduated_enabled", False)):
        boosted_score = compute_graduated_boost_score(
            current_score=current_score,
            support=support,
            threshold=threshold,
            confidence=confidence,
            boost_cfg=boost_cfg,
        )
        boost_mode = "graduated"
    else:
        boosted_score = plateau_target
        boost_mode = "plateau"

    if boosted_score <= current_score + 1e-9:
        return enriched

    enriched["function_score_calibrated_pre_signature"] = current_score
    enriched["function_score_calibrated"] = boosted_score
    enriched["function_score_plateau_target"] = plateau_target
    enriched["signature_risk_boosted"] = True

    _SCORE_DIR = Path(__file__).resolve().parents[3] / "05_SCORE"
    if str(_SCORE_DIR) not in sys.path:
        sys.path.insert(0, str(_SCORE_DIR))
    from deployment_tiers import apply_function_deployment_tier  # noqa: PLC0415

    enriched = apply_function_deployment_tier(enriched)

    high_confidence = float(boost_cfg.get("high_confidence", 0.80))
    confirmed_confidence = float(boost_cfg.get("confirmed_confidence", 0.88))
    window_confirmed = float((enriched.get("thresholds") or {}).get("window_confirmed", 0.0))
    max_window_prob = float(enriched.get("max_window_prob") or 0.0)

    if enriched.get("user_facing_vuln") and enriched.get("deployment_tier") == "vuln":
        if (
            boosted_score >= float(boost_cfg.get("confirmed_score_floor", 0.85))
            and (confidence >= confirmed_confidence or max_window_prob >= window_confirmed > 0)
        ):
            enriched["deployment_tier"] = "confirmed"
            enriched["whole_function_vuln"] = True
        elif confidence >= high_confidence:
            enriched["whole_function_vuln"] = True

    pattern_attr = dict(attr)
    pattern_attr["corroboration_blocked"] = False
    pattern_attr["corroboration_omega"] = omega
    pattern_attr["ml_support_score"] = round(support, 4)
    pattern_attr["boost_mode"] = boost_mode
    pattern_attr["plateau_target"] = plateau_target
    pattern_attr["agreement_alpha"] = agreement_alpha if boost_mode == "smooth_agreement" else None
    pattern_attr["risk_boosted"] = True
    pattern_attr["boosted_function_score"] = boosted_score
    if enriched.get("function_flagged") or enriched.get("function_needs_review"):
        primary = pattern_attr.get("primary_signature") or {}
        layer = primary.get("layer", "structural")
        if boost_mode == "smooth_agreement":
            pattern_attr["detail"] = (
                f"{primary.get('name', 'Pattern')} ({primary.get('cwe', 'CWE')}) via {layer} layer "
                f"(confidence {confidence:.0%}). ML and signature agree; smooth boost "
                f"{current_score:.0%} → {boosted_score:.0%} (α={agreement_alpha:.2f})."
            )
        elif boost_mode == "smooth_signature_only":
            pattern_attr["detail"] = (
                f"{primary.get('name', 'Pattern')} ({primary.get('cwe', 'CWE')}) detected via "
                f"{layer} layer (confidence {confidence:.0%}). Signature-only smooth boost "
                f"{current_score:.0%} → {boosted_score:.0%}."
            )
        elif boost_mode == "graduated":
            pattern_attr["detail"] = (
                f"{primary.get('name', 'Pattern')} ({primary.get('cwe', 'CWE')}) via {layer} layer "
                f"(confidence {confidence:.0%}). Graduated boost from ML {support:.0%} to "
                f"{boosted_score:.0%} (plateau cap {plateau_target:.0%})."
            )
        elif category == "signature_only":
            pattern_attr["detail"] = (
                f"{primary.get('name', 'Pattern')} ({primary.get('cwe', 'CWE')}) detected via "
                f"{layer} layer (confidence {confidence:.0%}). Signature classification escalated "
                "this function to vulnerable."
            )
        else:
            pattern_attr["detail"] = (
                f"{primary.get('name', 'Pattern')} ({primary.get('cwe', 'CWE')}) via {layer} layer "
                f"(confidence {confidence:.0%}). ML and signature agree; risk score boosted to "
                f"{boosted_score:.0%}."
            )
    enriched["pattern_attribution"] = pattern_attr
    _refresh_status_display(enriched)
    return enriched


def attach_signature_attribution(
    function_record: dict[str, Any],
    *,
    function_threshold: float | None = None,
) -> dict[str, Any]:
    """Attach multi-layer signature scan + ML×signature novelty category."""
    ml_flagged = _ml_flagged(function_record)
    if not ml_flagged and not function_record.get("full_code"):
        return function_record

    catalog = load_catalog()
    prototypes = function_record.get("cwe_prototype_vectors")
    proto_map = prototypes if isinstance(prototypes, dict) else None

    segment_hit_lists: list[list[SignatureHit]] = []
    merged_metadata = MetadataHints()
    window_hits: dict[int, list[dict[str, Any]]] = {}

    for code, window_index, embedding in _collect_code_segments(function_record, flagged_only=False):
        hits, hints = analyze_code_segment(
            code,
            embedding=embedding,
            prototypes=proto_map,
            catalog=catalog,
        )
        if hits:
            segment_hit_lists.append(hits)
        merged_metadata.cwes.extend(h for h in hints.cwes if h not in merged_metadata.cwes)
        merged_metadata.cves.extend(h for h in hints.cves if h not in merged_metadata.cves)

        if window_index is not None and hits:
            window_hits.setdefault(window_index, [])
            for hit in hits:
                payload = hit.to_dict()
                if payload not in window_hits[window_index]:
                    window_hits[window_index].append(payload)

    signature_matches = _merge_hit_dicts(segment_hit_lists)
    hit_objects = [
        SignatureHit(
            rule_id=str(m["rule_id"]),
            cwe=str(m["cwe"]),
            name=str(m["name"]),
            confidence=float(m["confidence"]),
            layer=str(m.get("layer", "structural")),
            matched_on=str(m.get("matched_on", m["rule_id"])),
            description=str(m.get("description", "")),
        )
        for m in signature_matches
    ]
    category, novelty_score = classify_novelty(ml_flagged=ml_flagged, hits=hit_objects, catalog=catalog)
    primary = signature_matches[0] if signature_matches else None

    enriched = dict(function_record)
    enriched["pattern_attribution"] = {
        "category": category,
        "category_label": _CATEGORY_LABELS[category],
        "ml_flagged": ml_flagged,
        "signature_matches": signature_matches,
        "primary_signature": primary,
        "signature_confidence": round(max_confidence(hit_objects), 4),
        "novelty_score": novelty_score,
        "is_known_pattern": is_known_pattern(hit_objects, catalog),
        "metadata_hints": {
            "cwes": merged_metadata.cwes,
            "cves": merged_metadata.cves,
        },
        "engine": "tri_layer_v2",
        "detail": _category_detail(
            category,
            signature_hits=signature_matches,
            metadata=merged_metadata,
            novelty_score=novelty_score,
        ),
    }

    def enrich_window(window: dict[str, Any]) -> dict[str, Any]:
        window_index = int(window.get("window_index", -1))
        hits = window_hits.get(window_index, [])
        if not hits:
            return window
        return {**window, "signature_matches": hits, "primary_signature": hits[0]}

    for key in ("flagged_windows", "contributing_windows", "prompt_windows"):
        enriched[key] = [enrich_window(dict(window)) for window in (enriched.get(key) or [])]

    enriched.pop("window_embeddings", None)
    enriched.pop("function_embedding", None)
    enriched.pop("cwe_prototype_vectors", None)

    return apply_signature_risk_boost(enriched, function_threshold=function_threshold)


def match_signatures(code: str) -> list[dict[str, Any]]:
    """Analyze a code string and return fused signature match dicts."""
    hits, _ = analyze_code_segment(code)
    return [hit.to_dict() for hit in hits]
