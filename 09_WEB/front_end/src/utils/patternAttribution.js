const CATEGORY_LABELS = {
  novel_pattern: "Novel-pattern vulnerability",
  known_signature: "Known signature vulnerability",
  signature_only: "Signature match (ML not flagged)",
  no_signal: "No signature match",
};

export function formatPatternCategory(category) {
  return CATEGORY_LABELS[category] || "Pattern classification";
}

export function formatPatternBadge(attribution) {
  if (!attribution) return null;

  const category = attribution.category || "no_signal";
  const primary = attribution.primary_signature;
  const detail = attribution.detail || "";

  if (category === "novel_pattern") {
    return {
      category,
      label: CATEGORY_LABELS.novel_pattern,
      detail,
      tone: "novel",
    };
  }

  if (category === "known_signature" && primary) {
    const conf = attribution.signature_confidence;
    const layer = primary.layer ? ` · ${primary.layer}` : "";
    const confLabel = conf != null ? ` (${Math.round(conf * 100)}%)` : "";
    return {
      category,
      label: `${primary.cwe} · ${primary.name}${confLabel}`,
      detail,
      tone: "known",
      layer: primary.layer,
    };
  }

  if (category === "signature_only" && primary) {
    return {
      category,
      label: `${primary.cwe} (signature only)`,
      detail,
      tone: "review",
    };
  }

  if (category === "no_signal") {
    return null;
  }

  return {
    category,
    label: formatPatternCategory(category),
    detail,
    tone: "review",
  };
}

export function resolvePatternAttribution(fn, marker) {
  const attr = fn.pattern_attribution;
  if (!attr) return null;

  if (marker?.marker_type === "window" && marker.window_index != null) {
    const pools = [
      ...(fn.flagged_windows || []),
      ...(fn.contributing_windows || []),
      ...(fn.prompt_windows || []),
    ];
    const win = pools.find((w) => Number(w.window_index) === Number(marker.window_index));
    if (win?.primary_signature || win?.signature_matches?.length) {
      const matches = win.signature_matches || [];
      return {
        ...attr,
        signature_matches: matches,
        primary_signature: win.primary_signature || matches[0] || null,
      };
    }
  }

  return attr;
}

export function countPatternSummary(functions = []) {
  let flagged = 0;
  let novel = 0;
  let known = 0;
  let signatureOnly = 0;

  for (const fn of functions) {
    const status = fn.agreement_status || fn.status || "agree_negative";
    if (status === "agree_negative") continue;
    flagged += 1;

    const category = fn.pattern_attribution?.category;
    if (category === "novel_pattern") novel += 1;
    else if (category === "known_signature") known += 1;
    else if (category === "signature_only") signatureOnly += 1;
  }

  return { flagged, novel, known, signatureOnly };
}
