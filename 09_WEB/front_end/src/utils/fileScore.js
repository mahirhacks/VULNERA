const STATUS_LABELS = {
  agree_positive: "Vulnerable",
  review_suggested: "Needs review",
  diffuse_risk: "Diffuse risk",
  agree_negative: "Safe",
  vuln: "Vulnerable",
  needs_review: "Needs review",
  confirmed: "Confirmed vulnerable",
  investigate: "Investigate (localized)",
  soft_review: "Soft review (function only)",
  safe: "Safe",
};

const STATUS_KIND = {
  agree_positive: "vuln",
  review_suggested: "review",
  diffuse_risk: "diffuse",
  agree_negative: "safe",
  vuln: "vuln",
  needs_review: "review",
  confirmed: "vuln",
  investigate: "review",
  soft_review: "diffuse",
  safe: "safe",
};

function resolveStatusKey(fn) {
  return fn?.deployment_tier || fn?.agreement_status || "agree_negative";
}

function statusLabel(fn) {
  return fn?.status_display?.label || STATUS_LABELS[resolveStatusKey(fn)] || "Safe";
}

function statusKind(fn) {
  return STATUS_KIND[resolveStatusKey(fn)] || "safe";
}

export const DEFAULT_FUNCTION_THRESHOLD = 0.29;
export const DEFAULT_SPREAD_WEIGHT = 0.25;

function meanExcessAboveThreshold(scores, threshold) {
  if (!scores.length) return 0;
  const total = scores.reduce((sum, score) => sum + Math.max(0, score - threshold), 0);
  return total / scores.length;
}

function otherFunctionScores(scores) {
  if (!scores.length) return [];
  const base = Math.max(...scores);
  return scores.filter((score) => score < base - 1e-9);
}

function compositeFileRisk(scores, { threshold, weight, pooling = "max_plus_mean_excess" }) {
  if (!scores.length) {
    return {
      file_risk_calibrated: 0,
      base_max_risk: 0,
      mean_excess_above_threshold: 0,
      spread_uplift: 0,
      other_function_count: 0,
    };
  }

  const base = Math.max(...scores);
  if (pooling === "window_prob_max" || pooling === "window_max_pool") {
    return {
      file_risk_calibrated: base,
      base_max_risk: base,
      mean_excess_above_threshold: 0,
      spread_uplift: 0,
      other_function_count: Math.max(0, scores.length - 1),
    };
  }

  const others = otherFunctionScores(scores);
  const meanExcess = others.length ? meanExcessAboveThreshold(others, threshold) : 0;
  const uplift = weight * meanExcess;
  return {
    file_risk_calibrated: Math.min(1, base + uplift),
    base_max_risk: base,
    mean_excess_above_threshold: meanExcess,
    spread_uplift: uplift,
    other_function_count: others.length,
  };
}

export function buildFileScore(
  functions = [],
  { threshold = DEFAULT_FUNCTION_THRESHOLD, weight = DEFAULT_SPREAD_WEIGHT, pooling = "max_plus_mean_excess" } = {},
) {
  const resolvedPooling =
    pooling === "window_max_pool" ? "window_prob_max" : pooling;
  const entries = functions.map((fn) => {
    const calibrated = Number(fn.function_score_calibrated ?? 0);
    return {
      function_id: fn.function_group_id,
      name: fn.name || "function",
      calibrated_risk: calibrated,
      status: resolveStatusKey(fn),
      deployment_tier: fn.deployment_tier || null,
      user_facing_vuln: Boolean(fn.user_facing_vuln),
      whole_function_vuln: Boolean(fn.whole_function_vuln),
      status_label: statusLabel(fn),
      status_kind: statusKind(fn),
      function_flagged: Boolean(fn.function_flagged),
      max_window_prob: Number(fn.max_window_prob ?? 0),
      pattern_attribution: fn.pattern_attribution ?? null,
    };
  });

  const scores = entries.map((entry) => entry.calibrated_risk);
  const composite = compositeFileRisk(scores, { threshold, weight, pooling: resolvedPooling });
  const contributorIds = entries
    .filter((entry) => Math.abs(entry.calibrated_risk - composite.base_max_risk) < 1e-9)
    .map((entry) => entry.function_id)
    .filter(Boolean);

  return {
    pooling: resolvedPooling,
    file_risk_calibrated: composite.file_risk_calibrated,
    base_max_risk: composite.base_max_risk,
    mean_excess_above_threshold: composite.mean_excess_above_threshold,
    spread_uplift: composite.spread_uplift,
    other_function_count: composite.other_function_count,
    function_threshold: threshold,
    spread_weight: weight,
    function_count: entries.length,
    functions: entries,
    max_pool_contributor_ids: contributorIds,
  };
}

export function resolveFileScore(scan) {
  const threshold = Number(
    scan?.thresholds?.function ??
      scan?.thresholds?.window ??
      DEFAULT_FUNCTION_THRESHOLD,
  );
  const weight = Number(scan?.file_score?.spread_weight ?? DEFAULT_SPREAD_WEIGHT);
  return buildFileScore(scan?.functions || [], { threshold, weight });
}

export function formatRiskPct(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return `${(num * 100).toFixed(1)}%`;
}
