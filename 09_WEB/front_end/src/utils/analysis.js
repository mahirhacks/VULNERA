const TREE_LABELS = {
  xgb: "XGBoost",
  lightgbm: "LightGBM",
  random_forest: "Random Forest",
  extra_trees: "Extra Trees",
};

const STATUS_NOTES = {
  agree_positive:
    "Function and window detectors agree — likely vulnerable (function flagged and ≥1 window flagged).",
  agree_negative:
    "Function and window detectors agree — not flagged at either level.",
  review_suggested:
    "Disagreement: window(s) crossed threshold but pooled function score did not.",
  diffuse_risk:
    "Disagreement: function flagged but no single window crossed threshold (diffuse / max-pool risk).",
};

function pct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function resolveThresholds(scan, fn) {
  return {
    function: scan?.thresholds?.function ?? fn?.thresholds?.function ?? 0.29,
    window: scan?.thresholds?.window ?? fn?.thresholds?.window ?? 0.36,
  };
}

function resolveWindows(fn) {
  if (fn.all_windows?.length) return fn.all_windows;
  const flagged = new Set(fn.flagged_window_indices || []);
  const contributing = new Set(fn.contributing_window_indices || []);
  const pools = [
    ...(fn.flagged_windows || []),
    ...(fn.contributing_windows || []),
    ...(fn.prompt_windows || []),
  ];
  const byIndex = new Map();
  for (const window of pools) {
    byIndex.set(window.window_index, {
      window_index: window.window_index,
      window_id: window.window_id,
      window_prob: window.window_prob ?? 0,
      flagged: flagged.has(window.window_index),
      max_pool_contributor: contributing.has(window.window_index),
    });
  }
  return [...byIndex.values()].sort((a, b) => a.window_index - b.window_index);
}

function buildFunctionAnalysis(fn, thresholds) {
  const baseProbs = fn.base_probs || {};
  const trees = Object.entries(baseProbs).map(([key, prob]) => ({
    key,
    label: TREE_LABELS[key] || key,
    prob: Number(prob),
    probLabel: pct(prob),
  }));

  const functionFlagged = Boolean(fn.function_flagged);
  const anyWindowFlagged = (fn.flagged_window_indices || []).length > 0;
  const windows = resolveWindows(fn).map((window) => ({
    ...window,
    probLabel: pct(window.window_prob),
    flaggedLabel: window.flagged ? "Flagged" : "Below threshold",
    contributorLabel: window.max_pool_contributor ? "Yes" : "No",
    active: false,
  }));

  return {
    scope: "function",
    trees,
    meta: {
      label: "Meta learner (logistic)",
      prob: Number(fn.raw_function_score ?? 0),
      probLabel: pct(fn.raw_function_score),
    },
    calibrated: {
      label: "Calibrated function score",
      prob: Number(fn.function_score_calibrated ?? 0),
      probLabel: pct(fn.function_score_calibrated),
      threshold: thresholds.function,
      thresholdLabel: pct(thresholds.function),
      flagged: functionFlagged,
      flaggedLabel: functionFlagged ? "Flagged" : "Below threshold",
    },
    aggregation: {
      status: fn.agreement_status || "agree_negative",
      statusLabel: fn.status_display?.label || fn.agreement_status || "Unknown",
      note: STATUS_NOTES[fn.agreement_status] || "",
      functionFlagged,
      anyWindowFlagged,
      flaggedWindowCount: (fn.flagged_window_indices || []).length,
      windowCount: fn.window_count ?? windows.length,
    },
    patternAttribution: fn.pattern_attribution
      ? {
          category: fn.pattern_attribution.category,
          categoryLabel: fn.pattern_attribution.category_label,
          mlFlagged: fn.pattern_attribution.ml_flagged,
          primarySignature: fn.pattern_attribution.primary_signature,
          signatureMatches: fn.pattern_attribution.signature_matches,
          detail: fn.pattern_attribution.detail,
        }
      : null,
    windows,
    windowModel: {
      label: "Window XGBoost (per-window)",
      threshold: thresholds.window,
      thresholdLabel: pct(thresholds.window),
      maxProb: fn.max_window_prob ?? null,
      maxProbLabel: pct(fn.max_window_prob),
    },
  };
}

function buildWindowAnalysis(fn, thresholds, windowIndex) {
  const windows = resolveWindows(fn).map((window) => ({
    ...window,
    probLabel: pct(window.window_prob),
    flaggedLabel: window.flagged ? "Flagged" : "Below threshold",
    contributorLabel: window.max_pool_contributor ? "Yes" : "No",
    active: window.window_index === windowIndex,
  }));
  const selected = windows.find((w) => w.window_index === windowIndex);

  return {
    scope: "window",
    selectedWindow: selected
      ? {
          ...selected,
          threshold: thresholds.window,
          thresholdLabel: pct(thresholds.window),
        }
      : null,
    windowModel: {
      label: "Window XGBoost",
      threshold: thresholds.window,
      thresholdLabel: pct(thresholds.window),
    },
    functionContext: {
      calibrated: Number(fn.function_score_calibrated ?? 0),
      calibratedLabel: pct(fn.function_score_calibrated),
      functionThreshold: thresholds.function,
      functionThresholdLabel: pct(thresholds.function),
      functionFlagged: Boolean(fn.function_flagged),
      functionFlaggedLabel: fn.function_flagged ? "Flagged" : "Below threshold",
    },
    aggregation: {
      status: fn.agreement_status || "agree_negative",
      statusLabel: fn.status_display?.label || fn.agreement_status || "Unknown",
      note: STATUS_NOTES[fn.agreement_status] || "",
      functionFlagged: Boolean(fn.function_flagged),
      anyWindowFlagged: (fn.flagged_window_indices || []).length > 0,
    },
    trees: Object.entries(fn.base_probs || {}).map(([key, prob]) => ({
      key,
      label: TREE_LABELS[key] || key,
      prob: Number(prob),
      probLabel: pct(prob),
    })),
    meta: {
      label: "Meta learner (inputs to function score)",
      prob: Number(fn.raw_function_score ?? 0),
      probLabel: pct(fn.raw_function_score),
    },
    calibrated: {
      label: "Calibrated function score",
      prob: Number(fn.function_score_calibrated ?? 0),
      probLabel: pct(fn.function_score_calibrated),
      threshold: thresholds.function,
      thresholdLabel: pct(thresholds.function),
      flagged: Boolean(fn.function_flagged),
      flaggedLabel: fn.function_flagged ? "Flagged" : "Below threshold",
    },
    windows,
  };
}

export function buildAnalysis(scan, marker, isFunctionAlert) {
  const functions = scan?.functions || [];
  const fn =
    functions.find((f) => f.function_group_id === marker.function_id) ||
    functions.find((f) => {
      const start = f.file_start_line || 1;
      const end = f.file_end_line || start;
      return marker.line >= start && marker.line <= end;
    }) ||
    {};

  const thresholds = resolveThresholds(scan, fn);
  if (isFunctionAlert) {
    return buildFunctionAnalysis(fn, thresholds);
  }
  return buildWindowAnalysis(fn, thresholds, marker.window_index);
}
