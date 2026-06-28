import { buildAnalysis } from "./analysis";
import { formatPatternBadge, resolvePatternAttribution } from "./patternAttribution";

const STATUS_HIGHLIGHT = {
  agree_positive: "vuln",
  review_suggested: "review",
  diffuse_risk: "diffuse",
  agree_negative: "safe",
  vuln: "vuln",
  needs_review: "review",
  safe: "safe",
};

export const MARKER_FILTERS = [
  { id: "findings", label: "Findings only" },
  { id: "all", label: "All" },
  { id: "vuln", label: "Vulnerable" },
  { id: "review", label: "Needs review" },
  { id: "diffuse", label: "Diffuse risk" },
  { id: "safe", label: "Safe" },
];

/** @deprecated use MARKER_FILTERS */
export const FINDING_FILTERS = MARKER_FILTERS;

export function findingMarkers(markers = []) {
  return markers.filter((m) => m.highlight_kind !== "safe");
}

function locateWindowLines(fullCode, windowCode) {
  const fullLines = fullCode.split("\n");
  const windowStripped = (windowCode || "").trim();
  if (!windowStripped) return [1, Math.max(1, fullLines.length)];

  const position = fullCode.indexOf(windowStripped);
  if (position >= 0) {
    const start = fullCode.slice(0, position).split("\n").length;
    const end = start + windowStripped.split("\n").length - 1;
    return [start, Math.max(start, end)];
  }

  const windowLines = windowStripped.split("\n").filter((line) => line.trim());
  const first = windowLines[0]?.trim() || "";
  for (let index = 0; index < fullLines.length; index += 1) {
    const line = fullLines[index];
    if (line.trim() === first || line.includes(first)) {
      const start = index + 1;
      const end = Math.min(start + windowLines.length - 1, fullLines.length);
      return [start, end];
    }
  }
  return [1, fullLines.length];
}

function mapCleanedLines(lineMap, localStart, localEnd) {
  if (!lineMap?.length) return [localStart, localEnd];
  const startIndex = Math.max(0, Math.min(localStart - 1, lineMap.length - 1));
  const endIndex = Math.max(0, Math.min(localEnd - 1, lineMap.length - 1));
  return [lineMap[startIndex], lineMap[Math.max(startIndex, endIndex)]];
}

function windowMarkerMeta(fn, windowIndex) {
  const tier = fn.deployment_tier;
  const status = fn.agreement_status || "agree_negative";
  const confirmed = new Set((fn.confirmed_window_indices || []).map(Number));
  const flagged = new Set((fn.flagged_window_indices || []).map(Number));
  const contributing = new Set((fn.contributing_window_indices || []).map(Number));

  if (tier && confirmed.has(windowIndex)) {
    if (tier === "confirmed") return { highlight_kind: "vuln", label: "Confirmed vulnerable" };
    if (tier === "investigate") return { highlight_kind: "review", label: "Investigate (localized)" };
    return { highlight_kind: "review", label: "High-confidence window" };
  }
  if (flagged.has(windowIndex)) {
    if (status === "agree_positive") return { highlight_kind: "vuln", label: "Vulnerable" };
    if (status === "review_suggested" || tier === "needs_review") {
      return { highlight_kind: "review", label: "Needs review" };
    }
    return { highlight_kind: "review", label: "Flagged window" };
  }
  if (contributing.has(windowIndex) && status === "diffuse_risk") {
    return { highlight_kind: "diffuse", label: "Diffuse contributor" };
  }
  return { highlight_kind: "safe", label: "Safe" };
}

function buildWindowMarkers(fn) {
  const status = fn.deployment_tier || fn.agreement_status || "agree_negative";
  const functionCode = fn.full_code || "";
  const lineMap = fn.line_map || [];
  const fileEndLine = fn.file_end_line || fn.code_start_line || 1;
  const windows = fn.prompt_windows || [];

  return windows.map((window) => {
    const windowIndex = Number(window.window_index);
    const { highlight_kind, label } = windowMarkerMeta(fn, windowIndex);
    const [localStart, localEnd] = locateWindowLines(functionCode, window.code || "");
    const [startLine, endLineRaw] = mapCleanedLines(lineMap, localStart, localEnd);
    const endLine = Math.min(fileEndLine, endLineRaw);

    return {
      line: startLine,
      end_line: endLine,
      window_index: windowIndex,
      status,
      highlight_kind,
      window_prob: Number(window.window_prob || 0),
      title: `Window ${windowIndex} · ${label}`,
      explanation: (fn.markers || []).find((m) => Number(m.window_index) === windowIndex)?.explanation || "",
      marker_type: "window",
      function_score_calibrated: Number(fn.function_score_calibrated || 0),
      function_id: fn.function_group_id,
      function_name: fn.name,
    };
  });
}

/** Rebuild display markers from function records (works for old and new scans). */
export function buildFileMarkers(scan) {
  const functions = scan?.functions || [];
  const merged = [];

  for (const fn of functions) {
    const status = fn.deployment_tier || fn.agreement_status || "agree_negative";
    const display = fn.status_display || {};
    const alertLine = Number(fn.code_start_line || fn.file_start_line || 1);
    const highlightKind =
      fn.whole_function_vuln ? "vuln" : fn.user_facing_vuln ? "review" : STATUS_HIGHLIGHT[status] || "safe";

    merged.push({
      line: alertLine,
      end_line: alertLine,
      marker_type: "function_alert",
      status,
      highlight_kind: highlightKind,
      window_index: null,
      window_prob: Number(fn.max_window_prob || fn.function_score_calibrated || 0),
      title: `${fn.name || "function"} · ${display.label || status}`,
      explanation: fn.explanation || "",
      function_score_calibrated: Number(fn.function_score_calibrated || 0),
      max_window_prob: Number(fn.max_window_prob || 0),
      function_id: fn.function_group_id,
      function_name: fn.name,
    });

    for (const marker of buildWindowMarkers(fn)) {
      merged.push(marker);
    }
  }

  merged.sort((a, b) => {
    if (a.line !== b.line) return a.line - b.line;
    return a.marker_type === "function_alert" ? -1 : 1;
  });

  return merged;
}

export function markerKey(marker) {
  return [
    marker.marker_type || "window",
    marker.function_id || marker.function_name || "",
    marker.window_index ?? "fn",
    marker.line,
  ].join(":");
}

export function formatGutterAlertTooltip(marker) {
  if (!marker) return "";
  const fnRisk = Number(marker.function_score_calibrated ?? 0);
  const pct = Math.round(fnRisk * 100);
  return `function risk ${pct}%`;
}

export function formatMarkerTooltip(marker) {
  if (!marker) return "";
  const fnRisk = Number(marker.function_score_calibrated ?? 0);
  const parts = [marker.title || "Finding"];
  if (marker.marker_type === "function_alert") {
    const windowRisk = Number(marker.max_window_prob ?? marker.window_prob ?? 0);
    parts.push(`Function risk: ${(fnRisk * 100).toFixed(1)}%`);
    parts.push(`Max window risk: ${(windowRisk * 100).toFixed(1)}%`);
  } else {
    const windowRisk = Number(marker.window_prob ?? 0);
    parts.push(`Function risk: ${(fnRisk * 100).toFixed(1)}%`);
    parts.push(`Window risk: ${(windowRisk * 100).toFixed(1)}%`);
  }
  return parts.join(" — ");
}

export function regionMarkerAtLine(lineNumber, markers = []) {
  return regionMarkers(markers).find((marker) => marker.line === lineNumber) || null;
}

export function filterMarkers(markers, filterId) {
  if (!filterId || filterId === "all") return markers;

  if (filterId === "findings") {
    return markers.filter((m) => m.highlight_kind !== "safe");
  }

  if (filterId === "safe") {
    return markers.filter((m) => m.highlight_kind === "safe");
  }

  const statusByFilter = {
    vuln: "agree_positive",
    review: "review_suggested",
    diffuse: "diffuse_risk",
  };
  const status = statusByFilter[filterId];
  if (!status) return markers;

  return markers.filter((m) => m.status === status || m.highlight_kind === filterId);
}

export function regionMarkers(markers = []) {
  return markers.filter((m) => m.marker_type !== "function_alert");
}

export function functionAlertByLine(markers = []) {
  const map = {};
  for (const marker of markers) {
    if (marker.marker_type === "function_alert") {
      map[marker.line] = marker;
    }
  }
  return map;
}

export function lineClasses(lineNumber, markers = []) {
  const classes = new Set();
  for (const marker of regionMarkers(markers)) {
    const start = marker.line;
    const end = marker.end_line ?? start;
    if (lineNumber >= start && lineNumber <= end) {
      classes.add(`hl-${marker.highlight_kind || "safe"}`);
    }
  }
  return [...classes].join(" ");
}

export function markerAtLine(lineNumber, markers = []) {
  for (const marker of markers) {
    const start = marker.line;
    const end = marker.end_line ?? start;
    if (lineNumber >= start && lineNumber <= end) {
      return marker;
    }
  }
  return null;
}

export function findingContext(scan, marker) {
  const functions = scan.functions || [];
  const fn =
    functions.find((f) => f.function_group_id === marker.function_id) ||
    functions.find((f) => {
      const start = f.file_start_line || 1;
      const end = f.file_end_line || start;
      return marker.line >= start && marker.line <= end;
    }) ||
    {};

  const isFunctionAlert = marker.marker_type === "function_alert";
  const functionRisk = marker.function_score_calibrated ?? fn.function_score_calibrated ?? 0;
  const status = fn.status_display || { label: marker.status || "Finding", color: "#fbbf24" };

  let windowRisk = marker.window_prob ?? 0;
  let windowIndex = marker.window_index ?? null;

  if (isFunctionAlert) {
    const flagged = fn.flagged_window_indices || [];
    const contributing = fn.contributing_window_indices || [];
    windowIndex = flagged[0] ?? contributing[0] ?? null;
    if (windowIndex != null) {
      const windows = fn.prompt_windows || fn.flagged_windows || [];
      const win = windows.find((w) => w.window_index === windowIndex);
      windowRisk = win?.window_prob ?? fn.max_window_prob ?? functionRisk;
    } else {
      windowRisk = fn.max_window_prob ?? functionRisk;
    }
  }

  const storedWindow = (fn.markers || []).find(
    (m) => Number(m.window_index) === Number(marker.window_index),
  );
  let explanation = isFunctionAlert
    ? fn.explanation || marker.explanation || ""
    : marker.explanation || storedWindow?.explanation || "";
  const explanationGrounding =
    marker.explanation_grounding ||
    storedWindow?.explanation_grounding ||
    fn.explanation_grounding ||
    null;

  const patternAttribution = resolvePatternAttribution(fn, marker);
  const patternBadge = formatPatternBadge(patternAttribution);

  const explanationsEnabled = scan.llm_explanation_enabled !== false;

  if (!explanation && explanationsEnabled && !isFunctionAlert && marker.highlight_kind === "safe") {
    const thresholds = fn.thresholds || scan.thresholds || {};
    const windowThreshold = Number(thresholds.window ?? 0.36);
    const prob = Number(marker.window_prob ?? 0);
    explanation = `Window ${marker.window_index} is below the detector threshold (${(prob * 100).toFixed(1)}% vs ${(windowThreshold * 100).toFixed(1)}%). No review is required for this segment.`;
  }

  return {
    title: marker.title || "Finding",
    functionName: marker.function_name || fn.name || "unknown",
    functionRisk,
    windowRisk,
    windowIndex,
    statusLabel: status.label,
    statusColor: status.color,
    lineStart: marker.line,
    lineEnd: marker.end_line ?? marker.line,
    explanation,
    explanationGrounding: explanationsEnabled ? explanationGrounding : null,
    explanationsEnabled,
    explanationScope: isFunctionAlert ? "function" : "window",
    explanationLabel: isFunctionAlert
      ? "Function explanation"
      : windowIndex != null
        ? `Window ${windowIndex} explanation`
        : "Window explanation",
    analysis: buildAnalysis(scan, marker, isFunctionAlert),
    highlightKind: marker.highlight_kind || "safe",
    patternAttribution,
    patternBadge,
  };
}
