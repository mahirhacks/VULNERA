import { useEffect, useState } from "react";
import AnalysisPanel from "./AnalysisPanel";
import SeverityIcon from "./SeverityIcon";

export default function FindingPeek({ context, embedded = false }) {
  const [analysisOpen, setAnalysisOpen] = useState(false);

  useEffect(() => {
    setAnalysisOpen(false);
  }, [context?.title, context?.lineStart, context?.explanationScope, context?.windowIndex]);

  if (!context) {
    return (
      <div className={`finding-peek finding-peek-empty${embedded ? " embedded" : ""}`}>
        <div className="finding-peek-placeholder">
          <p className="finding-peek-placeholder-title">No finding selected</p>
          <p className="finding-peek-placeholder-hint">
            Choose an item from the queue above, or click a highlighted line in the editor.
          </p>
        </div>
      </div>
    );
  }

  const kind = context.highlightKind || "review";
  const windowLabel =
    context.windowIndex != null ? `Window ${context.windowIndex}` : "Window";
  const lineRange =
    context.lineEnd !== context.lineStart
      ? `${context.lineStart}–${context.lineEnd}`
      : `${context.lineStart}`;

  return (
    <div className={`finding-peek open${embedded ? " embedded" : ""}`}>
      <header className="finding-peek-header">
        <div className="finding-peek-identity">
          <SeverityIcon kind={kind} size="lg" className="finding-peek-severity" />
          <div className="finding-peek-title-block">
            <h3 className="finding-peek-title">{context.title}</h3>
            <p className="finding-peek-location">
              <span>{context.functionName || "global scope"}</span>
              <span className="finding-peek-location-sep" aria-hidden="true">
                ·
              </span>
              <span>Lines {lineRange}</span>
            </p>
          </div>
        </div>
        <span
          className={`finding-status-pill kind-${kind}`}
          style={{ "--status-color": context.statusColor }}
        >
          {context.statusLabel}
        </span>
      </header>

      <section className="finding-metrics" aria-label="Risk scores">
        <div className="finding-metric">
          <span className="finding-metric-label">Function risk</span>
          <span className="finding-metric-value" style={{ color: context.statusColor }}>
            {(context.functionRisk * 100).toFixed(1)}%
          </span>
        </div>
        <div className="finding-metric">
          <span className="finding-metric-label">{windowLabel}</span>
          <span className="finding-metric-value">
            {(context.windowRisk * 100).toFixed(1)}%
          </span>
        </div>
      </section>

      {context.patternBadge ? (
        <section className={`finding-pattern-card tone-${context.patternBadge.tone}`}>
          <p className="finding-pattern-label">Pattern classification</p>
          <p className="finding-pattern-title">{context.patternBadge.label}</p>
          <p className="finding-pattern-detail">{context.patternBadge.detail}</p>
        </section>
      ) : null}

      {context.explanationsEnabled !== false ? (
      <section className="finding-explanation-section">
        <h4 className="finding-section-title">
          {context.explanationLabel || "Explanation"}
        </h4>
        {context.explanationGrounding ? (
          <div className="finding-grounding-meta" aria-label="Explanation grounding">
            {context.explanationGrounding.detected_cwe ? (
              <p className="finding-grounding-line">
                <span className="finding-grounding-label">Detected pattern</span>
                <span>{context.explanationGrounding.detected_cwe}</span>
                {context.explanationGrounding.pattern_name
                  ? ` — ${context.explanationGrounding.pattern_name}`
                  : ""}
              </p>
            ) : null}
            {Array.isArray(context.explanationGrounding.top_tokens) &&
            context.explanationGrounding.top_tokens.length > 0 ? (
              <p className="finding-grounding-line">
                <span className="finding-grounding-label">
                  {context.explanationGrounding.token_attribution_source === "shap"
                    ? "SHAP tokens"
                    : "Key tokens"}
                </span>
                <span>{context.explanationGrounding.top_tokens.join(", ")}</span>
              </p>
            ) : null}
            {typeof context.explanationGrounding.verified === "boolean" ? (
              <p className="finding-grounding-line">
                <span className="finding-grounding-label">Verified</span>
                <span>{context.explanationGrounding.verified ? "Yes" : "Corrected"}</span>
              </p>
            ) : null}
          </div>
        ) : null}
        <div className="finding-peek-body">
          {context.explanation || "No explanation was generated for this region."}
        </div>
      </section>
      ) : null}

      <div className="analysis-section">
        <button
          type="button"
          className="analysis-toggle"
          aria-expanded={analysisOpen}
          onClick={() => setAnalysisOpen((open) => !open)}
        >
          <span>Evidence and model signals</span>
          <span className="analysis-chevron">{analysisOpen ? "▾" : "▸"}</span>
        </button>
        {analysisOpen ? <AnalysisPanel analysis={context.analysis} /> : null}
      </div>
    </div>
  );
}
