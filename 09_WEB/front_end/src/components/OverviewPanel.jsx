import SeverityIcon from "./SeverityIcon";
import { formatRiskPct } from "../utils/fileScore";
import { countPatternSummary } from "../utils/patternAttribution";

export default function OverviewPanel({
  fileScore,
  functionThreshold,
  onSelectFunction,
  embedded = true,
}) {
  const functions = fileScore?.functions || [];
  const fileRisk = Number(fileScore?.file_risk_calibrated ?? 0);
  const threshold = Number(functionThreshold ?? 0.29);
  const fileKind = fileRisk >= threshold ? "vuln" : fileRisk >= threshold * 0.75 ? "review" : "safe";
  const flaggedCount = functions.filter((entry) => entry.status_kind && entry.status_kind !== "safe").length;
  const contributorCount = (fileScore?.max_pool_contributor_ids || []).length;
  const patternSummary = countPatternSummary(fileScore?.functions || []);

  if (!functions.length) {
    return (
      <div className={`overview-panel empty${embedded ? " embedded" : ""}`}>
        No functions were analyzed.
      </div>
    );
  }

  return (
    <div className={`overview-panel${embedded ? " embedded" : ""}`}>
      <div className="overview-file-risk">
        <div className="overview-file-risk-main">
          <div>
            <div className="overview-file-risk-label">File risk</div>
            <div className={`overview-file-risk-value kind-${fileKind}`}>
              <SeverityIcon kind={fileKind} size="lg" className="overview-file-risk-icon" />
              <span>{formatRiskPct(fileRisk)}</span>
            </div>
          </div>
          <span className={`overview-file-risk-pill kind-${fileKind}`}>
            {fileKind === "vuln" ? "Review required" : fileKind === "review" ? "Needs review" : "Low risk"}
          </span>
        </div>
        <div className="overview-summary-grid">
          <div className="overview-summary-card">
            <span>Functions</span>
            <strong>{functions.length}</strong>
          </div>
          <div className="overview-summary-card">
            <span>Flagged</span>
            <strong>{flaggedCount}</strong>
          </div>
          <div className="overview-summary-card">
            <span>Contributors</span>
            <strong>{contributorCount}</strong>
          </div>
          <div className="overview-summary-card">
            <span>Threshold</span>
            <strong>{formatRiskPct(threshold)}</strong>
          </div>
          {patternSummary.flagged > 0 ? (
            <>
              <div className="overview-summary-card">
                <span>Novel-pattern</span>
                <strong>{patternSummary.novel}</strong>
              </div>
              <div className="overview-summary-card">
                <span>Known signature</span>
                <strong>{patternSummary.known}</strong>
              </div>
            </>
          ) : null}
        </div>
      </div>

      <div className="overview-fn-list">
        {functions.map((entry) => {
          const isContributor = (fileScore.max_pool_contributor_ids || []).includes(entry.function_id);
          const kind = entry.status_kind || "safe";
          const rowClass = `overview-fn-row${isContributor ? " max-pool-contributor" : ""}`;

          if (!onSelectFunction) {
            return (
              <div key={entry.function_id || entry.name} className={rowClass}>
                <SeverityIcon kind={kind} size="md" className="overview-fn-severity" />
                <span className="overview-fn-name">{entry.name}</span>
                <span className={`overview-fn-status kind-${kind}`}>{entry.status_label}</span>
                <span className="overview-fn-risk">{formatRiskPct(entry.calibrated_risk)}</span>
              </div>
            );
          }

          return (
            <button
              key={entry.function_id || entry.name}
              type="button"
              className={`${rowClass} overview-fn-row-btn`}
              onClick={() => onSelectFunction(entry.function_id)}
            >
              <SeverityIcon kind={kind} size="md" className="overview-fn-severity" />
              <span className="overview-fn-name">{entry.name}</span>
              <span className={`overview-fn-status kind-${kind}`}>{entry.status_label}</span>
              <span className="overview-fn-risk">{formatRiskPct(entry.calibrated_risk)}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
