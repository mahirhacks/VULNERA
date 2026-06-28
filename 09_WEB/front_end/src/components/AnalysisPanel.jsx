function Flag({ active, yes = "Flagged", no = "Below threshold" }) {
  return (
    <span className={`analysis-flag${active ? " on" : ""}`}>{active ? yes : no}</span>
  );
}

function MetricRow({ label, value, extra }) {
  return (
    <div className="analysis-metric-row">
      <span className="analysis-metric-label">{label}</span>
      <span className="analysis-metric-value">{value}</span>
      {extra ? <span className="analysis-metric-extra">{extra}</span> : null}
    </div>
  );
}

function FunctionAnalysis({ analysis }) {
  return (
    <>
      <div className="analysis-block">
        <div className="analysis-block-title">Ensemble trees (function embedding)</div>
        {analysis.trees.map((tree) => (
          <MetricRow key={tree.key} label={tree.label} value={tree.probLabel} />
        ))}
      </div>

      <div className="analysis-block">
        <div className="analysis-block-title">Function-level stack</div>
        <MetricRow label={analysis.meta.label} value={analysis.meta.probLabel} />
        <MetricRow
          label={analysis.calibrated.label}
          value={analysis.calibrated.probLabel}
          extra={`threshold ${analysis.calibrated.thresholdLabel}`}
        />
        <MetricRow
          label="Function decision"
          value={<Flag active={analysis.calibrated.flagged} />}
        />
      </div>

      <div className="analysis-block">
        <div className="analysis-block-title">Aggregation triage</div>
        <MetricRow label="Status" value={analysis.aggregation.statusLabel} />
        <MetricRow
          label="Function flagged"
          value={<Flag active={analysis.aggregation.functionFlagged} />}
        />
        <MetricRow
          label="Any window flagged"
          value={<Flag active={analysis.aggregation.anyWindowFlagged} />}
        />
        <MetricRow
          label="Flagged windows"
          value={`${analysis.aggregation.flaggedWindowCount} / ${analysis.aggregation.windowCount}`}
        />
        <p className="analysis-note">{analysis.aggregation.note}</p>
      </div>

      {analysis.patternAttribution ? (
        <div className="analysis-block">
          <div className="analysis-block-title">ML × signature classification</div>
          <MetricRow label="Category" value={analysis.patternAttribution.categoryLabel} />
          <MetricRow
            label="ML flagged"
            value={<Flag active={analysis.patternAttribution.mlFlagged} />}
          />
          <MetricRow
            label="Primary signature"
            value={
              analysis.patternAttribution.primarySignature
                ? `${analysis.patternAttribution.primarySignature.cwe} · ${analysis.patternAttribution.primarySignature.name}`
                : "None"
            }
          />
          <p className="analysis-note">{analysis.patternAttribution.detail}</p>
        </div>
      ) : null}

      <div className="analysis-block">
        <div className="analysis-block-title">
          {analysis.windowModel.label} · max {analysis.windowModel.maxProbLabel} · threshold{" "}
          {analysis.windowModel.thresholdLabel}
        </div>
        {analysis.windows.length === 0 ? (
          <p className="analysis-note">No windows in this function.</p>
        ) : (
          <div className="analysis-table">
            <div className="analysis-table-head">
              <span>Window</span>
              <span>Prob</span>
              <span>Decision</span>
              <span>Max-pool</span>
            </div>
            {analysis.windows.map((window) => (
              <div key={window.window_index} className="analysis-table-row">
                <span>{window.window_index}</span>
                <span>{window.probLabel}</span>
                <span>
                  <Flag active={window.flagged} />
                </span>
                <span>{window.contributorLabel}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

function WindowAnalysis({ analysis }) {
  const selected = analysis.selectedWindow;

  return (
    <>
      <div className="analysis-block">
        <div className="analysis-block-title">Selected window</div>
        {selected ? (
          <>
            <MetricRow label="Window index" value={selected.window_index} />
            <MetricRow label={`${analysis.windowModel.label} probability`} value={selected.probLabel} />
            <MetricRow
              label="Window threshold"
              value={analysis.windowModel.thresholdLabel}
            />
            <MetricRow
              label="Window decision"
              value={<Flag active={selected.flagged} />}
            />
            <MetricRow
              label="Max-pool contributor"
              value={selected.contributorLabel}
            />
          </>
        ) : (
          <p className="analysis-note">Window details unavailable for this marker.</p>
        )}
      </div>

      <div className="analysis-block">
        <div className="analysis-block-title">Function context (aggregation)</div>
        <MetricRow label="Status" value={analysis.aggregation.statusLabel} />
        <MetricRow
          label={analysis.calibrated.label}
          value={analysis.calibrated.probLabel}
          extra={`threshold ${analysis.calibrated.thresholdLabel}`}
        />
        <MetricRow
          label="Function decision"
          value={<Flag active={analysis.functionContext.functionFlagged} />}
        />
        <MetricRow
          label="Any window flagged"
          value={<Flag active={analysis.aggregation.anyWindowFlagged} />}
        />
        <p className="analysis-note">{analysis.aggregation.note}</p>
      </div>

      <div className="analysis-block">
        <div className="analysis-block-title">Ensemble trees (function-level inputs)</div>
        {analysis.trees.map((tree) => (
          <MetricRow key={tree.key} label={tree.label} value={tree.probLabel} />
        ))}
        <MetricRow label={analysis.meta.label} value={analysis.meta.probLabel} />
      </div>

      <div className="analysis-block">
        <div className="analysis-block-title">All windows in function</div>
        <div className="analysis-table">
          <div className="analysis-table-head">
            <span>Window</span>
            <span>Prob</span>
            <span>Decision</span>
            <span>Max-pool</span>
          </div>
          {analysis.windows.map((window) => (
            <div
              key={window.window_index}
              className={`analysis-table-row${window.active ? " active" : ""}`}
            >
              <span>{window.window_index}</span>
              <span>{window.probLabel}</span>
              <span>
                <Flag active={window.flagged} />
              </span>
              <span>{window.contributorLabel}</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

export default function AnalysisPanel({ analysis }) {
  if (!analysis) return null;
  return (
    <div className="analysis-panel">
      {analysis.scope === "function" ? (
        <FunctionAnalysis analysis={analysis} />
      ) : (
        <WindowAnalysis analysis={analysis} />
      )}
    </div>
  );
}
