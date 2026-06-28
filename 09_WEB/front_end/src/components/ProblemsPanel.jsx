import SeverityIcon from "./SeverityIcon";

export default function ProblemsPanel({
  markers,
  activeMarkerKey,
  onSelect,
  title = "PROBLEMS",
  emptyMessage = "No items match this filter.",
  embedded = false,
}) {
  if (!markers?.length) {
    return <div className={`problems-panel empty${embedded ? " embedded" : ""}`}>{emptyMessage}</div>;
  }

  return (
    <div className={`problems-panel${embedded ? " embedded" : ""}`}>
      {!embedded && (
        <div className="problems-header">
          <span>{title}</span>
          <span className="problems-count">{markers.length}</span>
        </div>
      )}
      {markers.map((marker) => {
        const line = marker.line;
        const end = marker.end_line ?? line;
        const kind = marker.highlight_kind || "safe";
        const lineLabel = line === end ? `L${line}` : `L${line}–${end}`;
        const active = activeMarkerKey === marker._key ? "active" : "";

        return (
          <button
            type="button"
            key={marker._key}
            className={`problem-row ${active} ${kind}`}
            onClick={() => onSelect(marker._key)}
          >
            <SeverityIcon kind={kind} size="md" className="problem-severity" />
            <span className="problem-line">{lineLabel}</span>
            <span className="problem-main">
              <span className="problem-title">{marker.title}</span>
              <span className="problem-fn">{marker.function_name || "global scope"}</span>
            </span>
            <span className={`problem-kind kind-${kind}`}>{kind}</span>
          </button>
        );
      })}
    </div>
  );
}
