import { useCallback, useRef, useState } from "react";

const BAR_HEIGHT = 42;
const MIN_BODY = 72;
const DEFAULT_OVERVIEW_BODY = 132;
const DEFAULT_FINDINGS_BODY = 128;
const DEFAULT_DETAILS_BODY = 280;
const COLLAPSE_THRESHOLD = 36;

const PANEL_META = {
  overview: {
    icon: "◇",
    eyebrow: "Risk map",
    description: "File score and functions",
  },
  findings: {
    icon: "!",
    eyebrow: "Triage queue",
    description: "Actionable regions",
  },
  details: {
    icon: "{}",
    eyebrow: "Inspector",
    description: "Scores, pattern, and explanation",
  },
};

function PanelBar({ panelKey, title, count, subtitle, collapsed, onToggle }) {
  const meta = PANEL_META[panelKey];

  return (
    <button
      type="button"
      className="inspector-bar"
      onClick={onToggle}
      aria-expanded={!collapsed}
      title={collapsed ? "Expand panel" : "Collapse panel"}
    >
      <span className="inspector-chevron">{collapsed ? "\u25B8" : "\u25BE"}</span>
      <span className={`inspector-bar-icon ${panelKey}`}>{meta.icon}</span>
      <span className="inspector-bar-copy">
        <span className="inspector-bar-kicker">{meta.eyebrow}</span>
        <span className="inspector-bar-title">{title}</span>
      </span>
      <span className="inspector-bar-subtitle">{subtitle || meta.description}</span>
      {count != null ? (
        <span className="inspector-bar-count">{count}</span>
      ) : null}
    </button>
  );
}

function SplitHandle({ label, variant = "inner", onPointerDown }) {
  return (
    <div
      className={`inspector-split${variant === "bottom" ? " bottom" : ""}`}
      role="separator"
      aria-orientation="horizontal"
      aria-label={label}
      onPointerDown={onPointerDown}
    />
  );
}

function Panel({
  panelKey,
  title,
  count,
  subtitle,
  content,
  collapsed,
  onToggle,
  height,
  fill = false,
}) {
  const panelStyle = collapsed
    ? { height: `${height}px`, flexShrink: 0 }
    : fill
      ? { flex: "1 1 0", minHeight: 0 }
      : { height: `${height}px`, flexShrink: 0 };

  return (
    <section
      className={`inspector-panel ${panelKey} ${collapsed ? "collapsed" : "open"}${fill && !collapsed ? " fill" : ""}`}
      style={panelStyle}
    >
      <PanelBar
        panelKey={panelKey}
        title={title}
        count={count}
        subtitle={subtitle}
        collapsed={collapsed}
        onToggle={onToggle}
      />
      {!collapsed && <div className="inspector-panel-body">{content}</div>}
    </section>
  );
}

export default function InspectorPanels({
  overviewVisible = true,
  findingsVisible = true,
  detailsVisible = true,
  overviewTitle = "Risk Overview",
  overviewSubtitle,
  overviewContent,
  overviewCollapsed,
  onToggleOverview,
  findingsTitle = "Finding Queue",
  findingsCount,
  findingsContent,
  findingsCollapsed,
  onToggleFindings,
  detailsTitle = "Finding Details",
  detailsSubtitle,
  detailsContent,
  detailsCollapsed,
  onToggleDetails,
}) {
  const [overviewBody, setOverviewBody] = useState(DEFAULT_OVERVIEW_BODY);
  const [findingsBody, setFindingsBody] = useState(DEFAULT_FINDINGS_BODY);
  const [detailsBody, setDetailsBody] = useState(DEFAULT_DETAILS_BODY);
  const dragRef = useRef(null);

  const overviewHeight = overviewCollapsed ? BAR_HEIGHT : BAR_HEIGHT + overviewBody;
  const findingsHeight = findingsCollapsed ? BAR_HEIGHT : BAR_HEIGHT + findingsBody;
  const detailsHeight = detailsCollapsed ? BAR_HEIGHT : BAR_HEIGHT + detailsBody;

  const startOverviewFindingsDrag = useCallback(
    (event) => {
      event.preventDefault();
      dragRef.current = {
        mode: "overview_findings",
        startY: event.clientY,
        startOverviewBody: overviewCollapsed ? MIN_BODY : overviewBody,
        startFindingsBody: findingsCollapsed ? MIN_BODY : findingsBody,
        wasOverviewCollapsed: overviewCollapsed,
        wasFindingsCollapsed: findingsCollapsed,
      };

      const onMove = (moveEvent) => {
        const drag = dragRef.current;
        if (!drag || drag.mode !== "overview_findings") return;

        const dy = moveEvent.clientY - drag.startY;
        let nextOverview = drag.startOverviewBody + dy;
        let nextFindings = drag.startFindingsBody - dy;

        if (drag.wasOverviewCollapsed && dy > 6) {
          if (!drag.expandedOverview) {
            drag.expandedOverview = true;
            onToggleOverview();
          }
          setOverviewBody(Math.max(MIN_BODY, nextOverview));
          if (!drag.wasFindingsCollapsed) {
            setFindingsBody(Math.max(MIN_BODY, nextFindings));
          }
          return;
        }

        if (drag.wasFindingsCollapsed && dy < -6) {
          if (!drag.expandedFindings) {
            drag.expandedFindings = true;
            onToggleFindings();
          }
          setFindingsBody(Math.max(MIN_BODY, nextFindings));
          if (!drag.wasOverviewCollapsed) {
            setOverviewBody(Math.max(MIN_BODY, nextOverview));
          }
          return;
        }

        if (nextOverview <= COLLAPSE_THRESHOLD) {
          if (!drag.collapsedOverview && !overviewCollapsed) {
            drag.collapsedOverview = true;
            onToggleOverview();
          }
          if (!drag.wasFindingsCollapsed) {
            setFindingsBody((current) => current + Math.max(0, drag.startOverviewBody + dy));
          }
          return;
        }

        if (nextFindings <= COLLAPSE_THRESHOLD) {
          if (!drag.collapsedFindings && !findingsCollapsed) {
            drag.collapsedFindings = true;
            onToggleFindings();
          }
          if (!drag.wasOverviewCollapsed) {
            setOverviewBody((current) => current + Math.max(0, drag.startFindingsBody - dy));
          }
          return;
        }

        if (overviewCollapsed && !drag.expandedOverview) {
          drag.expandedOverview = true;
          onToggleOverview();
        }
        if (findingsCollapsed && !drag.expandedFindings) {
          drag.expandedFindings = true;
          onToggleFindings();
        }
        setOverviewBody(nextOverview);
        setFindingsBody(nextFindings);
      };

      const onUp = () => {
        dragRef.current = null;
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        document.body.classList.remove("inspector-dragging-v");
      };

      document.body.classList.add("inspector-dragging-v");
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [
      overviewBody,
      findingsBody,
      overviewCollapsed,
      findingsCollapsed,
      onToggleOverview,
      onToggleFindings,
    ],
  );

  const startFindingsDetailsDrag = useCallback(
    (event) => {
      event.preventDefault();
      dragRef.current = {
        mode: "findings_details",
        startY: event.clientY,
        startFindingsBody: findingsCollapsed ? MIN_BODY : findingsBody,
        startDetailsBody: detailsCollapsed ? MIN_BODY : detailsBody,
        wasFindingsCollapsed: findingsCollapsed,
        wasDetailsCollapsed: detailsCollapsed,
      };

      const onMove = (moveEvent) => {
        const drag = dragRef.current;
        if (!drag || drag.mode !== "findings_details") return;

        const dy = moveEvent.clientY - drag.startY;
        let nextFindings = drag.startFindingsBody + dy;
        let nextDetails = drag.startDetailsBody - dy;

        if (drag.wasFindingsCollapsed && dy > 6) {
          if (!drag.expandedFindings) {
            drag.expandedFindings = true;
            onToggleFindings();
          }
          setFindingsBody(Math.max(MIN_BODY, nextFindings));
          if (!drag.wasDetailsCollapsed) {
            setDetailsBody(Math.max(MIN_BODY, nextDetails));
          }
          return;
        }

        if (drag.wasDetailsCollapsed && dy < -6) {
          if (!drag.expandedDetails) {
            drag.expandedDetails = true;
            onToggleDetails();
          }
          setDetailsBody(Math.max(MIN_BODY, nextDetails));
          if (!drag.wasFindingsCollapsed) {
            setFindingsBody(Math.max(MIN_BODY, nextFindings));
          }
          return;
        }

        if (nextFindings <= COLLAPSE_THRESHOLD) {
          if (!drag.collapsedFindings && !findingsCollapsed) {
            drag.collapsedFindings = true;
            onToggleFindings();
          }
          if (!drag.wasDetailsCollapsed) {
            setDetailsBody((current) => current + Math.max(0, drag.startFindingsBody + dy));
          }
          return;
        }

        if (nextDetails <= COLLAPSE_THRESHOLD) {
          if (!drag.collapsedDetails && !detailsCollapsed) {
            drag.collapsedDetails = true;
            onToggleDetails();
          }
          if (!drag.wasFindingsCollapsed) {
            setFindingsBody((current) => current + Math.max(0, drag.startDetailsBody - dy));
          }
          return;
        }

        if (findingsCollapsed && !drag.expandedFindings) {
          drag.expandedFindings = true;
          onToggleFindings();
        }
        if (detailsCollapsed && !drag.expandedDetails) {
          drag.expandedDetails = true;
          onToggleDetails();
        }
        setFindingsBody(Math.max(MIN_BODY, nextFindings));
        setDetailsBody(Math.max(MIN_BODY, nextDetails));
      };

      const onUp = () => {
        dragRef.current = null;
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        document.body.classList.remove("inspector-dragging-v");
      };

      document.body.classList.add("inspector-dragging-v");
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [
      findingsBody,
      detailsBody,
      findingsCollapsed,
      detailsCollapsed,
      onToggleFindings,
      onToggleDetails,
    ],
  );

  const startDetailsBottomDrag = useCallback(
    (event) => {
      event.preventDefault();
      dragRef.current = {
        mode: "details_bottom",
        startY: event.clientY,
        startDetailsBody: detailsCollapsed ? MIN_BODY : detailsBody,
        wasDetailsCollapsed: detailsCollapsed,
      };

      const onMove = (moveEvent) => {
        const drag = dragRef.current;
        if (!drag || drag.mode !== "details_bottom") return;

        const dy = moveEvent.clientY - drag.startY;
        const nextDetails = drag.startDetailsBody + dy;

        if (drag.wasDetailsCollapsed && dy > 6) {
          if (!drag.expandedDetails) {
            drag.expandedDetails = true;
            onToggleDetails();
          }
          setDetailsBody(Math.max(MIN_BODY, nextDetails));
          return;
        }

        if (nextDetails <= COLLAPSE_THRESHOLD) {
          if (!drag.collapsedDetails && !detailsCollapsed) {
            drag.collapsedDetails = true;
            onToggleDetails();
          }
          return;
        }

        if (detailsCollapsed && !drag.expandedDetails) {
          drag.expandedDetails = true;
          onToggleDetails();
        }
        setDetailsBody(Math.max(MIN_BODY, nextDetails));
      };

      const onUp = () => {
        dragRef.current = null;
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        document.body.classList.remove("inspector-dragging-v");
      };

      document.body.classList.add("inspector-dragging-v");
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [detailsBody, detailsCollapsed, onToggleDetails],
  );

  if (!overviewVisible && !findingsVisible && !detailsVisible) {
    return null;
  }

  const showOverviewFindingsSplit = overviewVisible && findingsVisible;
  const showFindingsDetailsSplit = findingsVisible && detailsVisible;

  return (
    <div className="results-inspector side">
      {overviewVisible && (
        <Panel
          panelKey="overview"
          title={overviewTitle}
          subtitle={overviewSubtitle}
          content={overviewContent}
          collapsed={overviewCollapsed}
          onToggle={onToggleOverview}
          height={overviewHeight}
        />
      )}

      {showOverviewFindingsSplit && (
        <SplitHandle
          label="Resize Risk Overview and Finding Queue"
          onPointerDown={startOverviewFindingsDrag}
        />
      )}

      {findingsVisible && (
        <Panel
          panelKey="findings"
          title={findingsTitle}
          count={findingsCount}
          content={findingsContent}
          collapsed={findingsCollapsed}
          onToggle={onToggleFindings}
          height={findingsHeight}
        />
      )}

      {showFindingsDetailsSplit && (
        <SplitHandle
          label="Resize Finding Queue and Finding Details"
          onPointerDown={startFindingsDetailsDrag}
        />
      )}

      {detailsVisible && (
        <div
          className={`inspector-details-zone${detailsCollapsed ? " collapsed" : ""}`}
          style={detailsCollapsed ? undefined : { flex: `1 1 ${detailsBody}px` }}
        >
          <Panel
            panelKey="details"
            title={detailsTitle}
            subtitle={detailsSubtitle}
            content={detailsContent}
            collapsed={detailsCollapsed}
            onToggle={onToggleDetails}
            height={detailsCollapsed ? BAR_HEIGHT : undefined}
            fill={!detailsCollapsed}
          />
          <SplitHandle
            variant="bottom"
            label="Resize Finding Details"
            onPointerDown={startDetailsBottomDrag}
          />
        </div>
      )}
    </div>
  );
}
