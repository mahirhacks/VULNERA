import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { exportScanReport, getScan } from "../api/client";
import CodeEditor from "../components/CodeEditor";
import FindingPeek from "../components/FindingPeek";
import InspectorPanels from "../components/InspectorPanels";
import OverviewPanel from "../components/OverviewPanel";
import ProblemsPanel from "../components/ProblemsPanel";
import {
  MARKER_FILTERS,
  buildFileMarkers,
  filterMarkers,
  findingContext,
  findingMarkers,
  markerAtLine,
  markerKey,
} from "../utils/markers";
import { formatRiskPct, resolveFileScore } from "../utils/fileScore";
import "../styles/results.css";

const MIN_INSPECTOR_WIDTH = 280;
const MAX_INSPECTOR_WIDTH = 640;
const DEFAULT_INSPECTOR_WIDTH = 380;

export default function ResultsPage() {
  const { scanId } = useParams();
  const navigate = useNavigate();
  const [scan, setScan] = useState(null);
  const [markerFilter, setMarkerFilter] = useState("findings");
  const [activeMarkerKey, setActiveMarkerKey] = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [overviewVisible, setOverviewVisible] = useState(true);
  const [findingsVisible, setFindingsVisible] = useState(true);
  const [detailsVisible, setDetailsVisible] = useState(true);
  const [overviewCollapsed, setOverviewCollapsed] = useState(false);
  const [findingsCollapsed, setFindingsCollapsed] = useState(false);
  const [detailsCollapsed, setDetailsCollapsed] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [inspectorWidth, setInspectorWidth] = useState(DEFAULT_INSPECTOR_WIDTH);
  const [inspectorResizing, setInspectorResizing] = useState(false);
  const [exportingReport, setExportingReport] = useState(false);
  const [exportError, setExportError] = useState(null);
  const [error, setError] = useState(null);
  const [scrollTargetLine, setScrollTargetLine] = useState(null);
  const menuRef = useRef(null);
  const dragRef = useRef(null);

  const startInspectorResize = useCallback((event) => {
    event.preventDefault();
    setInspectorResizing(true);
    dragRef.current = { startX: event.clientX, startWidth: inspectorWidth };
    const onMove = (moveEvent) => {
      const drag = dragRef.current;
      if (!drag) return;
      const next = drag.startWidth + (drag.startX - moveEvent.clientX);
      setInspectorWidth(Math.min(MAX_INSPECTOR_WIDTH, Math.max(MIN_INSPECTOR_WIDTH, next)));
    };
    const onUp = () => {
      dragRef.current = null;
      setInspectorResizing(false);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.classList.remove("inspector-dragging-h");
    };
    document.body.classList.add("inspector-dragging-h");
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }, [inspectorWidth]);

  const allMarkers = useMemo(
    () => (scan ? buildFileMarkers(scan).map((m) => ({ ...m, _key: markerKey(m) })) : []),
    [scan],
  );

  const findingsOnly = useMemo(() => findingMarkers(allMarkers), [allMarkers]);

  const listMarkers = useMemo(
    () => filterMarkers(allMarkers, markerFilter),
    [allMarkers, markerFilter],
  );

  const fileScore = useMemo(
    () => (scan ? resolveFileScore(scan) : null),
    [scan],
  );

  useEffect(() => {
    getScan(scanId)
      .then((data) => {
        setScan(data);
      })
      .catch((err) => {
        setError(err.message);
        navigate("/");
      });
  }, [scanId, navigate]);

  useEffect(() => {
    if (!allMarkers.length) {
      setActiveMarkerKey(null);
      return;
    }
    const preferred =
      findingsOnly.find((m) => m.marker_type === "function_alert")
      || findingsOnly[0]
      || allMarkers[0];
    setActiveMarkerKey(preferred._key);
  }, [allMarkers, findingsOnly]);

  useEffect(() => {
    if (!menuOpen) return;
    const close = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [menuOpen]);

  useEffect(() => {
    if (!activeMarkerKey || !allMarkers.length) return;
    const marker = allMarkers.find((item) => item._key === activeMarkerKey);
    if (!marker?.line) return;
    setScrollTargetLine(marker.line);
  }, [activeMarkerKey, allMarkers]);

  if (error) return <div className="results-error">{error}</div>;
  if (!scan) return <div className="results-loading">Loading scan…</div>;

  const source = scan.source_code || "";
  const lines = source.split("\n");
  const lineCount = lines.length;

  const activeMarker = allMarkers.find((m) => m._key === activeMarkerKey) || null;
  const activeLine = activeMarker?.line ?? 1;
  const peekContext = activeMarker ? findingContext(scan, activeMarker) : null;

  const revealDetails = () => {
    setInspectorOpen(true);
    setDetailsVisible(true);
    setDetailsCollapsed(false);
  };

  const selectMarker = (key) => {
    setActiveMarkerKey(key);
    revealDetails();
  };

  const selectFunction = (functionId) => {
    const marker = allMarkers.find(
      (item) => item.marker_type === "function_alert" && item.function_id === functionId,
    );
    if (!marker) return;
    setActiveMarkerKey(marker._key);
    revealDetails();
  };

  const handleAlertClick = (line) => {
    const marker = allMarkers.find(
      (item) => item.marker_type === "function_alert" && item.line === line,
    );
    if (marker) selectMarker(marker._key);
  };

  const handleLineClick = (lineNo) => {
    const marker = markerAtLine(lineNo, allMarkers);
    if (marker) selectMarker(marker._key);
  };

  const col = activeLine && activeLine <= lineCount ? lines[activeLine - 1].length + 1 : 1;

  const anyInspectorVisible = overviewVisible || findingsVisible || detailsVisible;

  const handleExportReport = async () => {
    if (!scan?.scan_id || exportingReport) return;
    setExportError(null);
    setExportingReport(true);
    setMenuOpen(false);
    try {
      await exportScanReport(scan.scan_id);
    } catch (err) {
      setExportError(err.message || "Failed to export report.");
    } finally {
      setExportingReport(false);
    }
  };

  return (
    <div className="vscode-results-workspace">
      <div className="vscode-chrome">
        <div className="vscode-titlebar">
          <span className="vscode-title">
            VULNERA — {scan.filename}
            {fileScore ? ` · ${formatRiskPct(fileScore.file_risk_calibrated)} file risk` : ""}
          </span>
          <div className="results-menu-wrap" ref={menuRef}>
            <button
              type="button"
              className="results-menu-btn"
              aria-label="Menu"
              aria-expanded={menuOpen}
              onClick={() => setMenuOpen((open) => !open)}
            >
              Menu
            </button>
            {menuOpen && (
              <div className="results-dropdown" role="menu">
                <button
                  type="button"
                  className="results-dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    setOverviewVisible((visible) => !visible);
                    setMenuOpen(false);
                  }}
                >
                  {overviewVisible ? "Hide OVERVIEW" : "Unhide OVERVIEW"}
                </button>
                <button
                  type="button"
                  className="results-dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    setFindingsVisible((visible) => !visible);
                    setMenuOpen(false);
                  }}
                >
                  {findingsVisible ? "Hide FINDINGS" : "Unhide FINDINGS"}
                </button>
                <button
                  type="button"
                  className="results-dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    setDetailsVisible((visible) => !visible);
                    setMenuOpen(false);
                  }}
                >
                  {detailsVisible ? "Hide DETAILS" : "Unhide DETAILS"}
                </button>
                <button
                  type="button"
                  className="results-dropdown-item"
                  role="menuitem"
                  disabled={exportingReport}
                  onClick={handleExportReport}
                >
                  {exportingReport ? "Exporting Report…" : "Export Report"}
                </button>
              </div>
            )}
          </div>
        </div>
        <div className="vscode-tabbar">
          <div className="vscode-tab active">
            <span className="tab-icon">C</span>
            <span className="tab-label">{scan.filename}</span>
            <span className="tab-close">×</span>
          </div>
          <div className="results-filter-wrap tabbar-filter">
            <label className="results-filter-label" htmlFor="marker-filter">
              Filter
            </label>
            <select
              id="marker-filter"
              className="results-filter"
              value={markerFilter}
              onChange={(e) => setMarkerFilter(e.target.value)}
            >
              {MARKER_FILTERS.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          {anyInspectorVisible && (
            <button
              type="button"
              className={`inspector-toggle-btn${inspectorOpen ? " active" : ""}`}
              onClick={() => setInspectorOpen((open) => !open)}
              aria-pressed={inspectorOpen}
              title={inspectorOpen ? "Hide inspector panel" : "Show inspector panel"}
            >
              <span className="inspector-toggle-glyph" aria-hidden="true" />
              <span>Inspector</span>
            </button>
          )}
        </div>
      </div>

      <div className="vscode-main-row">
        <div className="vscode-editor-stage">
          <CodeEditor
            source={source}
            markers={allMarkers}
            activeMarker={activeMarker}
            onAlertClick={handleAlertClick}
            onLineClick={handleLineClick}
            scrollToLine={scrollTargetLine}
          />
        </div>

        {anyInspectorVisible && (
          <>
            <div
              className={`vscode-vsplit${inspectorOpen ? " open" : " closed"}${inspectorResizing ? " resizing" : ""}`}
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize inspector panel"
              aria-hidden={!inspectorOpen}
              onPointerDown={inspectorOpen ? startInspectorResize : undefined}
            />
            <div
              className={`inspector-dock${inspectorOpen ? " open" : " closed"}${inspectorResizing ? " resizing" : ""}`}
              style={{ "--inspector-width": `${inspectorWidth}px` }}
              aria-hidden={!inspectorOpen}
            >
              <InspectorPanels
                overviewVisible={overviewVisible}
                overviewSubtitle={fileScore ? `File risk ${formatRiskPct(fileScore.file_risk_calibrated)}` : undefined}
                overviewContent={(
                  <OverviewPanel
                    embedded
                    fileScore={fileScore}
                    functionThreshold={scan.thresholds?.function}
                    onSelectFunction={selectFunction}
                  />
                )}
                overviewCollapsed={overviewCollapsed}
                onToggleOverview={() => setOverviewCollapsed((v) => !v)}
                findingsVisible={findingsVisible}
                findingsTitle="Finding Queue"
                findingsCount={listMarkers.length}
                findingsContent={(
                  <ProblemsPanel
                    embedded
                    emptyMessage="No functions or windows match this filter."
                    markers={listMarkers}
                    activeMarkerKey={activeMarkerKey}
                    onSelect={selectMarker}
                  />
                )}
                findingsCollapsed={findingsCollapsed}
                onToggleFindings={() => setFindingsCollapsed((v) => !v)}
                detailsVisible={detailsVisible}
                detailsSubtitle={peekContext?.title}
                detailsContent={<FindingPeek embedded context={peekContext} />}
                detailsCollapsed={detailsCollapsed}
                onToggleDetails={() => setDetailsCollapsed((v) => !v)}
              />
            </div>
          </>
        )}
      </div>

      <div className="vscode-statusbar">
        <span>Ln {activeLine}, Col {col}</span>
        <span>{lineCount} lines</span>
        <span>C</span>
        <span>UTF-8</span>
        <span className="status-right">
          {exportError
            ? exportError
            : `${findingsOnly.length} finding(s) · click safe regions in editor for details`}
        </span>
      </div>
    </div>
  );
}
