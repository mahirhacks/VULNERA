import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import SeverityIcon from "./SeverityIcon";
import { highlightLine } from "../utils/highlight";
import {
  formatGutterAlertTooltip,
  formatMarkerTooltip,
  functionAlertByLine,
  lineClasses,
  regionMarkerAtLine,
  regionMarkers,
} from "../utils/markers";

const LINE_HEIGHT = 22.4;
const BUFFER_LINES = 100;
const CHAR_WIDTH = 8.2;
const GUTTER_COLUMNS_PX = 96;

function estimateWrappedRows(line, codeWidthPx) {
  if (!line) return 1;
  const charsPerRow = Math.max(8, Math.floor(codeWidthPx / CHAR_WIDTH));
  return Math.max(1, Math.ceil(line.length / charsPerRow));
}

function lineHeightPx(line, codeWidthPx) {
  return estimateWrappedRows(line, codeWidthPx) * LINE_HEIGHT;
}

function buildLineOffsets(lines, codeWidthPx) {
  const offsets = new Array(lines.length + 1);
  let y = 0;
  for (let index = 0; index < lines.length; index += 1) {
    offsets[index] = y;
    y += lineHeightPx(lines[index], codeWidthPx);
  }
  offsets[lines.length] = y;
  return offsets;
}

function findLineIndexAtOffset(offsets, offset) {
  if (!offsets.length) return 0;
  let low = 0;
  let high = offsets.length - 2;
  const y = Math.max(0, offset);
  while (low < high) {
    const mid = Math.floor((low + high + 1) / 2);
    if (offsets[mid] <= y) low = mid;
    else high = mid - 1;
  }
  return low;
}

function computeVisibleRange(offsets, scrollTop, viewportHeight, totalLines) {
  if (!totalLines) return { start: 0, end: 0 };
  const firstVisible = findLineIndexAtOffset(offsets, scrollTop);
  const lastVisible = findLineIndexAtOffset(offsets, scrollTop + viewportHeight);
  return {
    start: Math.max(0, firstVisible - BUFFER_LINES),
    end: Math.min(totalLines, lastVisible + BUFFER_LINES + 1),
  };
}

const CodeLine = memo(function CodeLine({
  lineNo,
  line,
  markers,
  alerts,
  activeMarker,
  onAlertClick,
  onLineActivate,
}) {
  const highlightClasses = lineClasses(lineNo, markers);
  const clickable = Boolean(highlightClasses);
  const isActive = activeMarker
    ? lineNo >= activeMarker.line && lineNo <= (activeMarker.end_line ?? activeMarker.line)
    : false;
  const classes = [
    "code-line",
    highlightClasses,
    isActive ? "hl-active" : "",
    clickable ? "code-line-clickable" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const alert = alerts[lineNo];
  const regionStart = regionMarkerAtLine(lineNo, markers);
  const lineTooltip = alert
    ? formatMarkerTooltip(alert)
    : regionStart
      ? formatMarkerTooltip(regionStart)
      : undefined;

  return (
    <div
      className={classes}
      id={`line-${lineNo}`}
      data-line={lineNo}
      title={lineTooltip}
      onClick={() => onLineActivate(lineNo)}
      onKeyDown={(event) => {
        if (clickable && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          onLineActivate(lineNo);
        }
      }}
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
    >
      {alert ? (
        <button
          type="button"
          className="gutter-alert-btn"
          onClick={(event) => {
            event.stopPropagation();
            onAlertClick(lineNo);
          }}
          title={formatGutterAlertTooltip(alert)}
          aria-label={formatMarkerTooltip(alert)}
        >
          <SeverityIcon kind={alert.highlight_kind || "review"} size="sm" />
        </button>
      ) : (
        <span className="gutter-alert-spacer" aria-hidden="true" />
      )}
      <span className="gutter">{lineNo}</span>
      <span
        className="code-text"
        dangerouslySetInnerHTML={{ __html: highlightLine(line) }}
      />
    </div>
  );
});

export default function CodeEditor({
  source,
  markers,
  activeMarker,
  onAlertClick,
  onLineClick,
  scrollToLine = null,
}) {
  const lines = useMemo(() => source.split("\n"), [source]);
  const totalLines = lines.length;
  const scrollRef = useRef(null);
  const rafRef = useRef(0);
  const regions = useMemo(() => regionMarkers(markers), [markers]);
  const alerts = useMemo(() => functionAlertByLine(markers), [markers]);
  const [editorWidth, setEditorWidth] = useState(0);

  const codeWidthPx = Math.max(200, editorWidth - GUTTER_COLUMNS_PX);
  const lineOffsets = useMemo(
    () => buildLineOffsets(lines, codeWidthPx),
    [codeWidthPx, lines],
  );
  const totalHeight = lineOffsets[totalLines] || 0;

  const [range, setRange] = useState({ start: 0, end: Math.min(totalLines, 200) });

  const syncRange = useCallback(() => {
    const el = scrollRef.current;
    if (!el || totalLines === 0) return;
    const next = computeVisibleRange(
      lineOffsets,
      el.scrollTop,
      el.clientHeight,
      totalLines,
    );
    setRange((prev) => (prev.start === next.start && prev.end === next.end ? prev : next));
  }, [lineOffsets, totalLines]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return undefined;

    const updateWidth = () => {
      setEditorWidth(el.clientWidth);
    };

    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    syncRange();
  }, [syncRange, totalLines, source, lineOffsets]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return undefined;

    const onScroll = () => {
      if (rafRef.current) return;
      rafRef.current = window.requestAnimationFrame(() => {
        rafRef.current = 0;
        syncRange();
      });
    };

    el.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      el.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (rafRef.current) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = 0;
      }
    };
  }, [syncRange]);

  useEffect(() => {
    if (!scrollToLine || !scrollRef.current || totalLines === 0) return;
    const el = scrollRef.current;
    const lineIndex = Math.max(1, Math.min(scrollToLine, totalLines));
    const targetTop = lineOffsets[lineIndex - 1] || 0;
    const lineH = (lineOffsets[lineIndex] || targetTop) - targetTop;
    const centered = targetTop - el.clientHeight / 2 + lineH / 2;
    el.scrollTop = Math.max(0, centered);
    syncRange();
  }, [lineOffsets, scrollToLine, totalLines, syncRange]);

  const lineHasRegion = useCallback(
    (lineNo) =>
      regions.some((marker) => {
        const start = marker.line;
        const end = marker.end_line ?? start;
        return lineNo >= start && lineNo <= end;
      }),
    [regions],
  );

  const handleLineActivate = useCallback(
    (lineNo) => {
      if (onLineClick && lineHasRegion(lineNo)) {
        onLineClick(lineNo);
      }
    },
    [lineHasRegion, onLineClick],
  );

  const topSpacerHeight = lineOffsets[range.start] || 0;
  const bottomSpacerHeight = Math.max(0, totalHeight - (lineOffsets[range.end] || totalHeight));
  const visibleLines = useMemo(
    () => lines.slice(range.start, range.end),
    [lines, range.end, range.start],
  );

  return (
    <div className="vscode-editor-wrap code-editor-virtual">
      <div className="vscode-editor" ref={scrollRef}>
        <div
          className="code-virtual-spacer"
          style={{ height: topSpacerHeight }}
          aria-hidden="true"
        />
        {visibleLines.map((line, index) => {
          const lineNo = range.start + index + 1;
          return (
            <CodeLine
              key={lineNo}
              lineNo={lineNo}
              line={line}
              markers={markers}
              alerts={alerts}
              activeMarker={activeMarker}
              onAlertClick={onAlertClick}
              onLineActivate={handleLineActivate}
            />
          );
        })}
        <div
          className="code-virtual-spacer"
          style={{ height: bottomSpacerHeight }}
          aria-hidden="true"
        />
      </div>
    </div>
  );
}
