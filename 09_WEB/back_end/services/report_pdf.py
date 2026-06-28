from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import sys

from services.paths import PROJECT_ROOT

_SCORE_DIR = PROJECT_ROOT / "05_SCORE"
if str(_SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCORE_DIR))

from file_score import build_file_score  # noqa: E402

STATUS_LABELS: dict[str, str] = {
    "agree_positive": "Vulnerable",
    "review_suggested": "Review suggested",
    "diffuse_risk": "Diffuse risk",
    "agree_negative": "Safe",
}

HIGHLIGHT_LABELS: dict[str, str] = {
    "vuln": "Vulnerable",
    "review": "Review",
    "diffuse": "Diffuse contributor",
    "safe": "Safe",
}


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def _para_text(text: str) -> str:
    return escape(text or "").replace("\n", "<br/>")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=20,
            leading=24,
            spaceAfter=10,
            textColor=colors.HexColor("#1e293b"),
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=6,
        ),
        "section": ParagraphStyle(
            "ReportSection",
            parent=base["Heading2"],
            fontSize=14,
            leading=18,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#0f172a"),
        ),
        "subsection": ParagraphStyle(
            "ReportSubsection",
            parent=base["Heading3"],
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#334155"),
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=6,
        ),
        "muted": ParagraphStyle(
            "ReportMuted",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=4,
        ),
        "code": ParagraphStyle(
            "ReportCode",
            parent=base["Code"],
            fontSize=8,
            leading=10,
            fontName="Courier",
            textColor=colors.HexColor("#0f172a"),
        ),
    }


def _resolve_file_score(scan: dict[str, Any], functions: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds = scan.get("thresholds") or {}
    tau = float(thresholds.get("function", 0.29))
    return build_file_score(functions, threshold=tau)


def _pooling_footnote(file_score: dict[str, Any]) -> str:
    weight = float(file_score.get("spread_weight", 0.25))
    threshold = float(file_score.get("function_threshold", 0.29))
    return (
        "File risk pooling: peak function calibrated risk plus weighted mean excess above "
        f"the function threshold from other functions (w={weight:g}, τ={threshold:.0%}). "
        "A single-function file equals that function's risk."
    )


def _function_status_label(function: dict[str, Any]) -> str:
    display = function.get("status_display") or {}
    label = display.get("label")
    if label:
        return str(label)
    status = str(function.get("agreement_status", "agree_negative"))
    return STATUS_LABELS.get(status, status)


def _function_table(functions: list[dict[str, Any]]) -> Table:
    header = [
        "Function",
        "Disagreement",
        "Function risk",
        "Max window",
        "Flagged",
    ]
    rows: list[list[str]] = [header]
    for function in functions:
        name = function.get("name") or function.get("function_group_id") or "function"
        rows.append(
            [
                escape(str(name)),
                escape(_function_status_label(function)),
                _pct(function.get("function_score_calibrated")),
                _pct(function.get("max_window_prob")),
                "Yes" if function.get("function_flagged") else "No",
            ]
        )

    table = Table(rows, colWidths=[1.35 * inch, 1.35 * inch, 0.95 * inch, 0.95 * inch, 0.65 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#0f172a")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _summary_counts(functions: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"Vulnerable": 0, "Review suggested": 0, "Diffuse risk": 0, "Safe": 0}
    for function in functions:
        status = str(function.get("agreement_status", "agree_negative"))
        label = STATUS_LABELS.get(status, "Safe")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _window_rows(function: dict[str, Any]) -> list[dict[str, Any]]:
    markers = {
        int(marker["window_index"]): marker
        for marker in (function.get("markers") or [])
        if marker.get("window_index") is not None
    }
    rows: list[dict[str, Any]] = []
    for window in function.get("prompt_windows") or []:
        index = int(window.get("window_index", 0))
        marker = markers.get(index, {})
        rows.append(
            {
                "window_index": index,
                "title": marker.get("title") or f"Window {index}",
                "highlight_kind": marker.get("highlight_kind", "safe"),
                "window_prob": window.get("window_prob", marker.get("window_prob", 0.0)),
                "line": marker.get("line"),
                "end_line": marker.get("end_line"),
                "code": window.get("code") or "",
                "explanation": marker.get("explanation") or "",
            }
        )
    if not rows and function.get("markers"):
        for marker in function["markers"]:
            if marker.get("window_index") is None:
                continue
            rows.append(
                {
                    "window_index": int(marker["window_index"]),
                    "title": marker.get("title", f"Window {marker['window_index']}"),
                    "highlight_kind": marker.get("highlight_kind", "safe"),
                    "window_prob": marker.get("window_prob", 0.0),
                    "line": marker.get("line"),
                    "end_line": marker.get("end_line"),
                    "code": "",
                    "explanation": marker.get("explanation") or "",
                }
            )
    rows.sort(key=lambda item: item["window_index"])
    return rows


def build_scan_report_pdf(scan: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    styles = _styles()
    functions = list(scan.get("functions") or [])
    thresholds = scan.get("thresholds") or {}
    file_score = _resolve_file_score(scan, functions)
    summary = _summary_counts(functions)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="VULNERA Security Report",
        author="VULNERA",
    )

    story: list[Any] = []
    uploaded = scan.get("uploaded_at", "")
    try:
        uploaded_display = datetime.fromisoformat(uploaded.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    except ValueError:
        uploaded_display = uploaded or "—"

    story.append(Paragraph("VULNERA Security Report", styles["title"]))
    story.append(Paragraph("Detailed vulnerability triage export", styles["subtitle"]))
    story.append(Spacer(1, 0.12 * inch))

    meta_rows = [
        ["Source file", escape(scan.get("filename", "—"))],
        ["Scan ID", escape(scan.get("scan_id", "—"))],
        ["Uploaded", escape(uploaded_display)],
        ["Functions analyzed", str(len(functions))],
        ["Overall file risk", _pct(file_score.get("file_risk_calibrated"))],
        ["Function threshold", _pct(thresholds.get("function"))],
        ["Window threshold", _pct(thresholds.get("window"))],
    ]
    meta_table = Table(meta_rows, colWidths=[1.55 * inch, 4.7 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748b")),
                ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0f172a")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Executive summary", styles["section"]))
    summary_rows = [
        ["Vulnerable", str(summary["Vulnerable"])],
        ["Review suggested", str(summary["Review suggested"])],
        ["Diffuse risk", str(summary["Diffuse risk"])],
        ["Safe", str(summary["Safe"])],
    ]
    summary_table = Table(summary_rows, colWidths=[2.2 * inch, 1.0 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#334155")),
                ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0f172a")),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("File-level risk overview", styles["section"]))
    story.append(
        Paragraph(
            f"<b>Overall file risk:</b> {_pct(file_score.get('file_risk_calibrated'))}",
            styles["body"],
        )
    )
    story.append(Paragraph(_pooling_footnote(file_score), styles["muted"]))
    story.append(Spacer(1, 0.1 * inch))

    if functions:
        story.append(Paragraph("Per-function summary", styles["subsection"]))
        story.append(_function_table(functions))
    else:
        story.append(Paragraph("No functions were analyzed in this scan.", styles["muted"]))

    story.append(PageBreak())

    story.append(Paragraph("Detailed findings", styles["section"]))
    story.append(
        Paragraph(
            "Per-function and per-window explanations generated by the VULNERA triage pipeline.",
            styles["muted"],
        )
    )

    if not functions:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("No functions were analyzed in this scan.", styles["body"]))
    else:
        for index, function in enumerate(functions, start=1):
            if index > 1:
                story.append(Spacer(1, 0.2 * inch))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))

            status = str(function.get("agreement_status", "agree_negative"))
            status_label = _function_status_label(function)
            name = function.get("name") or function.get("function_group_id") or f"function_{index}"
            lines = function.get("file_start_line"), function.get("file_end_line")
            line_range = (
                f"Lines {lines[0]}–{lines[1]}"
                if lines[0] is not None and lines[1] is not None
                else "Line range unavailable"
            )

            story.append(Paragraph(f"{index}. {escape(name)}", styles["section"]))
            story.append(
                Paragraph(
                    (
                        f"<b>Disagreement:</b> {escape(status_label)} ({escape(status)}) &nbsp; "
                        f"<b>Function risk:</b> {_pct(function.get('function_score_calibrated'))} &nbsp; "
                        f"<b>Max window risk:</b> {_pct(function.get('max_window_prob'))} &nbsp; "
                        f"<b>Flagged:</b> {'Yes' if function.get('function_flagged') else 'No'} &nbsp; "
                        f"<b>{escape(line_range)}</b>"
                    ),
                    styles["muted"],
                )
            )

            story.append(Paragraph("Function explanation", styles["subsection"]))
            story.append(
                Paragraph(_para_text(function.get("explanation", "")), styles["body"])
            )

            windows = _window_rows(function)
            if not windows:
                story.append(Paragraph("No window-level segments were recorded.", styles["muted"]))
                continue

            for window in windows:
                kind = HIGHLIGHT_LABELS.get(
                    str(window.get("highlight_kind", "safe")),
                    str(window.get("highlight_kind", "safe")),
                )
                line_info = ""
                if window.get("line") is not None:
                    end_line = window.get("end_line") or window["line"]
                    line_info = f" · Lines {window['line']}–{end_line}"

                story.append(
                    Paragraph(
                        (
                            f"Window {window['window_index']} · {escape(kind)}"
                            f"{escape(line_info)} · Risk {_pct(window.get('window_prob'))}"
                        ),
                        styles["subsection"],
                    )
                )

                code = str(window.get("code") or "").strip()
                if code:
                    story.append(Paragraph("Code window", styles["muted"]))
                    story.append(
                        Preformatted(
                            code,
                            styles["code"],
                            maxLineLength=96,
                            splitChars="",
                        )
                    )
                    story.append(Spacer(1, 0.05 * inch))

                explanation = str(window.get("explanation") or "").strip()
                story.append(Paragraph("Window explanation", styles["muted"]))
                if explanation:
                    story.append(Paragraph(_para_text(explanation), styles["body"]))
                else:
                    story.append(Paragraph("No explanation was generated for this window.", styles["body"]))

    doc.build(story)
    return buffer.getvalue()


def report_download_filename(scan: dict[str, Any]) -> str:
    stem = Path(scan.get("filename") or "scan").stem or "scan"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return f"{safe_stem}_vulnera_report.pdf"
