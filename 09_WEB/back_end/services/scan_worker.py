"""Run a scan and return a JSON-serializable payload for the scan store."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

ProgressCallback = Callable[[float, str], None]


def run_scan_job(
    *,
    source: str,
    filename: str,
    llm_provider: str,
    max_functions: int | None,
    project_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    from pipeline.scan_pipeline import run_scan

    result = run_scan(
        source=source,
        filename=filename,
        progress=progress,
        llm_provider=llm_provider,
        max_functions=max_functions,
    )
    return {
        "scan_id": result.scan_id,
        "project_id": project_id,
        "filename": result.filename,
        "uploaded_at": result.uploaded_at,
        "source_code": result.source_code,
        "function_count": len(result.functions),
        "functions": result.functions,
        "file_markers": result.file_markers,
        "file_score": result.file_score,
        "phase_log": result.phase_log,
        "thresholds": result.thresholds,
        "llm_explanation_enabled": result.llm_explanation_enabled,
        "llm_chain_of_thought": result.llm_chain_of_thought,
    }
