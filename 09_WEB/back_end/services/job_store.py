from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from pipeline.scan_pipeline import PHASES
from services.project_store import add_scan_to_project
from services.scan_store import save_scan
from services.scan_worker import run_scan_job

_jobs: dict[str, ScanJob] = {}
_jobs_lock = threading.Lock()
_scan_lock = threading.Lock()


@dataclass
class ScanJob:
    job_id: str
    status: str = "pending"
    phase_index: int = 0
    phase_label: str = PHASES[0][1]
    detail: str = ""
    progress: float = 0.0
    scan_id: str | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "phase_index": self.phase_index,
                "phase": PHASES[self.phase_index][0] if self.phase_index < len(PHASES) else "",
                "phase_label": self.phase_label,
                "detail": self.detail,
                "progress": self.progress,
                "scan_id": self.scan_id,
                "error": self.error,
            }

    def update_progress(self, progress: float, detail: str) -> None:
        with self._lock:
            self.progress = max(0.0, min(float(progress), 1.0))
            self.detail = detail
            self.phase_label = "Scanning"
            phase_index = min(int(self.progress * len(PHASES)), len(PHASES) - 1)
            self.phase_index = phase_index
            self.status = "running"


def scan_is_running() -> bool:
    return _scan_lock.locked()


def create_scan_job(
    *,
    source: str,
    filename: str,
    llm_provider: str,
    max_functions: int | None,
    project_id: str | None = None,
) -> ScanJob:
    job_id = uuid.uuid4().hex[:12]
    job = ScanJob(job_id=job_id)

    if not _scan_lock.acquire(blocking=False):
        with job._lock:
            job.status = "failed"
            job.error = "Another scan is already running. Wait for it to finish before starting a new one."
        with _jobs_lock:
            _jobs[job_id] = job
        return job

    def _run() -> None:
        try:
            with _jobs_lock:
                job.status = "running"

            payload = run_scan_job(
                source=source,
                filename=filename,
                llm_provider=llm_provider,
                max_functions=max_functions,
                project_id=project_id,
                progress=job.update_progress,
            )
            save_scan(payload)
            if project_id:
                add_scan_to_project(project_id, payload["scan_id"])
            with job._lock:
                job.scan_id = payload["scan_id"]
                job.status = "completed"
                job.progress = 1.0
                job.detail = "Scan complete · 100%"
        except Exception as exc:
            detail = str(exc).strip() or repr(exc)
            with job._lock:
                job.status = "failed"
                job.error = detail
            print(f"Scan {job_id} failed:\n{traceback.format_exc()}", flush=True)
        finally:
            _scan_lock.release()

    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run, daemon=True, name=f"scan-{job_id}")
    thread.start()
    return job


def get_job(job_id: str) -> ScanJob | None:
    with _jobs_lock:
        return _jobs.get(job_id)
