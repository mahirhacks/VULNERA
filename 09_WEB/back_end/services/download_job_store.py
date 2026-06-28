from __future__ import annotations

import os
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from services.hf_hub_service import folder_name_from_repo
from services.llm_config_service import (
    _resolve_path,
    get_preset,
    is_downloaded_model_dir,
)


@dataclass
class ModelDownloadJob:
    job_id: str
    preset_id: str | None
    repo_id: str
    target_dir: str
    folder_name: str
    status: str = "pending"
    progress: float = 0.0
    detail: str = "Preparing download…"
    error: str | None = None
    quantization_id: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": self.job_id,
                "preset_id": self.preset_id,
                "repo_id": self.repo_id,
                "target_dir": self.target_dir,
                "folder_name": self.folder_name,
                "status": self.status,
                "progress": self.progress,
                "detail": self.detail,
                "error": self.error,
                "quantization_id": self.quantization_id,
            }

    def update_progress(self, progress: float, detail: str) -> None:
        with self._lock:
            self.progress = max(0.0, min(float(progress), 1.0))
            self.detail = detail
            self.status = "running"


_jobs: dict[str, ModelDownloadJob] = {}
_jobs_lock = threading.Lock()
_download_lock = threading.Lock()


def download_is_running() -> bool:
    return _download_lock.locked()


_DEVNULL = open(os.devnull, "w", encoding="utf-8")


def _make_progress_tqdm(on_update: Callable[[float, str], None]):
    from tqdm.auto import tqdm

    class HubProgressBar(tqdm):
        """tqdm subclass so huggingface_hub thread_map can call get_lock()."""

        def __init__(self, *args, **kwargs):
            # Keep tqdm fully enabled for update()/desc; suppress console bars only.
            kwargs.setdefault("file", _DEVNULL)
            super().__init__(*args, **kwargs)
            self._on_update = on_update

        def update(self, n=1):
            result = super().update(n)
            self._report()
            return result

        def set_description(self, desc=None, refresh=True):
            result = super().set_description(desc, refresh=refresh)
            self._report()
            return result

        def _report(self) -> None:
            desc = getattr(self, "desc", "") or "Downloading"
            total = float(self.total or 0)
            if total > 0:
                frac = float(self.n) / total
                self._on_update(0.08 + frac * 0.87, desc)
            else:
                self._on_update(0.15, desc)

    return HubProgressBar


def _run_download(job: ModelDownloadJob, target: Path, repo_id: str) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    job.update_progress(0.05, f"Preparing {repo_id}")

    if is_downloaded_model_dir(target):
        job.update_progress(1.0, "Model already present")
        with job._lock:
            job.status = "completed"
        return

    target.mkdir(parents=True, exist_ok=True)
    tqdm_class = _make_progress_tqdm(job.update_progress)

    disable_progress_bars()
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            tqdm_class=tqdm_class,
        )
    finally:
        enable_progress_bars()

    if not is_downloaded_model_dir(target):
        raise RuntimeError("Download finished but model files were not found.")

    job.update_progress(1.0, "Download complete")
    with job._lock:
        job.status = "completed"


def _start_download_job(
    *,
    models_root_dir: str,
    repo_id: str,
    folder_name: str,
    preset_id: str | None = None,
) -> ModelDownloadJob:
    root = _resolve_path(models_root_dir)
    if not root.is_dir():
        raise ValueError("Select a models directory before downloading.")

    target = root / folder_name
    if is_downloaded_model_dir(target):
        raise ValueError(f"Model is already downloaded at {target}")

    job_id = uuid.uuid4().hex[:12]
    job = ModelDownloadJob(
        job_id=job_id,
        preset_id=preset_id,
        repo_id=repo_id,
        target_dir=str(target),
        folder_name=folder_name,
    )

    if not _download_lock.acquire(blocking=False):
        with job._lock:
            job.status = "failed"
            job.error = "Another model download is already running."
        with _jobs_lock:
            _jobs[job_id] = job
        return job

    def _run() -> None:
        try:
            with job._lock:
                job.status = "running"
            _run_download(job, target, repo_id)
        except Exception as exc:
            detail = str(exc).strip() or repr(exc)
            with job._lock:
                job.status = "failed"
                job.error = detail
            print(f"Model download {job_id} failed:\n{traceback.format_exc()}", flush=True)
        finally:
            _download_lock.release()

    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run, daemon=True, name=f"model-download-{job_id}")
    thread.start()
    return job


def create_download_job(*, models_root_dir: str, preset_id: str) -> ModelDownloadJob:
    preset = get_preset(preset_id)
    return _start_download_job(
        models_root_dir=models_root_dir,
        repo_id=preset["repo_id"],
        folder_name=preset["folder_name"],
        preset_id=preset_id,
    )


def create_repo_download_job(
    *,
    models_root_dir: str,
    repo_id: str,
    quantization_id: str | None = None,
) -> ModelDownloadJob:
    repo_id = repo_id.strip()
    if not repo_id or "/" not in repo_id:
        raise ValueError("A valid HuggingFace repo_id is required (e.g. Qwen/Qwen2.5-Coder-7B-Instruct).")
    job = _start_download_job(
        models_root_dir=models_root_dir,
        repo_id=repo_id,
        folder_name=folder_name_from_repo(repo_id),
        preset_id=None,
    )
    if quantization_id:
        with job._lock:
            job.quantization_id = quantization_id.strip()
    return job


def get_download_job(job_id: str) -> ModelDownloadJob | None:
    with _jobs_lock:
        return _jobs.get(job_id)
