from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

BACKEND_ROOT = Path(__file__).resolve().parent
WEB_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(WEB_ROOT.parent / "07_XAI" / "training_scripts"))

from pipeline.scan_pipeline import PHASES  # noqa: E402
from services.config_service import load_config, save_config  # noqa: E402
from services.directory_browser import browse_directories  # noqa: E402
from services.llm_config_service import (  # noqa: E402
    delete_llm_model,
    llm_settings_payload,
    scan_models_directory,
    update_llm_model,
    update_quick_presets,
)
from services.download_job_store import (  # noqa: E402
    create_download_job,
    create_repo_download_job,
    get_download_job,
)
from services.hardware_service import get_hardware_profile  # noqa: E402
from services.hf_hub_service import (  # noqa: E402
    check_huggingface_connectivity,
    enrich_model_with_variants,
    extract_param_b,
    folder_name_from_repo,
    hf_filter_options,
    model_loader_info,
    runtime_quant_options,
    search_hf_models,
)
from services.job_store import create_scan_job, get_job  # noqa: E402
from services.project_store import (  # noqa: E402
    create_project,
    delete_project,
    project_detail,
    project_summaries,
    standalone_scans,
)
from services.scan_store import (  # noqa: E402
    delete_scan,
    load_scan,
    load_scan_history,
    scan_summary,
)
from services.report_pdf import build_scan_report_pdf, report_download_filename  # noqa: E402

app = FastAPI(title="VULNERA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanConfigUpdate(BaseModel):
    llm_provider: str | None = None
    max_functions: int | None = Field(default=None, ge=1, le=500)
    use_cpu_embedder: bool | None = None
    llm_max_new_tokens: int | None = Field(default=None, ge=32, le=2048)
    llm_max_code_chars: int | None = Field(default=None, ge=256, le=16000)
    llm_explanation_enabled: bool | None = None
    llm_chain_of_thought: bool | None = None


class ConfigUpdate(BaseModel):
    scan: ScanConfigUpdate


class LlmModelUpdate(BaseModel):
    models_root_dir: str = Field(min_length=1, max_length=512)
    selected_model_id: str = Field(min_length=1, max_length=256)
    quantization_id: str | None = Field(default=None, max_length=32)


class QuickPresetEntry(BaseModel):
    repo_id: str = Field(min_length=3, max_length=256)
    label: str | None = Field(default=None, max_length=256)
    folder_name: str | None = Field(default=None, max_length=256)
    note: str | None = Field(default=None, max_length=512)
    id: str | None = Field(default=None, max_length=128)


class QuickPresetsUpdate(BaseModel):
    quick_presets: list[QuickPresetEntry] = Field(default_factory=list, max_length=3)


class LlmModelDelete(BaseModel):
    models_root_dir: str = Field(min_length=1, max_length=512)
    model_id: str = Field(min_length=1, max_length=256)


class ModelDownloadRequest(BaseModel):
    models_root_dir: str = Field(min_length=1, max_length=512)
    preset_id: str | None = Field(default=None, max_length=64)
    repo_id: str | None = Field(default=None, max_length=256)
    quantization_id: str | None = Field(default=None, max_length=32)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "phases": [label for _, label in PHASES]}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return load_config()


@app.put("/api/config")
def put_config(body: ConfigUpdate) -> dict[str, Any]:
    config = load_config()
    scan_cfg = dict(config.get("scan") or {})
    if body.scan.llm_provider is not None:
        scan_cfg["llm_provider"] = body.scan.llm_provider
    if body.scan.max_functions is not None:
        scan_cfg["max_functions"] = body.scan.max_functions
    if body.scan.use_cpu_embedder is not None:
        scan_cfg["use_cpu_embedder"] = body.scan.use_cpu_embedder
    if body.scan.llm_max_new_tokens is not None:
        scan_cfg["llm_max_new_tokens"] = body.scan.llm_max_new_tokens
    if body.scan.llm_max_code_chars is not None:
        scan_cfg["llm_max_code_chars"] = body.scan.llm_max_code_chars
    if body.scan.llm_explanation_enabled is not None:
        scan_cfg["llm_explanation_enabled"] = body.scan.llm_explanation_enabled
    if body.scan.llm_chain_of_thought is not None:
        scan_cfg["llm_chain_of_thought"] = body.scan.llm_chain_of_thought
    config["scan"] = scan_cfg
    return save_config(config)


@app.get("/api/llm-config")
def get_llm_config() -> dict[str, Any]:
    return llm_settings_payload()


def _preset_status_for_api(models_root_dir: str) -> list[dict[str, Any]]:
    from services.llm_config_service import _preset_status, load_llm_config

    return _preset_status(models_root_dir, load_llm_config())


@app.get("/api/llm-config/browse")
def browse_llm_directories(path: str | None = None) -> dict[str, Any]:
    try:
        return browse_directories(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/llm-config/scan")
def scan_llm_models(models_root_dir: str) -> dict[str, Any]:
    root = models_root_dir.strip()
    if not root:
        raise HTTPException(status_code=400, detail="models_root_dir is required")
    return {
        "models_root_dir": root,
        "available_models": scan_models_directory(root),
        "presets": _preset_status_for_api(root),
    }


@app.put("/api/llm-config")
def put_llm_config(body: LlmModelUpdate) -> dict[str, Any]:
    try:
        return update_llm_model(
            models_root_dir=body.models_root_dir,
            selected_model_id=body.selected_model_id,
            quantization_id=body.quantization_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/llm-config/quick-presets")
def put_quick_presets(body: QuickPresetsUpdate) -> dict[str, Any]:
    try:
        return update_quick_presets([item.model_dump() for item in body.quick_presets])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/llm-config/models")
def remove_llm_model(body: LlmModelDelete) -> dict[str, Any]:
    try:
        return delete_llm_model(
            models_root_dir=body.models_root_dir,
            model_id=body.model_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete model files: {exc}") from exc


@app.get("/api/llm-config/connectivity")
def llm_connectivity() -> dict[str, Any]:
    online = check_huggingface_connectivity()
    return {"online": online, "huggingface": online}


@app.get("/api/llm-config/hf-filters")
def hf_filters() -> dict[str, Any]:
    return hf_filter_options()


@app.get("/api/llm-config/hardware")
def llm_hardware() -> dict[str, Any]:
    return get_hardware_profile()


@app.get("/api/llm-config/hf-quants")
def llm_runtime_quants() -> dict[str, Any]:
    return {"runtime_quants": runtime_quant_options()}


@app.get("/api/llm-config/hf-models/variants")
def hf_model_variants(repo_id: str) -> dict[str, Any]:
    repo_id = repo_id.strip()
    if not repo_id or "/" not in repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
    hardware = get_hardware_profile()
    base = {
        "repo_id": repo_id,
        "label": repo_id.split("/")[-1],
        "folder_name": folder_name_from_repo(repo_id),
        "param_b": extract_param_b(repo_id),
        "param_label": (
            f"{extract_param_b(repo_id):g}B" if extract_param_b(repo_id) is not None else "—"
        ),
        "loader": model_loader_info(repo_id),
    }
    return {
        "hardware": hardware,
        "model": enrich_model_with_variants(base, hardware),
        "runtime_quants": runtime_quant_options(),
    }


@app.get("/api/llm-config/hf-models")
def hf_models(
    query: str = "",
    family: str = "all",
    param_size: str = "any",
    purpose: str = "code",
    page: int = 0,
    include_fit: bool = False,
) -> dict[str, Any]:
    try:
        return search_hf_models(
            query=query,
            family=family,
            param_size=param_size,
            purpose=purpose,
            page=max(0, page),
            limit=20,
            include_fit=include_fit,
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Hugging Face search failed: {exc}") from exc


@app.post("/api/llm-config/downloads")
def start_model_download(body: ModelDownloadRequest) -> dict[str, str]:
    try:
        if body.repo_id:
            job = create_repo_download_job(
                models_root_dir=body.models_root_dir,
                repo_id=body.repo_id,
                quantization_id=body.quantization_id,
            )
        elif body.preset_id:
            job = create_download_job(
                models_root_dir=body.models_root_dir,
                preset_id=body.preset_id,
            )
        else:
            raise ValueError("Provide either repo_id or preset_id.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    snapshot = job.snapshot()
    if snapshot["status"] == "failed":
        raise HTTPException(status_code=409, detail=snapshot["error"] or "Download could not start")
    return {"job_id": job.job_id}


@app.get("/api/llm-config/downloads/{job_id}")
def get_model_download_job(job_id: str) -> dict[str, Any]:
    job = get_download_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found")
    return job.snapshot()


@app.get("/api/llm-config/downloads/{job_id}/stream")
async def stream_model_download(job_id: str) -> StreamingResponse:
    job = get_download_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found")

    async def event_generator():
        last_payload = ""
        while True:
            snapshot = job.snapshot()
            payload = json.dumps(snapshot)
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if snapshot["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.35)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/scans")
def list_scans(standalone: bool = False) -> list[dict[str, Any]]:
    if standalone:
        return standalone_scans()
    return [scan_summary(scan) for scan in load_scan_history()]


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    return project_summaries()


@app.post("/api/projects")
def post_project(body: ProjectCreate) -> dict[str, Any]:
    try:
        project = create_project(body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return project_detail(project["project_id"]) or project


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    project = project_detail(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.delete("/api/projects/{project_id}")
def remove_project(project_id: str) -> dict[str, bool]:
    if not delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True}


@app.get("/api/scans/{scan_id}")
def get_scan(scan_id: str) -> dict[str, Any]:
    scan = load_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


@app.get("/api/scans/{scan_id}/report.pdf")
def export_scan_report(scan_id: str) -> Response:
    scan = load_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    try:
        pdf_bytes = build_scan_report_pdf(scan)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report export failed: {exc}") from exc
    filename = report_download_filename(scan)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/scans/{scan_id}")
def remove_scan(scan_id: str) -> dict[str, bool]:
    if not delete_scan(scan_id):
        raise HTTPException(status_code=404, detail="Scan not found")
    return {"deleted": True}


@app.post("/api/scans")
async def start_scan(
    file: UploadFile = File(...),
    llm_provider: str | None = Form(default=None),
    max_functions: int | None = Form(default=None),
    project_id: str | None = Form(default=None),
) -> dict[str, str]:
    config = load_config()
    scan_cfg = config.get("scan") or {}
    provider = llm_provider or str(scan_cfg.get("llm_provider", "mock"))
    max_fn = max_functions if max_functions is not None else scan_cfg.get("max_functions", 50)

    raw = await file.read()
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text") from exc

    filename = file.filename or "source.c"
    if project_id:
        from services.project_store import load_project

        if load_project(project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found")
    job = create_scan_job(
        source=source,
        filename=filename,
        llm_provider=provider,
        max_functions=int(max_fn) if max_fn is not None else None,
        project_id=project_id,
    )
    return {"job_id": job.job_id}


@app.get("/api/scans/jobs/{job_id}")
def get_scan_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.snapshot()


@app.get("/api/scans/jobs/{job_id}/stream")
async def stream_scan_job(job_id: str) -> StreamingResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_payload = ""
        while True:
            snapshot = job.snapshot()
            payload = json.dumps(snapshot)
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if snapshot["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
