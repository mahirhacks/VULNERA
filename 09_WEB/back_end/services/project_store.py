from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from services.paths import PROJECTS_DIR
from services.scan_store import delete_scan, load_scan_history, scan_summary


def ensure_projects_dir() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def project_path(project_id: str):
    return PROJECTS_DIR / f"{project_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_projects() -> list[dict[str, Any]]:
    ensure_projects_dir()
    projects: list[dict[str, Any]] = []
    for path in sorted(PROJECTS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            projects.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return projects


def load_project(project_id: str) -> dict[str, Any] | None:
    path = project_path(project_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_project(project: dict[str, Any]) -> dict[str, Any]:
    ensure_projects_dir()
    project_path(str(project["project_id"])).write_text(
        json.dumps(project, ensure_ascii=False),
        encoding="utf-8",
    )
    return project


def create_project(name: str) -> dict[str, Any]:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Project name is required.")
    now = _now_iso()
    project = {
        "project_id": uuid.uuid4().hex[:12],
        "name": cleaned,
        "created_at": now,
        "updated_at": now,
        "scan_ids": [],
    }
    return save_project(project)


def add_scan_to_project(project_id: str, scan_id: str) -> dict[str, Any] | None:
    project = load_project(project_id)
    if project is None:
        return None
    scan_ids = list(project.get("scan_ids") or [])
    if scan_id not in scan_ids:
        scan_ids.append(scan_id)
    project["scan_ids"] = scan_ids
    project["updated_at"] = _now_iso()
    return save_project(project)


def project_scans(project: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = {str(scan_id) for scan_id in project.get("scan_ids") or []}
    if not wanted:
        return []
    scans = []
    for scan in load_scan_history():
        scan_id = str(scan.get("scan_id", ""))
        if scan_id in wanted or str(scan.get("project_id", "")) == str(project.get("project_id")):
            scans.append(scan_summary(scan))
    scans.sort(key=lambda item: item.get("uploaded_at", ""), reverse=True)
    return scans


def project_detail(project_id: str) -> dict[str, Any] | None:
    project = load_project(project_id)
    if project is None:
        return None
    return {
        "project_id": project["project_id"],
        "name": project["name"],
        "created_at": project.get("created_at"),
        "updated_at": project.get("updated_at"),
        "scan_count": len(project.get("scan_ids") or []),
        "scans": project_scans(project),
    }


def project_summaries() -> list[dict[str, Any]]:
    rows = []
    for project in load_projects():
        scans = project_scans(project)
        rows.append(
            {
                "project_id": project["project_id"],
                "name": project["name"],
                "created_at": project.get("created_at"),
                "updated_at": project.get("updated_at"),
                "scan_count": len(scans),
                "scans": scans,
            }
        )
    return rows


def delete_project(project_id: str, *, delete_scans: bool = True) -> bool:
    project = load_project(project_id)
    if project is None:
        return False
    if delete_scans:
        for scan_id in project.get("scan_ids") or []:
            delete_scan(str(scan_id))
    project_path(project_id).unlink()
    return True


def standalone_scans() -> list[dict[str, Any]]:
    return [
        scan_summary(scan)
        for scan in load_scan_history()
        if not scan.get("project_id")
    ]
