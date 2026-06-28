from __future__ import annotations

import json
from typing import Any

from services.paths import SCANS_DIR


def ensure_scans_dir() -> None:
    SCANS_DIR.mkdir(parents=True, exist_ok=True)


def scan_path(scan_id: str) -> Any:
    return SCANS_DIR / f"{scan_id}.json"


def load_scan_history() -> list[dict[str, Any]]:
    ensure_scans_dir()
    scans: list[dict[str, Any]] = []
    for path in sorted(SCANS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            scans.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return scans


def load_scan(scan_id: str) -> dict[str, Any] | None:
    path = scan_path(scan_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_scan(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_scans_dir()
    scan_path(str(payload["scan_id"])).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def delete_scan(scan_id: str) -> bool:
    path = scan_path(scan_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def scan_summary(scan: dict[str, Any]) -> dict[str, Any]:
    markers = scan.get("file_markers") or []
    function_alerts = [m for m in markers if m.get("marker_type") == "function_alert"]
    regions = [m for m in markers if m.get("marker_type") != "function_alert"]
    return {
        "scan_id": scan.get("scan_id"),
        "project_id": scan.get("project_id"),
        "filename": scan.get("filename"),
        "uploaded_at": scan.get("uploaded_at"),
        "function_count": scan.get("function_count", len(scan.get("functions") or [])),
        "finding_count": len(regions) + len(function_alerts),
    }
