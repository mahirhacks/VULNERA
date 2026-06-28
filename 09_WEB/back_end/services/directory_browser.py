from __future__ import annotations

import string
from pathlib import Path
from typing import Any

from services.paths import PROJECT_ROOT

DEFAULT_MODELS_ROOT = PROJECT_ROOT / "08_LLM" / "models"


def _storage_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def _resolve_browse_path(path_value: str | None) -> Path:
    if not path_value or not str(path_value).strip():
        return PROJECT_ROOT.resolve()
    raw = str(path_value).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _list_subdirectories(directory: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        children = sorted(directory.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return entries

    for child in children:
        if not child.is_dir() or child.name.startswith("."):
            continue
        has_children = False
        try:
            has_children = any(
                grandchild.is_dir() and not grandchild.name.startswith(".")
                for grandchild in child.iterdir()
            )
        except OSError:
            has_children = False
        entries.append(
            {
                "name": child.name,
                "path": _storage_path(child),
                "has_children": has_children,
            }
        )
    return entries


def _browse_roots() -> list[dict[str, str]]:
    roots: list[dict[str, str]] = [
        {
            "label": "VULNERA project",
            "path": _storage_path(PROJECT_ROOT),
        },
    ]
    if DEFAULT_MODELS_ROOT.is_dir():
        roots.append(
            {
                "label": "Default models folder",
                "path": _storage_path(DEFAULT_MODELS_ROOT),
            }
        )
    home = Path.home()
    if home.is_dir():
        roots.append({"label": "Home", "path": _storage_path(home)})

    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:/")
        if drive.exists():
            roots.append({"label": f"{letter}:\\", "path": _storage_path(drive)})

    return roots


def browse_directories(path_value: str | None = None) -> dict[str, Any]:
    if path_value is None or not str(path_value).strip():
        return {
            "mode": "roots",
            "roots": _browse_roots(),
        }

    directory = _resolve_browse_path(path_value)
    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    parent = directory.parent
    parent_path = None
    if parent != directory:
        parent_path = _storage_path(parent)

    return {
        "mode": "directory",
        "name": directory.name or str(directory),
        "current_path": str(directory).replace("\\", "/"),
        "storage_path": _storage_path(directory),
        "parent_path": parent_path,
        "entries": _list_subdirectories(directory),
    }
