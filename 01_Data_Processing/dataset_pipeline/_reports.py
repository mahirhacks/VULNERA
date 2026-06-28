"""Shared markdown report helpers for numbered dataset_pipeline steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def report_header(title: str, stage: str, *, script_stem: str) -> list[str]:
    return [
        f"# {title}",
        "",
        f"- stage: `{stage}`",
        f"- script: `{script_stem}.py`",
        "",
    ]


def step_report_path(cfg: dict[str, Any], script_stem: str) -> Path:
    block = cfg.get(script_stem, {})
    report = block.get("report", f"reports/{script_stem}_report.md")
    base = cfg.get("_base_dir", Path("."))
    return Path(base) / report


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
