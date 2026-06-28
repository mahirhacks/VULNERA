"""
3_cleaner.py — clean C/C++ function code in extracted parquets.

Pipeline step 3. Reads:  data/interim/extracted/{dataset}.parquet
Writes: data/interim/cleaned/{dataset}.parquet
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "dataset_pipeline._runtime",
    Path(__file__).resolve().parent / "_runtime.py",
)
_runtime = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_runtime)
_runtime.ensure_app_root(__file__)

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_pipeline._loader import cfg as pcfg

_LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


@dataclass
class CleanStats:
    datasets: int = 0
    rows_in: int = 0
    rows_out: int = 0
    dropped: int = 0
    by_dataset: dict[str, dict[str, int]] = field(default_factory=dict)

    def drop(self, dataset: str, reason: str) -> None:
        self.dropped += 1
        self.by_dataset.setdefault(dataset, {}).setdefault(f"drop_{reason}", 0)
        self.by_dataset[dataset][f"drop_{reason}"] += 1


def _cleaning_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("3_cleaner") or cfg.get("cleaning") or {}


@dataclass
class CleanCodeResult:
    code: str
    line_map: list[int]


def _strip_line_comments(line: str, *, in_block: bool) -> tuple[str, bool]:
    """Remove C/C++ comments from one source line; return (text, still_in_block)."""
    out: list[str] = []
    index = 0
    while index < len(line):
        if in_block:
            end = line.find("*/", index)
            if end < 0:
                return "".join(out), True
            index = end + 2
            in_block = False
            continue

        block_start = line.find("/*", index)
        line_start = line.find("//", index)
        if block_start < 0 and line_start < 0:
            out.append(line[index:])
            break
        if line_start >= 0 and (block_start < 0 or line_start < block_start):
            out.append(line[index:line_start])
            break
        out.append(line[index:block_start])
        end = line.find("*/", block_start + 2)
        if end < 0:
            return "".join(out), True
        index = end + 2

    return "".join(out), in_block


def clean_code_with_tracker(
    code: str,
    cfg: dict[str, Any],
    *,
    base_line: int = 1,
) -> CleanCodeResult:
    """Clean one function body while preserving file line numbers for each kept line."""
    c = _cleaning_cfg(cfg)
    keep_comments = bool(c.get("keep_comments", False))
    normalize_ws = bool(c.get("normalize_whitespace", True))

    kept_lines: list[str] = []
    line_map: list[int] = []
    in_block = False

    for offset, raw in enumerate(code.splitlines()):
        file_line = int(base_line) + offset
        line = raw
        if not keep_comments:
            line, in_block = _strip_line_comments(line, in_block=in_block)
        if normalize_ws:
            line = line.rstrip()
        if not line.strip():
            continue
        kept_lines.append(line)
        line_map.append(file_line)

    cleaned = "\n".join(kept_lines)
    if normalize_ws:
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return CleanCodeResult(code=cleaned, line_map=line_map)


def strip_comments(code: str) -> str:
    code = _BLOCK_COMMENT.sub("", code)
    code = _LINE_COMMENT.sub("", code)
    return code


def normalize_whitespace(code: str) -> str:
    lines = [ln.rstrip() for ln in code.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def rough_token_count(code: str) -> int:
    return len(re.findall(r"\S+", code))


def clean_code(code: str, cfg: dict[str, Any]) -> str:
    c = _cleaning_cfg(cfg)
    if not c.get("keep_comments", False):
        code = strip_comments(code)
    if c.get("normalize_whitespace", True):
        code = normalize_whitespace(code)
    return code


def should_drop_row(code: str, cfg: dict[str, Any], dataset: str, stats: CleanStats) -> bool:
    c = _cleaning_cfg(cfg)
    if c.get("drop_empty_code", True) and not code.strip():
        stats.drop(dataset, "empty")
        return True
    lines = [ln for ln in code.splitlines() if ln.strip()]
    if len(lines) < int(c.get("min_lines", 3)):
        stats.drop(dataset, "min_lines")
        return True
    if rough_token_count(code) < int(c.get("min_tokens", 50)):
        stats.drop(dataset, "min_tokens")
        return True
    return False


def clean_dataframe(df: pd.DataFrame, dataset: str, cfg: dict[str, Any], stats: CleanStats) -> pd.DataFrame:
    if "code" not in df.columns:
        raise ValueError(f"{dataset}: missing code column")

    stats.rows_in += len(df)
    cleaned_codes: list[str] = []
    keep_mask: list[bool] = []

    for code in df["code"].astype(str):
        new_code = clean_code(code, cfg)
        if should_drop_row(new_code, cfg, dataset, stats):
            keep_mask.append(False)
            cleaned_codes.append(new_code)
        else:
            keep_mask.append(True)
            cleaned_codes.append(new_code)

    out = df.copy()
    out["code"] = cleaned_codes
    out = out[pd.Series(keep_mask, index=out.index)].reset_index(drop=True)
    stats.rows_out += len(out)
    stats.datasets += 1
    stats.by_dataset.setdefault(dataset, {})["kept"] = len(out)
    return out


def run_clean(cfg: dict[str, Any], stage: str) -> Path:
    in_dir = pcfg.extracted_output_dir(cfg)
    out_dir = pcfg.cleaned_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = CleanStats()

    in_paths = pcfg.extracted_chunk_paths(cfg)
    if not in_paths:
        in_paths = pcfg.list_chunk_files(in_dir, pcfg.extracted_chunk_prefix(cfg))
    out_prefix = pcfg.cleaned_chunk_prefix(cfg)

    for src in in_paths:
        m = re.search(r"_(\d+)\.parquet$", src.name)
        idx = int(m.group(1)) if m else 1
        df = pd.read_parquet(src)
        cleaned = clean_dataframe(df, f"chunk_{idx}", cfg, stats)
        cleaned.to_parquet(out_dir / f"{out_prefix}_{idx}.parquet", index=False)

    report = cfg["_base_dir"] / _cleaning_cfg(cfg).get(
        "report", "reports/cleaner_report.md"
    )
    _write_report(stats, report, stage, in_dir, out_dir)
    return out_dir


def _write_report(
    stats: CleanStats,
    path: Path,
    stage: str,
    in_dir: Path,
    out_dir: Path,
) -> None:
    lines = [
        f"# Cleaner report (stage {stage})",
        "",
        f"- datasets cleaned: {stats.datasets}",
        f"- rows in: {stats.rows_in}",
        f"- rows out: {stats.rows_out}",
        f"- rows dropped: {stats.dropped}",
        f"- input: `{in_dir.as_posix()}`",
        f"- output: `{out_dir.as_posix()}`",
        "",
    ]
    for ds, counts in sorted(stats.by_dataset.items()):
        lines.append(f"### {ds}")
        for k, v in sorted(counts.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean extracted dataset parquets")
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default="1a")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = pcfg.load_config(args.config)
    out = run_clean(cfg, args.stage)
    print(f"Wrote cleaned parquets under: {out}")


if __name__ == "__main__":
    main()
