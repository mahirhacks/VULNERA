"""
2_extractor.py — keep C/C++ functions from normalized combined Parquet chunks.

Pipeline step 2. Reads:  data/interim/normalized/normalized_{N}.parquet
Writes: data/interim/extracted/extracted_{N}.parquet
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
from typing import Any, Iterator

import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore[assignment]

from dataset_pipeline._loader import cfg as pcfg

_REJECT_LINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*def\s+\w+", re.MULTILINE),
    re.compile(r"^\s*async\s+def\s+", re.MULTILINE),
    re.compile(r"^\s*import\s+(java|javax|org\.|com\.)", re.MULTILINE),
    re.compile(r"^\s*package\s+(java|javax|com\.|org\.)", re.MULTILINE),
    re.compile(r"^\s*public\s+(class|interface|enum)\s+", re.MULTILINE),
    re.compile(r"^\s*<?php\b", re.MULTILINE),
    re.compile(r"^\s*fn\s+\w+\s*\(", re.MULTILINE),
    re.compile(r"^\s*func\s+\w+\s*\([^)]*\)\s*\{", re.MULTILINE),
]

_POSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\{"),
    re.compile(r"\}"),
    re.compile(r";"),
    re.compile(r"\b(int|void|char|float|double|long|short|unsigned|bool|size_t)\b"),
    re.compile(r"#\s*include\b"),
    re.compile(r"->"),
    re.compile(r"::"),
    re.compile(r"\breturn\b"),
    re.compile(r"\bif\s*\("),
]

_C_CPP_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp",
}


@dataclass
class ExtractStats:
    chunks: int = 0
    rows_in: int = 0
    rows_out: int = 0
    skipped: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.by_reason.setdefault(key, 0)
        self.by_reason[key] += n


def looks_like_c_cpp(code: str) -> bool:
    if not code or not code.strip():
        return False
    sample = "\n".join(code.strip().splitlines()[:30])
    for pat in _REJECT_LINE_PATTERNS:
        if pat.search(sample):
            return False
    return sum(1 for pat in _POSITIVE_PATTERNS if pat.search(code)) >= 2


def file_path_is_c_cpp(file_name: str | None, nulls: set[str]) -> bool:
    if not file_name or file_name in nulls:
        return True
    return Path(file_name).suffix in _C_CPP_EXTENSIONS


def passes_cpp_filter(row: dict[str, Any], nulls: set[str], stats: ExtractStats) -> bool:
    code = row.get("code") or ""
    if not isinstance(code, str) or not code.strip():
        stats.bump("skip_empty")
        return False
    fp = row.get("file_path")
    if isinstance(fp, str) and not file_path_is_c_cpp(fp, nulls):
        stats.bump("skip_extension")
        return False
    if not looks_like_c_cpp(code):
        stats.bump("skip_not_cpp")
        return False
    return True


def iter_parquet_rows(path: Path, batch_size: int = 50_000) -> Iterator[dict[str, Any]]:
    if pq is not None:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            for row in df.to_dict(orient="records"):
                clean: dict[str, Any] = {}
                for k, v in row.items():
                    clean[k] = None if (isinstance(v, float) and pd.isna(v)) else v
                yield clean
        return
    df = pd.read_parquet(path)
    for row in df.to_dict(orient="records"):
        yield row


def extract_chunk(
    path: Path, cfg: dict[str, Any], stats: ExtractStats
) -> pd.DataFrame:
    nulls = pcfg.null_placeholders(cfg)
    cols = pcfg.core_columns(cfg)
    kept: list[dict[str, Any]] = []
    batch_size = int(pcfg.extractor_cfg(cfg).get("parquet_batch_size", 50_000))

    for row in iter_parquet_rows(path, batch_size=batch_size):
        stats.rows_in += 1
        if not passes_cpp_filter(row, nulls, stats):
            stats.skipped += 1
            continue
        kept.append({c: row.get(c) for c in cols})
        stats.rows_out += 1

    return pd.DataFrame(kept, columns=cols) if kept else pd.DataFrame(columns=cols)


def run_extract(cfg: dict[str, Any], stage: str) -> Path:
    in_dir = pcfg.normalized_output_dir(cfg)
    out_dir = pcfg.extracted_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = ExtractStats()

    in_paths = pcfg.normalized_chunk_paths(cfg)
    if not in_paths:
        in_paths = pcfg.list_chunk_files(in_dir, pcfg.normalized_chunk_prefix(cfg))
    if not in_paths:
        print("No normalized chunks found; run normalizer first.")
        return out_dir

    out_prefix = pcfg.extracted_chunk_prefix(cfg)
    for in_path in in_paths:
        m = re.search(r"_(\d+)\.parquet$", in_path.name)
        idx = int(m.group(1)) if m else 1
        df = extract_chunk(in_path, cfg, stats)
        out_path = out_dir / f"{out_prefix}_{idx}.parquet"
        df.to_parquet(out_path, index=False)
        stats.chunks += 1

    report = cfg["_base_dir"] / pcfg.extractor_cfg(cfg).get("report", "reports/extractor_report.md")
    _write_report(stats, report, stage, in_dir, out_dir)
    return out_dir


def _write_report(
    stats: ExtractStats, path: Path, stage: str, in_dir: Path, out_dir: Path
) -> None:
    lines = [
        f"# Extractor report (stage {stage})",
        "",
        f"- chunks written: {stats.chunks}",
        f"- rows read: {stats.rows_in}",
        f"- rows kept (C/C++): {stats.rows_out}",
        f"- rows skipped: {stats.skipped}",
        f"- input: `{in_dir.as_posix()}/normalized_*.parquet`",
        f"- output: `{out_dir.as_posix()}/extracted_*.parquet`",
        "",
        "## Skip reasons",
        "",
    ]
    for k, v in sorted(stats.by_reason.items()):
        lines.append(f"- {k}: {v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract C/C++ rows from normalized Parquet")
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default="1a")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = pcfg.load_config(args.config)
    out = run_extract(cfg, args.stage)
    print(f"Wrote: {list(out.glob('extracted_*.parquet'))}")


if __name__ == "__main__":
    main()
