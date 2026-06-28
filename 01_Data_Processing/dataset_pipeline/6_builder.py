"""
6_builder.py — materialize train / valid / test parquets from temporal split output.

Pipeline step 6. Reads:  data/interim/split/split_*.parquet
Writes: data/processed/whole/train.parquet, valid.parquet, test.parquet

The temporal splitter (step 5) already assigns each row a `split` label using dynamic
year groups (~70% train / 20% valid / 10% test on commit_date). This step groups rows
by that column and writes three whole-table parquets for validator / batcher / modeling.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_spec = importlib.util.spec_from_file_location(
    "dataset_pipeline._runtime",
    Path(__file__).resolve().parent / "_runtime.py",
)
_runtime = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_runtime)
_runtime.ensure_app_root(__file__)

import argparse

import pandas as pd

from dataset_pipeline._loader import cfg as pcfg

VALID_SPLITS = ("train", "valid", "test")

_SPLIT_ALIASES = {
    "dev": "valid",
    "validation": "valid",
    "val": "valid",
    "eval": "valid",
    "development": "valid",
    "testset": "test",
    "testing": "test",
}


@dataclass
class BuildStats:
    chunks_read: int = 0
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped: int = 0
    by_split: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)
    date_ranges: dict[str, dict[str, str] | None] = field(default_factory=dict)
    target_ratios: dict[str, float] = field(default_factory=dict)

    def bump_source(self, source: str, split: str, n: int = 1) -> None:
        self.by_source.setdefault(source, {}).setdefault(split, 0)
        self.by_source[source][split] += n


def _normalize_split(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if s in VALID_SPLITS:
        return s
    return _SPLIT_ALIASES.get(s)


def _split_ratios(cfg: dict[str, Any]) -> dict[str, float]:
    scfg = pcfg.splits_cfg(cfg)
    return {
        "train": float(scfg.get("temporal_train_ratio", 0.7)),
        "valid": float(scfg.get("temporal_valid_ratio", 0.2)),
        "test": float(scfg.get("temporal_test_ratio", 0.1)),
    }


def _date_range(df: pd.DataFrame, date_field: str) -> dict[str, str] | None:
    if date_field not in df.columns or df.empty:
        return None
    dates = pd.to_datetime(df[date_field], errors="coerce", utc=True).dropna()
    if dates.empty:
        return None
    return {
        "min": str(dates.min().date()),
        "max": str(dates.max().date()),
    }


def build_splits(
    cfg: dict[str, Any],
    stats: BuildStats,
) -> dict[str, pd.DataFrame]:
    """
    Read all split_* chunks and partition rows by the `split` column from step 5.
    """
    bcfg = pcfg.builder_cfg(cfg)
    split_field = str(bcfg.get("split_field", "split"))
    date_field = str(pcfg.splits_cfg(cfg).get("date_field", "commit_date"))

    in_dir = pcfg.split_output_dir(cfg)
    in_paths = pcfg.split_chunk_paths(cfg)
    if not in_paths:
        in_paths = pcfg.list_chunk_files(in_dir, pcfg.split_chunk_prefix(cfg))
    if not in_paths:
        raise FileNotFoundError(
            f"No split parquets under {in_dir.as_posix()} — run 5_temporal_splitter.py first."
        )

    parts: dict[str, list[pd.DataFrame]] = {s: [] for s in VALID_SPLITS}

    for path in in_paths:
        stats.chunks_read += 1
        df = pd.read_parquet(path)
        stats.rows_in += len(df)

        if split_field not in df.columns:
            raise ValueError(
                f"{path.name}: missing '{split_field}' column — run temporal splitter first."
            )

        normalized: list[str | None] = []
        for val in df[split_field]:
            normalized.append(_normalize_split(val))

        keep_mask: list[bool] = []
        for split in normalized:
            if split in VALID_SPLITS:
                keep_mask.append(True)
            else:
                keep_mask.append(False)
                stats.rows_dropped += 1

        df = df[pd.Series(keep_mask, index=df.index)].copy()
        df[split_field] = [s for s in normalized if s in VALID_SPLITS]

        if "source_dataset" in df.columns:
            for source, group in df.groupby("source_dataset", dropna=False):
                for split_name in VALID_SPLITS:
                    n = int((group[split_field] == split_name).sum())
                    if n:
                        stats.bump_source(str(source), split_name, n)

        for split_name in VALID_SPLITS:
            sub = df[df[split_field] == split_name]
            if len(sub):
                parts[split_name].append(sub)

    out: dict[str, pd.DataFrame] = {}
    for split_name in VALID_SPLITS:
        if parts[split_name]:
            out[split_name] = pd.concat(parts[split_name], ignore_index=True)
        else:
            out[split_name] = pd.DataFrame()
        stats.by_split[split_name] = len(out[split_name])
        stats.rows_out += len(out[split_name])

    if stats.rows_out == 0:
        raise ValueError("No rows with valid train/valid/test labels — check temporal splitter output.")

    stats.date_ranges = {
        split_name: _date_range(out[split_name], date_field)
        for split_name in VALID_SPLITS
    }
    stats.target_ratios = _split_ratios(cfg)

    return out


def write_processed_parquets(
    cfg: dict[str, Any],
    splits: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    bcfg = pcfg.builder_cfg(cfg)
    out_fmt = str(bcfg.get("output_format", "parquet")).lower()
    written: dict[str, Path] = {}

    out_dir = pcfg.processed_whole_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in VALID_SPLITS:
        df = splits[split_name]
        path = pcfg.builder_output_path(cfg, split_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        if out_fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported builder output_format: {out_fmt}")

        written[split_name] = path

    return written


def run_build(cfg: dict[str, Any], stage: str) -> Path:
    stats = BuildStats()
    splits = build_splits(cfg, stats)
    paths = write_processed_parquets(cfg, splits)

    report = cfg["_base_dir"] / pcfg.builder_cfg(cfg).get(
        "report", "reports/builder_report.md"
    )
    _write_report(stats, report, stage, paths)
    return pcfg.processed_whole_dir(cfg)


def _write_report(
    stats: BuildStats,
    path: Path,
    stage: str,
    paths: dict[str, Path],
) -> None:
    total = stats.rows_out or 1
    target = stats.target_ratios or {"train": 0.7, "valid": 0.2, "test": 0.1}
    date_ranges = stats.date_ranges

    lines = [
        f"# Builder report (stage {stage})",
        "",
        "Reads `split` labels from step 5 (dynamic temporal ~70/20/10 by commit year).",
        "Writes whole-table `train.parquet`, `valid.parquet`, `test.parquet`.",
        "",
        f"- chunks read: {stats.chunks_read}",
        f"- rows in: {stats.rows_in}",
        f"- rows out: {stats.rows_out}",
        f"- rows dropped (invalid/missing split): {stats.rows_dropped}",
        "",
        "## Output files",
        "",
    ]
    for split_name in VALID_SPLITS:
        p = paths.get(split_name)
        n = stats.by_split.get(split_name, 0)
        ratio = n / total
        tgt = target.get(split_name, 0)
        lines.append(f"### {split_name}")
        lines.append(f"- path: `{p.as_posix() if p else 'n/a'}`")
        lines.append(f"- rows: {n}")
        lines.append(f"- actual ratio: {ratio:.1%}")
        lines.append(f"- target ratio (from temporal splitter): {tgt:.1%}")
        dr = date_ranges.get(split_name)
        if dr:
            lines.append(f"- commit_date range: {dr['min']} → {dr['max']}")
        lines.append("")

    if stats.by_source:
        lines.append("## Rows per source_dataset")
        lines.append("")
        for source in sorted(stats.by_source):
            lines.append(f"### {source}")
            for k, v in sorted(stats.by_source[source].items()):
                lines.append(f"- {k}: {v}")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/valid/test parquets from temporal split output"
    )
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default="1a")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = pcfg.load_config(args.config)
    out = run_build(cfg, args.stage)
    print(f"Wrote whole-table parquets under: {out}")
    for split_name in VALID_SPLITS:
        p = pcfg.builder_output_path(cfg, split_name)
        if p.exists():
            print(f"  {split_name}: {p}")


if __name__ == "__main__":
    main()
