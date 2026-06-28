"""
4_deduplicator.py — remove exact and near-duplicate functions.

Pipeline step 4. Reads:  data/interim/cleaned/{dataset}.parquet
Writes: data/interim/deduped/{dataset}.parquet

Uses SHA1 on code for exact duplicates and MinHash-LSH (Jaccard >= threshold)
for near-duplicates. A single global index is built across all active datasets
in stage order (PrimeVul first, then DiverseVul, Big-Vul, CVEfixes) so 1B/1C
get cross-dataset dedup. Rows are processed train → valid → test first when kept.
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
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import warnings

import pandas as pd

_DATASKETCH_IMPORT_ERROR: Exception | None = None
try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except Exception as exc:  # cupy/CUDA or missing package
    MinHash = None  # type: ignore[misc, assignment]
    MinHashLSH = None  # type: ignore[misc, assignment]
    _HAS_DATASKETCH = False
    _DATASKETCH_IMPORT_ERROR = exc

from dataset_pipeline._loader import cfg as pcfg

SPLIT_PRIORITY = {"train": 0, "valid": 1, "test": 2}


@dataclass
class DedupStats:
    datasets: int = 0
    rows_in: int = 0
    rows_out: int = 0
    dropped_exact: int = 0
    dropped_near: int = 0
    by_dataset: dict[str, dict[str, int]] = field(default_factory=dict)

    def bump(self, dataset: str, key: str, n: int = 1) -> None:
        self.by_dataset.setdefault(dataset, {}).setdefault(key, 0)
        self.by_dataset[dataset][key] += n


class DedupIndex:
    """Exact-hash and MinHash-LSH index; first registered row wins."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        d = pcfg.dedup_cfg(cfg)
        self.near_enabled = bool(d.get("near_dup_enabled", True))
        self.shingle_size = int(d.get("shingle_size", 5))
        self.threshold = float(d.get("jaccard_threshold", 0.85))
        self.num_perm = int(d.get("num_perm", 128))
        self._exact: set[str] = set()
        self._lsh: Any = None
        if self.near_enabled and not _HAS_DATASKETCH:
            warnings.warn(
                f"Near-dedup disabled: datasketch unavailable ({_DATASKETCH_IMPORT_ERROR})",
                stacklevel=2,
            )
            self.near_enabled = False
        if self.near_enabled and MinHashLSH is not None:
            self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._next_key = 0

    def _exact_hash(self, code: str) -> str:
        return hashlib.sha1(code.encode("utf-8")).hexdigest()

    def _token_shingles(self, code: str) -> list[str]:
        tokens = code.split()
        k = self.shingle_size
        if not tokens:
            return []
        if len(tokens) < k:
            return [" ".join(tokens)]
        return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]

    def _minhash(self, code: str) -> Any:
        mh = MinHash(num_perm=self.num_perm)  # type: ignore[misc]
        for shingle in self._token_shingles(code):
            mh.update(shingle.encode("utf-8"))
        return mh

    def is_duplicate(self, code: str) -> tuple[bool, str]:
        """Return (is_dup, reason) where reason is '' | 'exact' | 'near'."""
        if not code or not code.strip():
            return False, ""

        digest = self._exact_hash(code)
        if digest in self._exact:
            return True, "exact"

        if self._lsh is not None:
            mh = self._minhash(code)
            if self._lsh.query(mh):
                return True, "near"

        return False, ""

    def register(self, code: str) -> None:
        digest = self._exact_hash(code)
        self._exact.add(digest)
        if self._lsh is not None:
            mh = self._minhash(code)
            key = f"fn_{self._next_key}"
            self._next_key += 1
            self._lsh.insert(key, mh)


def _split_sort_key(split: Any) -> int:
    if split is None or (isinstance(split, float) and pd.isna(split)):
        return 99
    return SPLIT_PRIORITY.get(str(split).strip().lower(), 50)


def _resolve_index(
    indexes: dict[str, DedupIndex],
    global_index: DedupIndex,
    cfg: dict[str, Any],
    split: Any,
    cross_split: bool,
) -> DedupIndex:
    if cross_split:
        return global_index
    key = "none"
    if split is not None and not (isinstance(split, float) and pd.isna(split)):
        key = str(split).strip().lower()
    if key not in indexes:
        indexes[key] = DedupIndex(cfg)
    return indexes[key]


def deduplicate_dataframe(
    df: pd.DataFrame,
    dataset: str,
    cfg: dict[str, Any],
    global_index: DedupIndex,
    indexes: dict[str, DedupIndex],
    cross_split: bool,
    stats: DedupStats,
) -> pd.DataFrame:
    if "code" not in df.columns:
        raise ValueError(f"{dataset}: missing code column")

    stats.rows_in += len(df)
    work = df.copy()
    if cross_split and "split" in work.columns:
        work["_dedup_order"] = work["split"].map(_split_sort_key)
        work = work.sort_values("_dedup_order", kind="stable").drop(columns="_dedup_order")
    work = work.reset_index(drop=True)

    keep_rows: list[int] = []
    for i, row in work.iterrows():
        code = str(row["code"])
        split = row.get("split") if "split" in work.columns else None
        index = _resolve_index(indexes, global_index, cfg, split, cross_split)
        is_dup, reason = index.is_duplicate(code)
        if is_dup:
            if reason == "exact":
                stats.dropped_exact += 1
                stats.bump(dataset, "drop_exact")
            elif reason == "near":
                stats.dropped_near += 1
                stats.bump(dataset, "drop_near")
            continue
        index.register(code)
        keep_rows.append(i)

    out = work.loc[keep_rows].reset_index(drop=True)
    stats.rows_out += len(out)
    stats.datasets += 1
    stats.by_dataset.setdefault(dataset, {})["kept"] = len(out)
    return out


def run_dedup(cfg: dict[str, Any], stage: str) -> Path:
    in_dir = pcfg.cleaned_output_dir(cfg)
    out_dir = pcfg.deduped_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = DedupStats()
    dcfg = pcfg.dedup_cfg(cfg)
    cross_split = bool(dcfg.get("check_cross_split", True))
    global_index = DedupIndex(cfg)
    per_split_indexes: dict[str, DedupIndex] = {}

    in_paths = pcfg.cleaned_chunk_paths(cfg)
    if not in_paths:
        in_paths = pcfg.list_chunk_files(in_dir, pcfg.cleaned_chunk_prefix(cfg))
    out_prefix = pcfg.deduped_chunk_prefix(cfg)

    for src in in_paths:
        m = re.search(r"_(\d+)\.parquet$", src.name)
        idx = int(m.group(1)) if m else 1
        df = pd.read_parquet(src)
        deduped = deduplicate_dataframe(
            df, f"chunk_{idx}", cfg, global_index, per_split_indexes, cross_split, stats
        )
        deduped.to_parquet(out_dir / f"{out_prefix}_{idx}.parquet", index=False)

    report = cfg["_base_dir"] / dcfg.get("report", "reports/deduplicator_report.md")
    _write_report(stats, report, stage, in_dir, out_dir, global_index, cross_split)
    return out_dir


def _write_report(
    stats: DedupStats,
    path: Path,
    stage: str,
    in_dir: Path,
    out_dir: Path,
    index: DedupIndex,
    cross_split: bool,
) -> None:
    lines = [
        f"# Deduplicator report (stage {stage})",
        "",
        f"- datasets written: {stats.datasets}",
        f"- rows in: {stats.rows_in}",
        f"- rows out: {stats.rows_out}",
        f"- dropped (exact SHA1): {stats.dropped_exact}",
        f"- dropped (near MinHash-LSH >= {index.threshold}): {stats.dropped_near}",
        f"- near-dup enabled: {index.near_enabled}",
        f"- cross_split / cross_dataset index: {cross_split}",
        f"- input: `{in_dir.as_posix()}`",
        f"- output: `{out_dir.as_posix()}`",
        "",
        "## Per dataset",
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
    parser = argparse.ArgumentParser(description="Deduplicate cleaned dataset parquets")
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default="1a")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = pcfg.load_config(args.config)
    out = run_dedup(cfg, args.stage)
    print(f"Wrote deduped parquets under: {out}")


if __name__ == "__main__":
    main()
