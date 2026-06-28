"""
5_temporal_splitter.py — assign train / valid / test splits.

Pipeline step 5. Reads:  data/interim/deduped/deduped_*.parquet
Writes: data/interim/split/split_*.parquet

Temporal split modes (global scope, default):
  - **optimized** (default): count rows per year, search contiguous year boundaries to
    minimize squared error vs target ratios (~70/20/10).
  - **chronological**: greedy year stacking — stop before exceeding each split's row budget.
  - **fixed** (`splits.mode: fixed`): hard calendar cutoffs via `*_year_max_exclusive`.

All modes build one commit_hash -> split map and apply it to all sources.

Same commit_hash never crosses splits; commits whose rows span incompatible year groups are dropped.
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
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_pipeline._loader import cfg as pcfg


@dataclass
class SplitStats:
    datasets: int = 0
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped: int = 0
    commits_crossing_groups: int = 0
    rows_missing_date: int = 0
    by_dataset: dict[str, dict[str, Any]] = field(default_factory=dict)

    def bump(self, dataset: str, key: str, n: int = 1) -> None:
        self.by_dataset.setdefault(dataset, {}).setdefault(key, 0)
        self.by_dataset[dataset][key] += n


def _parse_date(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    text = str(value).strip()
    if not text or text.lower() in ("none", "null", "nat"):
        return None

    if re.fullmatch(r"\d{10,13}", text):
        ts = int(text[:10])
        try:
            return datetime.utcfromtimestamp(ts)
        except (OSError, ValueError):
            return None

    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _parse_year(value: Any) -> int | None:
    parsed = _parse_date(value)
    return parsed.year if parsed is not None else None


def _split_ratios(scfg: dict[str, Any]) -> tuple[float, float, float]:
    train_r = float(scfg.get("temporal_train_ratio", 0.7))
    valid_r = float(scfg.get("temporal_valid_ratio", 0.2))
    test_r = float(scfg.get("temporal_test_ratio", 0.1))
    total = train_r + valid_r + test_r
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"temporal split ratios must sum to 1.0, got {total}")
    return train_r, valid_r, test_r


def _split_mode(scfg: dict[str, Any]) -> str:
    return str(scfg.get("mode", "optimized")).lower()


def build_fixed_year_map(
    scfg: dict[str, Any],
    year_counts: dict[int, int],
) -> tuple[dict[int, str], dict[str, Any]]:
    """
    Assign splits by calendar year using exclusive upper bounds:
      train: year < train_year_max_exclusive
      valid: train_year_max_exclusive <= year < valid_year_max_exclusive
      test:  valid_year_max_exclusive <= year < test_year_max_exclusive
    Years >= test_year_max_exclusive are excluded (not in year_map).
    """
    train_max = int(scfg.get("train_year_max_exclusive", 2021))
    valid_max = int(scfg.get("valid_year_max_exclusive", 2022))
    test_max = int(scfg.get("test_year_max_exclusive", 2023))
    if not (train_max < valid_max < test_max):
        raise ValueError(
            "fixed year bounds must satisfy "
            "train_year_max_exclusive < valid_year_max_exclusive < test_year_max_exclusive, "
            f"got {train_max}, {valid_max}, {test_max}"
        )

    def split_for_year(year: int) -> str | None:
        if year < train_max:
            return "train"
        if year < valid_max:
            return "valid"
        if year < test_max:
            return "test"
        return None

    year_map: dict[int, str] = {}
    for year in year_counts:
        split = split_for_year(year)
        if split is not None:
            year_map[year] = split

    years = sorted(year_counts)
    total = sum(year_counts.values()) or 1
    train_c = sum(year_counts[y] for y in years if y < train_max)
    valid_c = sum(year_counts[y] for y in years if train_max <= y < valid_max)
    test_c = sum(year_counts[y] for y in years if valid_max <= y < test_max)
    excluded_c = sum(year_counts[y] for y in years if y >= test_max)

    meta: dict[str, Any] = {
        "mode": "fixed",
        "fixed_bounds": {
            "train": f"year < {train_max}",
            "valid": f"{train_max} <= year < {valid_max}",
            "test": f"{valid_max} <= year < {test_max}",
            "excluded": f"year >= {test_max}",
        },
        "year_row_counts": {int(y): int(year_counts[y]) for y in years},
        "boundaries": {
            "train_years": [y for y in years if y < train_max],
            "valid_years": [y for y in years if train_max <= y < valid_max],
            "test_years": [y for y in years if valid_max <= y < test_max],
            "excluded_years": [y for y in years if y >= test_max],
        },
        "actual_counts": {
            "train": train_c,
            "valid": valid_c,
            "test": test_c,
            "excluded": excluded_c,
        },
        "actual_ratios": {
            "train": train_c / total,
            "valid": valid_c / total,
            "test": test_c / total,
            "excluded": excluded_c / total,
        },
        "year_to_split": dict(sorted(year_map.items())),
    }
    return year_map, meta


def _greedy_year_block_end(
    years: list[int],
    counts: list[int],
    start: int,
    row_budget: float,
) -> int:
    """
    Return end index (exclusive) for a contiguous year block starting at `start`.
    Adds whole years while the running row sum stays at or under `row_budget`; if the
    first year alone exceeds the budget, still take that year so the split is non-empty.
    """
    cum = 0
    i = start
    n = len(years)
    while i < n:
        c = counts[i]
        if cum > 0 and cum + c > row_budget:
            break
        cum += c
        i += 1
    if i == start and start < n:
        return start + 1
    return i


def build_chronological_ratio_year_map(
    year_counts: dict[int, int],
    train_r: float = 0.7,
    valid_r: float = 0.2,
    test_r: float = 0.1,
) -> tuple[dict[int, str], dict[str, Any]]:
    """
    Greedy chronological split: oldest years -> train, then valid, then test.

    Counts rows per calendar year, sets row budgets from ratios (default 70/20/10),
    and walks years in order. Each split receives consecutive whole years; train and
    valid stop before adding a year that would exceed their budget. All later years go to test.
    """
    years = sorted(year_counts)
    counts = [year_counts[y] for y in years]
    n = len(years)
    total = sum(counts)

    meta: dict[str, Any] = {
        "mode": "chronological",
        "algorithm": (
            "greedy contiguous years: train/valid stop before exceeding row budget; "
            "remainder -> test"
        ),
        "total_rows": total,
        "year_row_counts": {int(y): int(year_counts[y]) for y in years},
        "target_ratios": {"train": train_r, "valid": valid_r, "test": test_r},
    }
    if n == 0 or total == 0:
        return {}, meta

    target_train = total * train_r
    target_valid = total * valid_r
    meta["target_counts"] = {
        "train": target_train,
        "valid": target_valid,
        "test": total * test_r,
    }

    train_end = _greedy_year_block_end(years, counts, 0, target_train)
    valid_end = _greedy_year_block_end(years, counts, train_end, target_valid)

    year_map: dict[int, str] = {}
    for idx, year in enumerate(years):
        if idx < train_end:
            year_map[year] = "train"
        elif idx < valid_end:
            year_map[year] = "valid"
        else:
            year_map[year] = "test"

    meta.update(_partition_meta(years, counts, train_end, valid_end, total, year_map))
    return year_map, meta


def resolve_year_map(
    scfg: dict[str, Any],
    year_counts: dict[int, int],
) -> tuple[dict[int, str], dict[str, Any]]:
    mode = _split_mode(scfg)
    if mode == "fixed":
        return build_fixed_year_map(scfg, year_counts)
    train_r, valid_r, test_r = _split_ratios(scfg)
    if mode == "chronological":
        return build_chronological_ratio_year_map(
            year_counts, train_r, valid_r, test_r
        )
    if mode not in ("optimized", "optimize"):
        raise ValueError(
            f"Unknown splits.mode={mode!r}; use optimized, chronological, or fixed"
        )
    return optimize_year_partitions(year_counts, train_r, valid_r, test_r)


def count_rows_per_year(df: pd.DataFrame, date_field: str) -> dict[int, int]:
    """Count rows for each calendar year present in the dataframe."""
    if date_field not in df.columns:
        return {}
    years = df[date_field].map(_parse_year)
    counts: dict[int, int] = defaultdict(int)
    for year in years:
        if year is None or (isinstance(year, float) and pd.isna(year)) or pd.isna(year):
            continue
        counts[int(year)] += 1
    return dict(counts)


def optimize_year_partitions(
    year_counts: dict[int, int],
    train_r: float = 0.7,
    valid_r: float = 0.2,
    test_r: float = 0.1,
) -> tuple[dict[int, str], dict[str, Any]]:
    """
    Partition sorted calendar years into three contiguous groups (train, valid, test)
    so row counts are as close as possible to the target ratios.

    Oldest years are always in train; newest in test. Searches all boundary pairs (i, j)
    where train = years[:i], valid = years[i:j], test = years[j:].
    """
    years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]
    n = len(years)
    total = sum(counts)

    meta: dict[str, Any] = {
        "mode": "optimized",
        "algorithm": (
            "search all contiguous year boundary pairs (train|valid|test); "
            "minimize squared error vs target row ratios"
        ),
        "total_rows": total,
        "year_row_counts": {int(y): int(year_counts[y]) for y in years},
        "target_ratios": {"train": train_r, "valid": valid_r, "test": test_r},
    }
    if n == 0 or total == 0:
        return {}, meta

    target_train = total * train_r
    target_valid = total * valid_r
    target_test = total * test_r
    meta["target_counts"] = {
        "train": target_train,
        "valid": target_valid,
        "test": target_test,
    }

    if n == 1:
        year_map = {years[0]: "train"}
        meta.update(
            _partition_meta(years, counts, 1, 1, total, year_map),
        )
        return year_map, meta

    best_cost = float("inf")
    best_i, best_j = 1, n

    # i = end index of train block; j = end index of valid block (test = years[j:])
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            train_c = sum(counts[:i])
            valid_c = sum(counts[i:j])
            test_c = sum(counts[j:])

            cost = (
                (train_c - target_train) ** 2
                + (valid_c - target_valid) ** 2
                + (test_c - target_test) ** 2
            )
            # Prefer non-empty valid/test when enough distinct years exist
            if n >= 3:
                if valid_c == 0:
                    cost += (total * 0.05) ** 2
                if test_c == 0:
                    cost += (total * 0.05) ** 2

            if cost < best_cost:
                best_cost = cost
                best_i, best_j = i, j

    year_map: dict[int, str] = {}
    for idx, year in enumerate(years):
        if idx < best_i:
            year_map[year] = "train"
        elif idx < best_j:
            year_map[year] = "valid"
        else:
            year_map[year] = "test"

    meta.update(_partition_meta(years, counts, best_i, best_j, total, year_map))
    meta["optimization_cost"] = best_cost
    return year_map, meta


def _partition_meta(
    years: list[int],
    counts: list[int],
    i: int,
    j: int,
    total: int,
    year_map: dict[int, str],
) -> dict[str, Any]:
    train_c = sum(counts[:i])
    valid_c = sum(counts[i:j])
    test_c = sum(counts[j:])
    return {
        "boundaries": {
            "train_years": years[:i],
            "valid_years": years[i:j],
            "test_years": years[j:],
        },
        "actual_counts": {"train": train_c, "valid": valid_c, "test": test_c},
        "actual_ratios": {
            "train": train_c / total,
            "valid": valid_c / total,
            "test": test_c / total,
        },
        "year_to_split": dict(sorted(year_map.items())),
    }


def collect_year_counts_global(
    in_paths: list[Path],
    cfg: dict[str, Any],
) -> dict[int, int]:
    """Pass 1 (global): row counts per year across all sources combined."""
    scfg = pcfg.splits_cfg(cfg)
    date_field = scfg.get("date_field", "commit_date")
    counts: dict[int, int] = defaultdict(int)

    for path in in_paths:
        df = pd.read_parquet(path)
        if date_field not in df.columns:
            continue
        for year, n in count_rows_per_year(df, date_field).items():
            counts[year] += n

    return dict(counts)


def collect_year_counts_by_source(
    in_paths: list[Path],
    cfg: dict[str, Any],
) -> dict[str, dict[int, int]]:
    """Pass 1: aggregate row counts per year, keyed by source_dataset (or 'all')."""
    scfg = pcfg.splits_cfg(cfg)
    date_field = scfg.get("date_field", "commit_date")
    by_source: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    cols = [date_field]
    for path in in_paths:
        df = pd.read_parquet(path)
        if date_field not in df.columns:
            continue
        use_cols = cols + (["source_dataset"] if "source_dataset" in df.columns else [])
        df = df[[c for c in use_cols if c in df.columns]]

        if "source_dataset" in df.columns:
            for source, group in df.groupby("source_dataset", dropna=False):
                key = str(source)
                for year, n in count_rows_per_year(group, date_field).items():
                    by_source[key][year] += n
        else:
            for year, n in count_rows_per_year(df, date_field).items():
                by_source["all"][year] += n

    return {k: dict(v) for k, v in by_source.items()}


def build_year_maps(
    year_counts_by_source: dict[str, dict[int, int]],
    cfg: dict[str, Any],
) -> dict[str, dict[int, str]]:
    scfg = pcfg.splits_cfg(cfg)
    maps: dict[str, dict[int, str]] = {}
    for source, counts in year_counts_by_source.items():
        year_map, _meta = resolve_year_map(scfg, counts)
        maps[source] = year_map
    return maps


def _commit_key(row: pd.Series, commit_field: str) -> str:
    commit = row.get(commit_field)
    if commit is not None and not (isinstance(commit, float) and pd.isna(commit)):
        text = str(commit).strip()
        if text and text.lower() not in ("none", "null"):
            return text
    row_id = row.get("id", row.name)
    return f"_row_{row_id}"


def _split_for_commit_years(years: set[int], year_map: dict[int, str]) -> str | None:
    splits: set[str] = set()
    for year in years:
        split = year_map.get(year)
        if split is None:
            return None
        splits.add(split)
    return splits.pop() if len(splits) == 1 else None


def _audit_temporal_order(year_map: dict[int, str], dataset: str) -> None:
    train_y = sorted(y for y, s in year_map.items() if s == "train")
    valid_y = sorted(y for y, s in year_map.items() if s == "valid")
    test_y = sorted(y for y, s in year_map.items() if s == "test")
    if train_y and valid_y and max(train_y) >= min(valid_y):
        raise ValueError(f"{dataset}: train years overlap valid years in partition")
    if valid_y and test_y and max(valid_y) >= min(test_y):
        raise ValueError(f"{dataset}: valid years overlap test years in partition")
    if train_y and test_y and max(train_y) >= min(test_y):
        raise ValueError(f"{dataset}: train years overlap test years in partition")


def _audit_commit_leakage(df: pd.DataFrame, commit_field: str, stats: SplitStats, dataset: str) -> None:
    if "split" not in df.columns:
        return
    if commit_field not in df.columns:
        keys = df.apply(lambda r: _commit_key(r, commit_field), axis=1)
        grouped = df.groupby(keys, dropna=False)["split"].nunique()
    else:
        grouped = df.groupby(commit_field, dropna=False)["split"].nunique()
    leaky = grouped[grouped > 1]
    if len(leaky):
        stats.bump(dataset, "commits_spanning_splits", len(leaky))
        raise ValueError(
            f"{dataset}: {len(leaky)} commit(s) appear in more than one split."
        )


def build_global_commit_to_split(
    in_paths: list[Path],
    cfg: dict[str, Any],
    year_map: dict[int, str],
    stats: SplitStats,
) -> dict[str, str]:
    """Pass 2 (global): one split per commit_hash from merged years across all chunks."""
    scfg = pcfg.splits_cfg(cfg)
    date_field = scfg.get("date_field", "commit_date")
    commit_field = scfg.get("commit_field", "commit_hash")
    commit_years: dict[str, set[int]] = defaultdict(set)

    use_cols = [date_field]
    if commit_field not in use_cols:
        use_cols.append(commit_field)
    if "id" not in use_cols:
        use_cols.append("id")

    for path in in_paths:
        df = pd.read_parquet(path)
        cols = [c for c in use_cols if c in df.columns]
        if date_field not in cols:
            continue
        sub = df[cols].copy()
        sub["_year"] = sub[date_field].map(_parse_year)
        sub["_commit_key"] = sub.apply(lambda r: _commit_key(r, commit_field), axis=1)
        for commit, group in sub.groupby("_commit_key", sort=False):
            for year in group["_year"]:
                if year is None or (isinstance(year, float) and pd.isna(year)) or pd.isna(year):
                    continue
                commit_years[str(commit)].add(int(year))

    commit_to_split: dict[str, str] = {}
    dropped_commits = 0
    for commit, years in commit_years.items():
        if not years:
            continue
        split = _split_for_commit_years(years, year_map)
        if split is None:
            dropped_commits += 1
            continue
        commit_to_split[commit] = split

    stats.commits_crossing_groups = dropped_commits
    stats.bump("global", "drop_commit_crosses_groups", dropped_commits)
    stats.by_dataset.setdefault("global", {})["commits_assigned"] = len(commit_to_split)
    stats.by_dataset["global"]["commits_dropped_cross_groups"] = dropped_commits
    return commit_to_split


def apply_global_commit_splits(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    commit_to_split: dict[str, str],
    stats: SplitStats,
) -> pd.DataFrame:
    """Pass 3 (global): apply precomputed commit_hash -> split to all rows."""
    scfg = pcfg.splits_cfg(cfg)
    date_field = scfg.get("date_field", "commit_date")
    commit_field = scfg.get("commit_field", "commit_hash")
    drop_no_date = bool(scfg.get("drop_rows_without_date", True))

    out = df.copy()
    if date_field not in out.columns:
        raise ValueError(f"global: missing date column '{date_field}' for temporal splits")

    out["_commit_key"] = out.apply(lambda r: _commit_key(r, commit_field), axis=1)
    assigned: list[str | None] = []
    keep_mask: list[bool] = []
    for commit in out["_commit_key"]:
        split = commit_to_split.get(commit)
        if split is None:
            assigned.append(None)
            keep_mask.append(not drop_no_date)
            continue
        assigned.append(split)
        keep_mask.append(True)

    out["split"] = assigned
    before = len(out)
    out = out[pd.Series(keep_mask, index=out.index)].copy()
    stats.rows_dropped += before - len(out)
    out = out.drop(columns=["_commit_key"], errors="ignore")
    return out.reset_index(drop=True)


def assign_splits_dynamic(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    stats: SplitStats,
    dataset: str,
    year_map: dict[int, str],
    partition_meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    scfg = pcfg.splits_cfg(cfg)
    date_field = scfg.get("date_field", "commit_date")
    commit_field = scfg.get("commit_field", "commit_hash")
    drop_no_date = bool(scfg.get("drop_rows_without_date", True))

    if not year_map:
        stats.rows_dropped += len(df)
        stats.bump(dataset, "drop_no_years_in_data", len(df))
        return df.iloc[0:0].copy()

    out = df.copy()
    if date_field not in out.columns:
        raise ValueError(f"{dataset}: missing date column '{date_field}' for temporal splits")

    out["_year"] = out[date_field].map(_parse_year)
    out["_commit_key"] = out.apply(lambda r: _commit_key(r, commit_field), axis=1)

    commit_to_split: dict[str, str] = {}
    for commit, group in out.groupby("_commit_key", sort=False):
        years = {
            int(y) for y in group["_year"].tolist()
            if y is not None and not (isinstance(y, float) and pd.isna(y)) and not pd.isna(y)
        }
        if not years:
            stats.rows_missing_date += len(group)
            stats.bump(dataset, "drop_no_date", len(group))
            continue

        split = _split_for_commit_years(years, year_map)
        if split is None:
            stats.commits_crossing_groups += 1
            stats.bump(dataset, "drop_commit_crosses_groups", len(group))
            continue

        commit_to_split[commit] = split

    keep_mask: list[bool] = []
    assigned: list[str | None] = []
    for _, row in out.iterrows():
        commit = row["_commit_key"]
        if commit not in commit_to_split:
            keep_mask.append(not drop_no_date)
            assigned.append(None)
            continue
        assigned.append(commit_to_split[commit])
        keep_mask.append(True)

    out["split"] = assigned
    before = len(out)
    out = out[pd.Series(keep_mask, index=out.index)].copy()
    stats.rows_dropped += before - len(out)
    out = out.drop(columns=["_year", "_commit_key"], errors="ignore")

    _audit_temporal_order(year_map, dataset)
    _audit_commit_leakage(out, commit_field, stats, dataset)

    info = stats.by_dataset.setdefault(dataset, {})
    info["mode"] = "dynamic_temporal"
    info["year_to_split"] = dict(sorted(year_map.items()))
    if partition_meta:
        for key in ("target_ratios", "target_counts", "actual_ratios", "actual_counts", "boundaries"):
            if key in partition_meta:
                info[key] = partition_meta[key]
    if "split" in out.columns:
        info["split_train"] = int((out["split"] == "train").sum())
        info["split_valid"] = int((out["split"] == "valid").sum())
        info["split_test"] = int((out["split"] == "test").sum())

    return out.reset_index(drop=True)


def split_dataframe(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    stats: SplitStats,
    year_maps: dict[str, dict[int, str]],
    partition_metas: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    partition_metas = partition_metas or {}

    if "source_dataset" not in df.columns:
        ymap = year_maps.get("all", {})
        if not ymap and year_maps:
            ymap = next(iter(year_maps.values()))
        return assign_splits_dynamic(
            df, cfg, stats, "all", ymap, partition_metas.get("all")
        )

    parts: list[pd.DataFrame] = []
    for source in df["source_dataset"].dropna().unique():
        src = str(source)
        sub = df[df["source_dataset"] == source].copy()
        ymap = year_maps.get(src, {})
        part = assign_splits_dynamic(
            sub, cfg, stats, src, ymap, partition_metas.get(src)
        )
        parts.append(part)

    if not parts:
        return df
    return pd.concat(parts, ignore_index=True)


def run_temporal_split(cfg: dict[str, Any], stage: str) -> Path:
    in_dir = pcfg.deduped_output_dir(cfg)
    out_dir = pcfg.split_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = SplitStats()
    scfg = pcfg.splits_cfg(cfg)
    scope = str(scfg.get("assignment_scope", "global")).lower()
    split_mode = _split_mode(scfg)

    in_paths = pcfg.deduped_chunk_paths(cfg)
    if not in_paths:
        in_paths = pcfg.list_chunk_files(in_dir, pcfg.deduped_chunk_prefix(cfg))
    out_prefix = pcfg.split_chunk_prefix(cfg)

    print(
        f"[splitter] reading {len(in_paths)} chunk(s) from {in_dir.as_posix()} "
        f"(assignment_scope={scope}, mode={split_mode})"
    )

    partition_metas: dict[str, dict[str, Any]] = {}

    if scope == "per_source":
        year_counts_by_source = collect_year_counts_by_source(in_paths, cfg)
        if not year_counts_by_source or all(
            not counts for counts in year_counts_by_source.values()
        ):
            raise ValueError(
                f"No parseable '{scfg.get('date_field', 'commit_date')}' values found in "
                f"{in_dir.as_posix()}."
            )

        year_maps: dict[str, dict[int, str]] = {}
        for source, counts in year_counts_by_source.items():
            ymap, meta = resolve_year_map(scfg, counts)
            year_maps[source] = ymap
            partition_metas[source] = meta
            _print_partition_summary(source, counts, meta)

        for src in in_paths:
            m = re.search(r"_(\d+)\.parquet$", src.name)
            idx = int(m.group(1)) if m else 1
            df = pd.read_parquet(src)
            stats.rows_in += len(df)
            split_df = split_dataframe(df, cfg, stats, year_maps, partition_metas)
            stats.rows_out += len(split_df)
            stats.datasets += 1
            split_df.to_parquet(out_dir / f"{out_prefix}_{idx}.parquet", index=False)
    else:
        global_counts = collect_year_counts_global(in_paths, cfg)
        if not global_counts:
            raise ValueError(
                f"No parseable '{scfg.get('date_field', 'commit_date')}' values found in "
                f"{in_dir.as_posix()}."
            )

        year_map, meta = resolve_year_map(scfg, global_counts)
        partition_metas["global"] = meta
        _print_partition_summary("global (all sources)", global_counts, meta)
        _audit_temporal_order(year_map, "global")

        print("[splitter] building global commit_hash -> split map ...", flush=True)
        commit_to_split = build_global_commit_to_split(in_paths, cfg, year_map, stats)
        print(
            f"[splitter] assigned {len(commit_to_split):,} commits; "
            f"dropped {stats.commits_crossing_groups:,} crossing year groups",
            flush=True,
        )

        for src in in_paths:
            m = re.search(r"_(\d+)\.parquet$", src.name)
            idx = int(m.group(1)) if m else 1
            df = pd.read_parquet(src)
            stats.rows_in += len(df)
            split_df = apply_global_commit_splits(df, cfg, commit_to_split, stats)
            stats.rows_out += len(split_df)
            stats.datasets += 1
            split_df.to_parquet(out_dir / f"{out_prefix}_{idx}.parquet", index=False)

        stats.by_dataset.setdefault("global", {})["mode"] = "global_temporal"
        stats.by_dataset["global"]["year_to_split"] = dict(sorted(year_map.items()))

    report = cfg["_base_dir"] / scfg.get("report", "reports/temporal_splitter_report.md")
    _write_report(stats, report, stage, in_dir, out_dir, partition_metas, scope, scfg)
    return out_dir


def _print_partition_summary(label: str, counts: dict[int, int], meta: dict[str, Any]) -> None:
    ratios = meta.get("actual_ratios", {})
    bounds = meta.get("boundaries", {})
    print(
        f"[splitter] {label}: years {min(counts)}-{max(counts)} -> "
        f"train {bounds.get('train_years', [])} ({ratios.get('train', 0):.1%}), "
        f"valid {bounds.get('valid_years', [])} ({ratios.get('valid', 0):.1%}), "
        f"test {bounds.get('test_years', [])} ({ratios.get('test', 0):.1%})"
    )


def _write_report(
    stats: SplitStats,
    path: Path,
    stage: str,
    in_dir: Path,
    out_dir: Path,
    partition_metas: dict[str, dict[str, Any]],
    scope: str = "global",
    scfg: dict[str, Any] | None = None,
) -> None:
    scfg = scfg or {}
    mode = _split_mode(scfg)
    if mode == "fixed":
        bounds = (
            f"train year < {scfg.get('train_year_max_exclusive', 2021)}, "
            f"valid year < {scfg.get('valid_year_max_exclusive', 2022)}, "
            f"test year < {scfg.get('test_year_max_exclusive', 2023)}"
        )
        mode_line = f"Mode: **fixed temporal** (`assignment_scope={scope}`) — {bounds}."
    elif mode == "chronological":
        mode_line = (
            f"Mode: **chronological temporal** (`assignment_scope={scope}`) — "
            "rows counted per year; consecutive years assigned to train/valid until the "
            "next year would exceed the configured row ratio, then remainder to test."
        )
    else:
        mode_line = (
            f"Mode: **optimized temporal** (`assignment_scope={scope}`) — year groups chosen "
            "from data to minimize squared error vs target ratios."
        )
    lines = [
        f"# Temporal splitter report (stage {stage})",
        "",
        mode_line,
        "",
        "Global scope uses one timeline and one split per `commit_hash` across all sources.",
        "",
        f"- datasets written: {stats.datasets}",
        f"- rows in: {stats.rows_in}",
        f"- rows out: {stats.rows_out}",
        f"- rows dropped (no date / bad commit): {stats.rows_dropped}",
        f"- commits dropped (cross year groups): {stats.commits_crossing_groups}",
        f"- rows without parseable date: {stats.rows_missing_date}",
        f"- input: `{in_dir.as_posix()}`",
        f"- output: `{out_dir.as_posix()}`",
        "",
        "## Per source partition",
        "",
    ]
    for ds, meta in sorted(partition_metas.items()):
        lines.append(f"### {ds}")
        for key in (
            "mode",
            "fixed_bounds",
            "year_row_counts",
            "target_ratios",
            "target_counts",
            "actual_ratios",
            "actual_counts",
            "boundaries",
            "year_to_split",
        ):
            if key in meta:
                lines.append(f"- {key}: {meta[key]}")
        lines.append("")

    lines.append("## Per dataset assignment stats")
    lines.append("")
    for ds, info in sorted(stats.by_dataset.items()):
        lines.append(f"### {ds}")
        if isinstance(info, dict):
            for k, v in sorted(info.items()):
                if k != "year_row_counts":
                    lines.append(f"- {k}: {v}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign temporal train/valid/test splits")
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default="1a")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = pcfg.load_config(args.config)
    out = run_temporal_split(cfg, args.stage)
    print(f"Wrote split parquets under: {out}")


if __name__ == "__main__":
    main()
