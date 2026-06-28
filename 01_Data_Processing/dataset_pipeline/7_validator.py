"""
7_validator.py — data quality gate on processed train / valid / test parquets.

Pipeline step 7. Reads:  data/processed/{train,valid,test}.parquet
Writes: reports/data_quality.md

Fails (exit code 1) when cross-split exact or near duplicates exceed configured limits.
No training should start until this gate passes.
"""

from __future__ import annotations

import importlib.util
import hashlib
import re
import sys
import warnings
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

_DATASKETCH_IMPORT_ERROR: Exception | None = None
try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except Exception as exc:
    MinHash = None  # type: ignore[misc, assignment]
    MinHashLSH = None  # type: ignore[misc, assignment]
    _HAS_DATASKETCH = False
    _DATASKETCH_IMPORT_ERROR = exc

from dataset_pipeline._loader import cfg as pcfg

SPLITS = ("train", "valid", "test")
REQUIRED_COLUMNS = [
    "id",
    "code",
    "label",
    "source_dataset",
    "commit_hash",
    "commit_date",
]


@dataclass
class SplitProfile:
    name: str
    rows: int = 0
    label_0: int = 0
    label_1: int = 0
    empty_code: int = 0
    date_min: str | None = None
    date_max: str | None = None
    code_len_p50: float = 0.0
    code_len_p90: float = 0.0
    code_len_p99: float = 0.0
    token_p50: float = 0.0
    token_p90: float = 0.0
    token_p99: float = 0.0
    over_trunc_tokens: int = 0
    by_source: dict[str, int] = field(default_factory=dict)


@dataclass
class ValidationResult:
    profiles: dict[str, SplitProfile] = field(default_factory=dict)
    cross_split_exact_hashes: int = 0
    cross_split_exact_rows: int = 0
    cross_split_near_pairs: int = 0
    commits_spanning_splits: int = 0
    near_dup_enabled: bool = False
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: bool = False

    def fail(self, msg: str) -> None:
        self.issues.append(msg)


def _validator_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("validator", {})


def rough_token_count(code: str) -> int:
    return len(re.findall(r"\S+", code))


def _code_digest(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8")).hexdigest()


class _NearDupIndex:
    """MinHash-LSH index (same settings as deduplicator)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        d = pcfg.dedup_cfg(cfg)
        self.shingle_size = int(d.get("shingle_size", 5))
        self.threshold = float(d.get("jaccard_threshold", 0.85))
        self.num_perm = int(d.get("num_perm", 128))
        self._lsh: Any = None
        if _HAS_DATASKETCH and MinHashLSH is not None:
            self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._next_key = 0

    @property
    def enabled(self) -> bool:
        return self._lsh is not None

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

    def has_near_match(self, code: str) -> bool:
        if self._lsh is None or not code.strip():
            return False
        mh = self._minhash(code)
        return bool(self._lsh.query(mh))

    def add(self, code: str) -> None:
        if self._lsh is None or not code.strip():
            return
        mh = self._minhash(code)
        key = str(self._next_key)
        self._next_key += 1
        self._lsh.insert(key, mh)


def _load_split_frames(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split_name in SPLITS:
        path = pcfg.builder_output_path(cfg, split_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name} at {path.as_posix()} — run 6_builder.py first."
            )
        frames[split_name] = pd.read_parquet(path)
    return frames


def _update_profile(
    profile: SplitProfile,
    df: pd.DataFrame,
    trunc_at: int,
) -> None:
    profile.rows = len(df)
    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        profile.label_1 = int((labels == 1).sum())
        profile.label_0 = int((labels == 0).sum())

    codes = df["code"].astype(str) if "code" in df.columns else pd.Series(dtype=str)
    profile.empty_code = int((codes.str.strip() == "").sum())

    code_lens = codes.str.len()
    if len(code_lens):
        profile.code_len_p50 = float(code_lens.quantile(0.5))
        profile.code_len_p90 = float(code_lens.quantile(0.9))
        profile.code_len_p99 = float(code_lens.quantile(0.99))

    tokens = codes.map(rough_token_count)
    if len(tokens):
        profile.token_p50 = float(tokens.quantile(0.5))
        profile.token_p90 = float(tokens.quantile(0.9))
        profile.token_p99 = float(tokens.quantile(0.99))
        profile.over_trunc_tokens = int((tokens > trunc_at).sum())

    if "commit_date" in df.columns:
        dates = pd.to_datetime(df["commit_date"], errors="coerce", utc=True).dropna()
        if len(dates):
            profile.date_min = str(dates.min().date())
            profile.date_max = str(dates.max().date())

    if "source_dataset" in df.columns:
        for source, n in df["source_dataset"].value_counts().items():
            profile.by_source[str(source)] = int(n)


def _check_schema(result: ValidationResult, frames: dict[str, pd.DataFrame]) -> None:
    for split_name, df in frames.items():
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            result.fail(f"{split_name}: missing columns {missing}")


def _check_cross_split_leakage(
    result: ValidationResult,
    frames: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> None:
    vcfg = _validator_cfg(cfg)
    max_exact = int(vcfg.get("max_cross_split_exact_dupes", 0))
    max_near = int(vcfg.get("max_cross_split_near_dupes", 0))
    near_enabled = bool(pcfg.dedup_cfg(cfg).get("near_dup_enabled", True))

    hash_splits: dict[str, set[str]] = {}
    commit_splits: dict[str, set[str]] = {}
    near_index = _NearDupIndex(cfg)
    result.near_dup_enabled = near_index.enabled

    if near_enabled and not near_index.enabled:
        result.warnings.append(
            f"Near-dup cross-split check skipped: datasketch unavailable "
            f"({_DATASKETCH_IMPORT_ERROR})"
        )

    check_near = bool(vcfg.get("check_cross_split_near_dup", True))
    if not check_near:
        near_enabled = False

    cross_near = 0
    progress_every = int(vcfg.get("progress_every", 25_000))

    for split_name in SPLITS:
        df = frames[split_name]
        codes = df["code"].astype(str)
        commits = (
            df["commit_hash"].astype(str)
            if "commit_hash" in df.columns
            else pd.Series([""] * len(df))
        )
        for i, (code, commit) in enumerate(zip(codes, commits, strict=True)):
            if progress_every and i and i % progress_every == 0:
                print(f"  leakage scan {split_name}: {i:,}/{len(df):,} rows...", flush=True)

            digest = _code_digest(code)
            hash_splits.setdefault(digest, set()).add(split_name)

            commit = commit.strip()
            if commit and commit.lower() not in ("none", "null", "nan"):
                commit_splits.setdefault(commit, set()).add(split_name)

            if near_index.enabled and near_enabled:
                if near_index.has_near_match(code):
                    cross_near += 1
                near_index.add(code)

    result.cross_split_exact_hashes = sum(1 for splits in hash_splits.values() if len(splits) > 1)
    result.cross_split_exact_rows = sum(
        sum(len(frames[s]) for s in splits) - 1
        for splits in hash_splits.values()
        if len(splits) > 1
    )
    result.cross_split_near_pairs = cross_near
    result.commits_spanning_splits = sum(1 for splits in commit_splits.values() if len(splits) > 1)

    if result.cross_split_exact_hashes > max_exact:
        result.fail(
            f"cross-split exact duplicate code hashes: {result.cross_split_exact_hashes} "
            f"(max allowed {max_exact})"
        )
    if near_enabled and near_index.enabled and cross_near > max_near:
        result.fail(
            f"cross-split near duplicates (Jaccard >= threshold): {cross_near} "
            f"(max allowed {max_near})"
        )
    if result.commits_spanning_splits > 0:
        result.fail(
            f"commits appearing in more than one split: {result.commits_spanning_splits}"
        )


def _check_temporal_order(result: ValidationResult, profiles: dict[str, SplitProfile]) -> None:
    train = profiles.get("train")
    valid = profiles.get("valid")
    test = profiles.get("test")
    if not train or not valid or not test:
        return
    if train.date_max and valid.date_min and train.date_max >= valid.date_min:
        result.warnings.append(
            f"train max date ({train.date_max}) >= valid min date ({valid.date_min})"
        )
    if valid.date_max and test.date_min and valid.date_max >= test.date_min:
        result.warnings.append(
            f"valid max date ({valid.date_max}) >= test min date ({test.date_min})"
        )


def validate_processed(cfg: dict[str, Any]) -> ValidationResult:
    vcfg = _validator_cfg(cfg)
    trunc_at = int(vcfg.get("report_truncation_at_tokens", 512))
    result = ValidationResult()

    frames = _load_split_frames(cfg)
    _check_schema(result, frames)

    for split_name, df in frames.items():
        if df.empty:
            result.fail(f"{split_name}: parquet is empty")
        profile = SplitProfile(name=split_name)
        _update_profile(profile, df, trunc_at)
        result.profiles[split_name] = profile

    if not result.issues:
        _check_cross_split_leakage(result, frames, cfg)
        _check_temporal_order(result, result.profiles)

    result.passed = len(result.issues) == 0
    return result


def _write_report(
    result: ValidationResult,
    path: Path,
    cfg: dict[str, Any],
    stage: str,
) -> None:
    vcfg = _validator_cfg(cfg)
    trunc_at = int(vcfg.get("report_truncation_at_tokens", 512))
    lines = [
        f"# Data quality report (stage {stage})",
        "",
        f"**Gate status: {'PASSED' if result.passed else 'FAILED'}**",
        "",
        "## Leakage gate (must pass before training)",
        "",
        f"- cross-split exact duplicate code hashes: **{result.cross_split_exact_hashes}** "
        f"(limit {vcfg.get('max_cross_split_exact_dupes', 0)})",
        f"- cross-split exact duplicate rows (in later splits): **{result.cross_split_exact_rows}**",
        f"- cross-split near duplicates: **{result.cross_split_near_pairs}** "
        f"(limit {vcfg.get('max_cross_split_near_dupes', 0)}, "
        f"near-dup check enabled: {result.near_dup_enabled})",
        f"- commits spanning multiple splits: **{result.commits_spanning_splits}**",
        "",
    ]

    if result.issues:
        lines.append("### Failures")
        lines.append("")
        for issue in result.issues:
            lines.append(f"- {issue}")
        lines.append("")

    if result.warnings:
        lines.append("### Warnings")
        lines.append("")
        for warn in result.warnings:
            lines.append(f"- {warn}")
        lines.append("")

    lines.append("## Per-split summary")
    lines.append("")
    for split_name in SPLITS:
        p = result.profiles.get(split_name)
        if not p:
            continue
        lines.append(f"### {split_name}")
        lines.append(f"- rows: {p.rows}")
        if p.rows:
            lines.append(
                f"- labels: vulnerable={p.label_1} ({p.label_1 / p.rows:.1%}), "
                f"benign={p.label_0} ({p.label_0 / p.rows:.1%})"
            )
        lines.append(f"- empty code: {p.empty_code}")
        if p.date_min:
            lines.append(f"- commit_date range: {p.date_min} → {p.date_max}")
        lines.append(
            f"- code length (chars) p50/p90/p99: {p.code_len_p50:.0f} / "
            f"{p.code_len_p90:.0f} / {p.code_len_p99:.0f}"
        )
        lines.append(
            f"- rough tokens p50/p90/p99: {p.token_p50:.0f} / {p.token_p90:.0f} / {p.token_p99:.0f}"
        )
        lines.append(
            f"- rows over {trunc_at} rough tokens (would need windowing): "
            f"{p.over_trunc_tokens} ({p.over_trunc_tokens / p.rows:.1%} of split)"
            if p.rows
            else f"- rows over {trunc_at} rough tokens: 0"
        )
        if p.by_source:
            lines.append(f"- by source_dataset: {dict(sorted(p.by_source.items()))}")
        lines.append("")

    lines.append("## CWE distribution")
    lines.append("")
    lines.append(
        "_CWE not in core schema for this pipeline run — skipped. "
        "Re-enable in normalizer if needed for thesis tables._"
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_validate(cfg: dict[str, Any], stage: str) -> bool:
    result = validate_processed(cfg)
    report = cfg["_base_dir"] / _validator_cfg(cfg).get(
        "output_report", "reports/data_quality.md"
    )
    _write_report(result, report, cfg, stage)

    status = "PASSED" if result.passed else "FAILED"
    print(f"Validator gate {status}: {report}")
    if result.issues:
        for issue in result.issues:
            print(f"  - {issue}")
    return result.passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed train/valid/test parquets")
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default="1a")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = pcfg.load_config(args.config)
    passed = run_validate(cfg, args.stage)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
