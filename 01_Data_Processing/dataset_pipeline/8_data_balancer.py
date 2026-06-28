"""
8_data_balancer.py — downsample majority class to target label ratios.

Pipeline step 8. Reads / writes: data/processed/whole/{train,valid,test}.parquet

Runs after 7_validator (gate on unbalanced splits) and before 9_batcher.

Only downsamples the majority class (never upsamples the minority). For each
configured split (default: train, valid, test), if vulnerable % is below target,
random benign rows are removed; if above target, random vulnerable rows are removed.
Uses config seed for reproducibility.

Note: 30/70 is for comparable metrics and stable training; real-world C/C++ repos
are often ~5–15% vulnerable — acknowledge this in thesis writeup.
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

import numpy as np
import pandas as pd

from dataset_pipeline._loader import cfg as pcfg

SPLITS = ("train", "valid", "test")
LABEL_VULN = 1
LABEL_BENIGN = 0


@dataclass
class SplitBalanceStats:
    rows_in: int = 0
    rows_out: int = 0
    vuln_in: int = 0
    benign_in: int = 0
    vuln_out: int = 0
    benign_out: int = 0
    vuln_removed: int = 0
    benign_removed: int = 0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class BalanceStats:
    by_split: dict[str, SplitBalanceStats] = field(default_factory=dict)


def _balancer_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return pcfg.balancer_cfg(cfg)


def _target_ratios(bcfg: dict[str, Any]) -> tuple[float, float]:
    vuln_r = float(bcfg.get("target_vulnerable_ratio", 0.30))
    benign_r = float(bcfg.get("target_benign_ratio", 0.70))
    total = vuln_r + benign_r
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"balancer target_vulnerable_ratio + target_benign_ratio must sum to 1.0, got {total}"
        )
    if vuln_r <= 0 or benign_r <= 0:
        raise ValueError("balancer target ratios must be positive")
    return vuln_r, benign_r


def _label_mask(series: pd.Series, vuln_value: int) -> tuple[np.ndarray, np.ndarray]:
    labels = pd.to_numeric(series, errors="coerce").fillna(-1).astype(int)
    vuln = (labels == vuln_value).to_numpy()
    benign = (labels != vuln_value).to_numpy()
    unknown = ~(vuln | benign)
    if unknown.any():
        n_bad = int(unknown.sum())
        raise ValueError(f"Found {n_bad} rows with unexpected label values")
    return vuln, benign


def _keep_counts(
    n_vuln: int,
    n_benign: int,
    target_vuln: float,
) -> tuple[int, int, str]:
    """
    Return (n_vuln_keep, n_benign_keep, action) using downsample-only logic.

    action: none | downsample_benign | downsample_vuln
    """
    total = n_vuln + n_benign
    if total == 0:
        return 0, 0, "empty"

    actual_vuln = n_vuln / total
    tol = 1e-4
    if abs(actual_vuln - target_vuln) <= tol:
        return n_vuln, n_benign, "none"

    if actual_vuln < target_vuln:
        # Minority vuln: reduce benign count so n_vuln / (n_vuln + n_benign_keep) ~= target
        if n_vuln == 0:
            return 0, 0, "no_vuln_rows"
        n_benign_keep = int(np.floor(n_vuln * (1.0 - target_vuln) / target_vuln))
        n_benign_keep = min(n_benign_keep, n_benign)
        return n_vuln, n_benign_keep, "downsample_benign"

    # actual_vuln > target: reduce vuln count
    if n_benign == 0:
        return 0, 0, "no_benign_rows"
    n_vuln_keep = int(np.floor(n_benign * target_vuln / (1.0 - target_vuln)))
    n_vuln_keep = min(n_vuln_keep, n_vuln)
    return n_vuln_keep, n_benign, "downsample_vuln"


def balance_dataframe(
    df: pd.DataFrame,
    *,
    target_vuln: float,
    label_field: str,
    vuln_value: int,
    seed: int,
) -> tuple[pd.DataFrame, SplitBalanceStats]:
    stats = SplitBalanceStats(rows_in=len(df))
    if df.empty:
        stats.skipped = True
        stats.skip_reason = "empty"
        return df, stats

    if label_field not in df.columns:
        raise KeyError(f"Missing label column {label_field!r}")

    vuln_mask, benign_mask = _label_mask(df[label_field], vuln_value)
    n_vuln = int(vuln_mask.sum())
    n_benign = int(benign_mask.sum())
    stats.vuln_in = n_vuln
    stats.benign_in = n_benign

    n_vuln_keep, n_benign_keep, action = _keep_counts(n_vuln, n_benign, target_vuln)

    if action in ("empty", "no_vuln_rows", "no_benign_rows"):
        stats.skipped = True
        stats.skip_reason = action
        stats.rows_out = len(df)
        stats.vuln_out = n_vuln
        stats.benign_out = n_benign
        return df, stats

    if action == "none":
        stats.skipped = True
        stats.skip_reason = "already_at_target"
        stats.rows_out = len(df)
        stats.vuln_out = n_vuln
        stats.benign_out = n_benign
        return df, stats

    rng = np.random.default_rng(seed)
    keep = np.zeros(len(df), dtype=bool)

    vuln_idx = np.flatnonzero(vuln_mask)
    benign_idx = np.flatnonzero(benign_mask)

    if action == "downsample_benign":
        keep[vuln_idx] = True
        if n_benign_keep > 0:
            chosen = rng.choice(benign_idx, size=n_benign_keep, replace=False)
            keep[chosen] = True
        stats.benign_removed = n_benign - n_benign_keep
    else:  # downsample_vuln
        keep[benign_idx] = True
        if n_vuln_keep > 0:
            chosen = rng.choice(vuln_idx, size=n_vuln_keep, replace=False)
            keep[chosen] = True
        stats.vuln_removed = n_vuln - n_vuln_keep

    out = df.loc[keep].reset_index(drop=True)
    stats.rows_out = len(out)
    stats.vuln_out = int((pd.to_numeric(out[label_field], errors="coerce") == vuln_value).sum())
    stats.benign_out = stats.rows_out - stats.vuln_out
    return out, stats


def _splits_to_balance(bcfg: dict[str, Any]) -> list[str]:
    raw = bcfg.get("splits", list(SPLITS))
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip().lower()]
    return [str(s).strip().lower() for s in raw]


def balance_split(
    cfg: dict[str, Any],
    split_name: str,
    target_vuln: float,
    stats: BalanceStats,
) -> Path | None:
    bcfg = _balancer_cfg(cfg)
    path = pcfg.builder_output_path(cfg, split_name)
    if not path.exists():
        print(f"Balancer: skipping {split_name} (no {path.name}).")
        return None

    df = pd.read_parquet(path)
    label_field = str(bcfg.get("label_field", "label"))
    vuln_value = int(bcfg.get("vulnerable_label", 1))
    seed = int(bcfg.get("seed", cfg.get("seed", 42)))
    split_seed = seed + sum(ord(c) for c in split_name)

    balanced, split_stats = balance_dataframe(
        df,
        target_vuln=target_vuln,
        label_field=label_field,
        vuln_value=vuln_value,
        seed=split_seed,
    )
    stats.by_split[split_name] = split_stats

    path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_parquet(path, index=False)

    if split_stats.skipped:
        print(
            f"  {split_name}: {split_stats.rows_in:,} rows unchanged ({split_stats.skip_reason}) "
            f"vuln={split_stats.vuln_in / max(split_stats.rows_in, 1):.1%}"
        )
    else:
        print(
            f"  {split_name}: {split_stats.rows_in:,} -> {split_stats.rows_out:,} "
            f"(vuln {split_stats.vuln_in:,}->{split_stats.vuln_out:,}, "
            f"benign {split_stats.benign_in:,}->{split_stats.benign_out:,}, "
            f"removed vuln={split_stats.vuln_removed:,} benign={split_stats.benign_removed:,}) "
            f"actual vuln%={split_stats.vuln_out / max(split_stats.rows_out, 1):.1%}"
        )
    return path


def run_balance(cfg: dict[str, Any], stage: str) -> Path:
    bcfg = _balancer_cfg(cfg)
    if not bool(bcfg.get("enabled", True)):
        print("Balancer: disabled in config (balancer.enabled: false).")
        return pcfg.processed_whole_dir(cfg)

    target_vuln, target_benign = _target_ratios(bcfg)
    splits = _splits_to_balance(bcfg)
    stats = BalanceStats()

    print(
        f"Balancer: target vulnerable={target_vuln:.1%} benign={target_benign:.1%} "
        f"splits={splits or '(none)'}"
    )

    for split_name in splits:
        if split_name not in SPLITS:
            raise ValueError(f"Unknown balancer split {split_name!r}; use train, valid, or test")
        balance_split(cfg, split_name, target_vuln, stats)

    report = cfg["_base_dir"] / bcfg.get("report", "reports/balancer_report.md")
    _write_report(stats, report, stage, cfg, target_vuln, target_benign, splits)
    return pcfg.processed_whole_dir(cfg)


def _write_report(
    stats: BalanceStats,
    path: Path,
    stage: str,
    cfg: dict[str, Any],
    target_vuln: float,
    target_benign: float,
    splits: list[str],
) -> None:
    lines = [
        f"# Data balancer report (stage {stage})",
        "",
        "Downsample-only label balancing on processed whole parquets.",
        "",
        f"- target vulnerable: **{target_vuln:.1%}**",
        f"- target benign: **{target_benign:.1%}**",
        f"- splits balanced: {', '.join(splits) if splits else '(none)'}",
        f"- output: `{pcfg.processed_whole_dir(cfg).as_posix()}/`",
        "",
        "> **Evaluation caveat:** Balanced splits improve comparability and training stability.",
        "> Production C/C++ codebases are typically ~5–15% vulnerable; absolute F1 on 30/70",
        "> splits may overstate real-world performance — state this explicitly in the thesis.",
        "",
        "## Per split",
        "",
    ]
    for split_name in splits:
        s = stats.by_split.get(split_name)
        if s is None:
            continue
        lines.append(f"### {split_name}")
        lines.append(f"- rows in: {s.rows_in}")
        lines.append(f"- rows out: {s.rows_out}")
        lines.append(f"- vulnerable in: {s.vuln_in} ({100.0 * s.vuln_in / max(s.rows_in, 1):.2f}%)")
        lines.append(f"- benign in: {s.benign_in}")
        if s.rows_out:
            lines.append(
                f"- vulnerable out: {s.vuln_out} ({100.0 * s.vuln_out / s.rows_out:.2f}%)"
            )
            lines.append(f"- benign out: {s.benign_out}")
        lines.append(f"- removed vulnerable: {s.vuln_removed}")
        lines.append(f"- removed benign: {s.benign_removed}")
        if s.skipped:
            lines.append(f"- skipped: {s.skip_reason}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance label ratios in processed splits")
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = pcfg.load_config(args.config)
    stage = args.stage or str(cfg.get("stage", "1a"))
    out = run_balance(cfg, stage)
    print(f"Wrote balanced whole parquets under: {out}")


if __name__ == "__main__":
    main()
