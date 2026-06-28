"""
9_batcher.py — Token windowing for BERT-safe sequence lengths.

Pipeline step 9. Reads:  data/processed/whole/{train,valid,test}.parquet
Writes: data/processed/whole/{train,valid,test}_windowed.parquet

Windowing only — no shard packing.
Functions over token_max are split into line-aware blocks (<= token_safe,
stretch to token_high / token_max for small tails). Each block becomes one
output row; rows share function_group_id so the embedder can mean-pool
windows back into a single function-level vector.

Non-windowed functions pass through unchanged with window_count=1.

DataLoader handles shuffling — no shard files needed.
"""

from __future__ import annotations

import importlib.util
import re
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

SPLITS = ("train", "valid", "test")

# Columns added by this step
WINDOW_COLUMNS = [
    "token_count",
    "is_windowed",
    "function_group_id",
    "window_index",
    "window_count",
]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class WindowerStats:
    by_split: dict[str, dict[str, int]] = field(default_factory=dict)

    def bump(self, split: str, key: str, n: int = 1) -> None:
        self.by_split.setdefault(split, {}).setdefault(key, 0)
        self.by_split[split][key] += n


# ---------------------------------------------------------------------------
# Token counter (CodeBERT tokenizer with whitespace fallback)
# ---------------------------------------------------------------------------

class TokenCounter:
    """
    Uses the configured model tokenizer when available.
    Falls back to whitespace token count so the pipeline
    never hard-fails due to a missing tokenizer.
    """

    def __init__(self, model_name: str) -> None:
        self._tokenizer = None
        try:
            from transformers import AutoTokenizer
            from transformers import logging as hf_logging

            from dataset_pipeline._hf_models import (
                is_filesystem_path,
                load_pretrained_kwargs,
                resolve_pretrained_source,
            )

            hf_logging.set_verbosity_error()
            source = (
                resolve_pretrained_source(model_name, require_weights=False)
                if is_filesystem_path(model_name)
                else model_name
            )
            load_kwargs = load_pretrained_kwargs(source)
            self._tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
            print(f"Windower: using tokenizer '{source}'")
        except Exception as exc:
            print(
                f"Windower: tokenizer unavailable ({exc}); "
                "falling back to whitespace token count."
            )

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            chunk = 200_000
            if len(text) <= chunk:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            total = 0
            for i in range(0, len(text), chunk):
                total += len(
                    self._tokenizer.encode(text[i : i + chunk], add_special_tokens=False)
                )
            return total
        return len(re.findall(r"\S+", text))

    def encode_lines(self, lines: list[str]) -> list[int]:
        return [self.count(line + ("\n" if not line.endswith("\n") else "")) for line in lines]


# ---------------------------------------------------------------------------
# Splitting helpers (preserved from original batcher logic)
# ---------------------------------------------------------------------------

def _split_by_char_budget(
    text: str,
    counter: TokenCounter,
    max_tok: int,
) -> list[tuple[str, int]]:
    """Binary-search character split when no word boundaries exist."""
    if not text:
        return [("", 0)]
    out: list[tuple[str, int]] = []
    start, n = 0, len(text)
    while start < n:
        lo, hi, best_end = start + 1, n, start + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if counter.count(text[start:mid]) <= max_tok:
                best_end = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best_end <= start:
            best_end = min(start + 1, n)
        piece = text[start:best_end]
        out.append((piece, counter.count(piece)))
        start = best_end
    return out


def _hard_split_by_tokens(
    text: str,
    counter: TokenCounter,
    safe: int,
    max_tok: int,
) -> list[tuple[str, int]]:
    """Word-aware split into chunks <= max_tok tokens."""
    words = text.split()
    if not words:
        return [("", 0)]

    windows: list[tuple[str, int]] = []
    chunk: list[str] = []
    chunk_tok = 0

    for word in words:
        word_tok = counter.count(word + " ")
        if word_tok > max_tok:
            if chunk:
                joined = " ".join(chunk)
                windows.append((joined, counter.count(joined)))
                chunk, chunk_tok = [], 0
            windows.extend(_split_by_char_budget(word, counter, max_tok))
            continue
        if chunk and chunk_tok + word_tok > safe:
            joined = " ".join(chunk)
            windows.append((joined, counter.count(joined)))
            chunk, chunk_tok = [word], word_tok
        else:
            chunk.append(word)
            chunk_tok += word_tok

    if chunk:
        joined = " ".join(chunk)
        windows.append((joined, counter.count(joined)))

    # Verify and recursively fix any over-limit windows
    fixed: list[tuple[str, int]] = []
    for w_code, _ in windows:
        actual = counter.count(w_code)
        if actual <= max_tok:
            fixed.append((w_code, actual))
        elif len(w_code.split()) <= 1:
            fixed.extend(_split_by_char_budget(w_code, counter, max_tok))
        else:
            fixed.extend(_hard_split_by_tokens(w_code, counter, safe, max_tok))
    return fixed or [("", 0)]


def _split_to_windows(
    code: str,
    counter: TokenCounter,
    safe: int,
    high: int,
    max_tok: int,
) -> list[tuple[str, int]]:
    """
    Split a function into (window_code, token_count) pairs.
    Prefers line-aware splits; falls back to word/char splits.
    Merges tiny tails into the previous window when within high limit.
    """
    if not code.strip():
        return [("", 0)]

    lines = code.splitlines()
    if not lines:
        return _hard_split_by_tokens(code, counter, safe, max_tok)

    # Single long line — no newlines to split on
    if len(lines) == 1:
        tok = counter.count(lines[0])
        if tok <= max_tok:
            return [(code, tok)]
        return _hard_split_by_tokens(code, counter, safe, max_tok)

    line_toks = counter.encode_lines(lines)
    windows: list[tuple[str, int]] = []
    buf_lines: list[str] = []
    buf_tok = 0

    def _flush() -> None:
        nonlocal buf_lines, buf_tok
        if buf_lines:
            windows.append(("\n".join(buf_lines), buf_tok))
            buf_lines, buf_tok = [], 0

    for line, ltok in zip(lines, line_toks):
        if buf_lines and buf_tok + ltok > safe:
            _flush()
        if ltok > max_tok:
            _flush()
            windows.extend(_hard_split_by_tokens(line, counter, safe, max_tok))
            continue
        buf_lines.append(line)
        buf_tok += ltok

    _flush()

    if not windows:
        return _hard_split_by_tokens(code, counter, safe, max_tok)

    # Merge tiny tail into previous window when it fits within high
    if len(windows) >= 2:
        last_code, last_tok = windows[-1]
        prev_code, prev_tok = windows[-2]
        if last_tok < safe // 3 and prev_tok + last_tok <= high:
            merged = prev_code + "\n" + last_code
            merged_tok = counter.count(merged)
            if merged_tok <= max_tok:
                windows[-2] = (merged, merged_tok)
                windows.pop()

    # Final verification pass
    fixed: list[tuple[str, int]] = []
    for w_code, _ in windows:
        actual = counter.count(w_code)
        if actual <= max_tok:
            fixed.append((w_code, actual))
        elif len(w_code.split()) <= 1:
            fixed.extend(_split_by_char_budget(w_code, counter, max_tok))
        else:
            fixed.extend(_hard_split_by_tokens(w_code, counter, safe, max_tok))
    return fixed or [("", 0)]


# ---------------------------------------------------------------------------
# Row expansion
# ---------------------------------------------------------------------------

def expand_row(
    row: dict[str, Any],
    counter: TokenCounter,
    safe: int,
    high: int,
    max_tok: int,
) -> list[dict[str, Any]]:
    """
    Expand one input row into one or more windowed rows.
    Non-windowed rows pass through with window_count=1.
    function_group_id is always set to the original row id
    so downstream mean-pooling can always group by it,
    even for single-window functions.
    """
    code = str(row.get("code") or "")
    group_id = str(row.get("id", ""))

    windows = _split_to_windows(code, counter, safe, high, max_tok)
    window_count = len(windows)
    is_windowed = window_count > 1

    out: list[dict[str, Any]] = []
    for idx, (w_code, w_tok) in enumerate(windows):
        new_row = dict(row)
        new_row["code"] = w_code
        new_row["token_count"] = int(w_tok)
        new_row["is_windowed"] = is_windowed
        new_row["function_group_id"] = group_id
        new_row["window_index"] = idx
        new_row["window_count"] = window_count
        # Give windowed rows a unique id so parquet rows are distinguishable
        if is_windowed:
            new_row["id"] = f"{group_id}_w{idx}"
        out.append(new_row)
    return out


# ---------------------------------------------------------------------------
# Split processor
# ---------------------------------------------------------------------------

def _batcher_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("batcher", {})


def _token_limits(cfg: dict[str, Any]) -> tuple[int, int, int]:
    b = _batcher_cfg(cfg)
    safe = int(b.get("token_safe", 450))
    high = int(b.get("token_high", 480))
    max_tok = int(b.get("token_max", 500))
    return safe, high, max_tok


def process_split(
    cfg: dict[str, Any],
    split_name: str,
    counter: TokenCounter,
    stats: WindowerStats,
) -> Path | None:
    safe, high, max_tok = _token_limits(cfg)

    in_path = pcfg.builder_output_path(cfg, split_name)
    if not in_path.exists():
        print(f"Windower: skipping {split_name} — {in_path.name} not found.")
        return None

    df = pd.read_parquet(in_path)
    print(f"  {split_name}: {len(df):,} functions in")

    all_rows: list[dict[str, Any]] = []
    funcs_windowed = 0
    extra_rows = 0

    for row in df.to_dict(orient="records"):
        # Normalise NaN to None so parquet round-trips cleanly
        clean = {
            k: (None if (isinstance(v, float) and pd.isna(v)) else v)
            for k, v in row.items()
        }
        expanded = expand_row(clean, counter, safe, high, max_tok)
        if len(expanded) > 1:
            funcs_windowed += 1
            extra_rows += len(expanded) - 1
        all_rows.extend(expanded)

    out_df = pd.DataFrame(all_rows)
    out_path = pcfg.windowed_output_path(cfg, split_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    stats.bump(split_name, "functions_in", len(df))
    stats.bump(split_name, "rows_out", len(all_rows))
    stats.bump(split_name, "functions_windowed", funcs_windowed)
    stats.bump(split_name, "extra_window_rows", extra_rows)

    print(
        f"  {split_name}: {len(all_rows):,} rows out "
        f"({funcs_windowed} functions windowed, +{extra_rows} extra rows) "
        f"→ {out_path.name}"
    )
    return out_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(
    stats: WindowerStats,
    path: Path,
    stage: str,
    cfg: dict[str, Any],
) -> None:
    safe, high, max_tok = _token_limits(cfg)
    bcfg = _batcher_cfg(cfg)
    model_name = str(
        bcfg.get(
            "tokenizer_name",
            cfg.get("model", {}).get("codebert_name", "microsoft/codebert-base"),
        )
    )

    lines = [
        f"# Windower report (stage {stage})",
        "",
        "Token windowing only — no shard packing.",
        "Each split produces a single `*_windowed.parquet`.",
        "DataLoader handles mini-batch shuffling during training.",
        "",
        f"- tokenizer: `{model_name}`",
        f"- token_safe / token_high / token_max: {safe} / {high} / {max_tok}",
        "",
        "## Per split",
        "",
    ]

    for split_name in SPLITS:
        info = stats.by_split.get(split_name, {})
        if not info:
            continue
        funcs_in = info.get("functions_in", 0)
        rows_out = info.get("rows_out", 0)
        windowed = info.get("functions_windowed", 0)
        extra = info.get("extra_window_rows", 0)
        pct = windowed / funcs_in * 100 if funcs_in else 0.0

        lines += [
            f"### {split_name}",
            f"- functions in:        {funcs_in:,}",
            f"- rows out:            {rows_out:,}",
            f"- functions windowed:  {windowed:,} ({pct:.1f}% of split)",
            f"- extra rows created:  {extra:,}",
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_window(cfg: dict[str, Any], stage: str) -> Path:
    bcfg = _batcher_cfg(cfg)
    model_name = str(
        bcfg.get(
            "tokenizer_name",
            cfg.get("model", {}).get("codebert_name", "microsoft/codebert-base"),
        )
    )
    counter = TokenCounter(model_name)
    stats = WindowerStats()
    safe, high, max_tok = _token_limits(cfg)

    print(
        f"Windower: token limits safe={safe} high={high} max={max_tok}"
    )

    for split_name in SPLITS:
        process_split(cfg, split_name, counter, stats)

    report = cfg["_base_dir"] / bcfg.get("report", "reports/windower_report.md")
    _write_report(stats, report, stage, cfg)
    print(f"Windower report: {report}")

    return pcfg.processed_output_dir(cfg)


# Backward-compatible alias for run_pipeline.py and older scripts.
run_batch = run_window


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token-window processed splits for BERT (512 token limit)."
    )
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = pcfg.load_config(args.config)
    stage = args.stage or str(cfg.get("stage", "1a"))
    run_window(cfg, stage)


if __name__ == "__main__":
    main()