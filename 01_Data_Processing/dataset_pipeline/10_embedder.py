"""
10_embedder.py — GraphCodeBERT window embedding extraction.

Pipeline step 10. Reads:  data/processed/{train,valid,test}/{split}_batch_{n}.parquet
Writes: data/embeddings/{variant}/{train,valid,test}/{split}_window_embeddings.parquet

Each window row is encoded with GraphCodeBERT. Function-level risk is computed at
train/inference time by pooling window probabilities, not embedding vectors.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "dataset_pipeline._runtime",
    Path(__file__).resolve().parent / "_runtime.py",
)
_runtime = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_runtime)
_runtime.ensure_app_root(__file__)

import argparse

from dataset_pipeline._loader import cfg as pcfg
from dataset_pipeline._reports import report_header, step_report_path, write_report

SPLITS = ("train", "valid", "test")


@dataclass
class SplitEmbedStats:
    windows_in: int = 0
    functions_out: int = 0
    shards_read: int = 0


@dataclass
class EmbedStats:
    by_split: dict[str, SplitEmbedStats] = field(default_factory=dict)

    def split(self, name: str) -> SplitEmbedStats:
        return self.by_split.setdefault(name, SplitEmbedStats())


def _embedder_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return pcfg.embedder_cfg(cfg)


def _resolve_model_path(cfg: dict[str, Any]) -> Path:
    ecfg = _embedder_cfg(cfg)
    rel = str(ecfg.get("model_path", "02_ML_Model/graphcodebert-base"))
    path = Path(rel)
    if path.is_absolute():
        return path
    return pcfg.PROJECT_ROOT / path


class GraphCodeEncoder:
    def __init__(
        self,
        model_path: str | Path,
        *,
        max_length: int,
        fp16: bool,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()

        self._torch = torch
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from dataset_pipeline._hf_models import load_pretrained_kwargs, resolve_pretrained_source

        source = resolve_pretrained_source(model_path, require_weights=True)
        load_kwargs = load_pretrained_kwargs(source)
        self.tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
        self.model = AutoModel.from_pretrained(source, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        if self.fp16:
            self.model.half()
        print(f"Embedder: model={source} device={self.device} fp16={self.fp16}")

    def encode(self, codes: list[str], batch_size: int) -> np.ndarray:
        torch = self._torch
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(codes), batch_size):
                batch_codes = codes[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_codes,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                model_out = self.model(**encoded)
                hidden = model_out.last_hidden_state
                if self.fp16:
                    hidden = hidden.float()
                cls_vectors = hidden[:, 0, :].cpu().numpy().astype(np.float32)
                outputs.append(cls_vectors)
        if not outputs:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        return np.vstack(outputs)


@dataclass
class WindowRecord:
    window_id: str
    function_group_id: str
    label: int
    embedding: np.ndarray


def _read_shard_frames(
    cfg: dict[str, Any],
    split_name: str,
    *,
    smoke_test_shards: int | None = None,
) -> list[pd.DataFrame]:
    shards = pcfg.list_batch_shards(cfg, split_name)
    if not shards:
        fallback = pcfg.windowed_output_path(cfg, split_name)
        if fallback.exists():
            print(
                f"Embedder: no batch shards for {split_name}; "
                f"falling back to {fallback.name}"
            )
            return [pd.read_parquet(fallback)]
        return []

    if smoke_test_shards is not None:
        shards = shards[:smoke_test_shards]
    return [pd.read_parquet(path) for path in shards]


def _collect_split_windows(
    cfg: dict[str, Any],
    split_name: str,
    encoder: GraphCodeEncoder,
    *,
    smoke_test_windows: int | None = None,
    smoke_test_shards: int | None = None,
) -> tuple[list[WindowRecord], SplitEmbedStats]:
    ecfg = _embedder_cfg(cfg)
    code_col = str(ecfg.get("code_column", "code"))
    label_col = str(ecfg.get("label_column", "label"))
    window_id_col = str(ecfg.get("window_id_column", "id"))
    group_col = str(ecfg.get("function_group_column", "function_group_id"))
    batch_size = int(ecfg.get("batch_size", 16))

    stats = SplitEmbedStats()
    window_records: list[WindowRecord] = []
    frames = _read_shard_frames(
        cfg,
        split_name,
        smoke_test_shards=smoke_test_shards,
    )
    stats.shards_read = len(frames)

    windows_seen = 0
    for frame in frames:
        if frame.empty:
            continue
        for col in (code_col, label_col, window_id_col, group_col):
            if col not in frame.columns:
                raise KeyError(
                    f"{split_name}: missing required column {col!r} in batch shard"
                )

        rows = frame.to_dict(orient="records")
        if smoke_test_windows is not None:
            remaining = smoke_test_windows - windows_seen
            if remaining <= 0:
                break
            rows = rows[:remaining]

        codes = [str(row.get(code_col) or "") for row in rows]
        embeddings = encoder.encode(codes, batch_size)

        for row, vector in zip(rows, embeddings):
            window_records.append(
                WindowRecord(
                    window_id=str(row[window_id_col]),
                    function_group_id=str(row[group_col]),
                    label=int(row[label_col]),
                    embedding=vector,
                )
            )
            windows_seen += 1

    stats.windows_in = len(window_records)
    return window_records, stats


def _build_window_rows(window_records: list[WindowRecord]) -> list[dict[str, Any]]:
    return [
        {
            "window_id": record.window_id,
            "function_group_id": record.function_group_id,
            "label": int(record.label),
            "embedding": record.embedding.astype(np.float32),
        }
        for record in window_records
    ]


def _write_split_outputs(
    cfg: dict[str, Any],
    split_name: str,
    window_rows: list[dict[str, Any]],
    *,
    output_subdir: str,
) -> Path | None:
    out_dir = pcfg.embedding_split_dir(cfg, split_name, pool=output_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_path = pcfg.embedding_window_split_path(cfg, split_name, pool=output_subdir)
    pd.DataFrame(window_rows).to_parquet(window_path, index=False)
    return window_path


def process_split(
    cfg: dict[str, Any],
    split_name: str,
    encoder: GraphCodeEncoder,
    stats: EmbedStats,
    *,
    output_subdir: str,
    smoke_test_windows: int | None = None,
    smoke_test_shards: int | None = None,
) -> Path | None:
    window_records, split_stats = _collect_split_windows(
        cfg,
        split_name,
        encoder,
        smoke_test_windows=smoke_test_windows,
        smoke_test_shards=smoke_test_shards,
    )
    stats.by_split[split_name] = split_stats

    if not window_records:
        print(f"Embedder: skipping {split_name} - no batch shard rows found.")
        return None

    window_rows = _build_window_rows(window_records)
    split_stats.functions_out = len({row["function_group_id"] for row in window_rows})
    window_path = _write_split_outputs(cfg, split_name, window_rows, output_subdir=output_subdir)
    print(
        f"  {split_name}: {split_stats.windows_in:,} windows "
        f"({split_stats.functions_out:,} functions, {split_stats.shards_read} shard(s)) "
        f"-> {window_path.name if window_path else 'n/a'}"
    )
    return window_path


def _write_report(
    stats: EmbedStats,
    path: Path,
    stage: str,
    cfg: dict[str, Any],
    *,
    variant: str,
    output_subdir: str,
    output_dir: Path,
    summary_path: Path,
) -> None:
    ecfg = _embedder_cfg(cfg)
    lines = report_header("10_embedder — GraphCodeBERT windows", stage, script_stem="10_embedder")
    lines += [
        "GraphCodeBERT encodes each token window; function risk is pooled from window probabilities at train/inference time.",
        "",
        f"- model: `{ecfg.get('model_path', '02_ML_Model/graphcodebert-base')}`",
        f"- encoder variant: `{variant}`",
        f"- output subdir: `{output_subdir}`",
        f"- output dir: `{output_dir.as_posix()}`",
        f"- summary: `{summary_path.as_posix()}`",
        "",
        "## Per split",
        "",
    ]
    for split_name in SPLITS:
        split_stats = stats.by_split.get(split_name)
        if split_stats is None:
            continue
        lines += [
            f"### {split_name}",
            f"- shards read: {split_stats.shards_read}",
            f"- windows encoded: {split_stats.windows_in:,}",
            f"- distinct functions: {split_stats.functions_out:,}",
            "",
        ]
    total_windows = sum(s.windows_in for s in stats.by_split.values())
    total_funcs = sum(s.functions_out for s in stats.by_split.values())
    lines += [
        "## Analysis",
        "",
        f"- **Encoded:** {total_windows:,} window vectors across {total_funcs:,} functions.",
        "- Embedding dimension and dtype are recorded in `embedding_extraction_summary.json`.",
        "",
    ]
    write_report(path, lines)


def run_embed(
    cfg: dict[str, Any],
    stage: str,
    *,
    variant: str | None = None,
    output_subdir: str | None = None,
    smoke_test: bool = False,
) -> Path:
    ecfg = _embedder_cfg(cfg)
    variant = variant or str(ecfg.get("variant", "base"))
    output_subdir = output_subdir or str(ecfg.get("output_subdir", ecfg.get("pool", "max")))
    model_path = _resolve_model_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(f"GraphCodeBERT model not found: {model_path}")

    encoder = GraphCodeEncoder(
        model_path,
        max_length=int(ecfg.get("max_length", 512)),
        fp16=bool(ecfg.get("fp16", True)),
    )
    stats = EmbedStats()
    smoke_windows = int(ecfg.get("smoke_test_windows", 3000)) if smoke_test else None
    smoke_shards = int(ecfg.get("smoke_test_shards", 3)) if smoke_test else None

    print(f"Embedder: variant={variant} output={output_subdir}")
    window_output_paths: dict[str, str] = {}
    for split_name in SPLITS:
        window_path = process_split(
            cfg,
            split_name,
            encoder,
            stats,
            output_subdir=output_subdir,
            smoke_test_windows=smoke_windows,
            smoke_test_shards=smoke_shards,
        )
        if window_path is not None:
            window_output_paths[split_name] = str(window_path)

    out_root = pcfg.embeddings_output_root(cfg) / output_subdir
    summary = {
        "stage": stage,
        "variant": variant,
        "output_subdir": output_subdir,
        "model_path": str(model_path),
        "output_dir": str(out_root),
        "smoke_test": smoke_test,
        "splits": {
            split: {
                "shards_read": stats.by_split[split].shards_read,
                "windows_in": stats.by_split[split].windows_in,
                "functions_out": stats.by_split[split].functions_out,
                "window_embeddings_path": window_output_paths.get(split),
            }
            for split in SPLITS
            if split in stats.by_split
        },
    }
    summary_path = out_root / "embedding_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = step_report_path(cfg, "10_embedder")
    _write_report(
        stats,
        report,
        stage,
        cfg,
        variant=variant,
        output_subdir=output_subdir,
        output_dir=out_root,
        summary_path=summary_path,
    )
    print(f"Embedder report: {report}")
    print(f"Embedder summary: {summary_path}")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract GraphCodeBERT window embeddings from batch shards."
    )
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--variant",
        choices=["base"],
        default=None,
        help="Encoder variant label (default: embedder.variant in config).",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = pcfg.load_config(args.config)
    stage = args.stage or str(cfg.get("stage", "1a"))
    run_embed(cfg, stage, variant=args.variant, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
