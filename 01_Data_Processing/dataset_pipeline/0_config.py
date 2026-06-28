"""0_config.py — load and query configs/global_config.yaml (paths, stages, interim dirs)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def list_chunk_files(directory: Path, prefix: str) -> list[Path]:
    """Return sorted paths like {prefix}_1.parquet, {prefix}_2.parquet, ..."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.parquet$")
    files: list[tuple[int, Path]] = []
    if not directory.exists():
        return []
    for path in directory.glob(f"{prefix}_*.parquet"):
        m = pattern.match(path.name)
        if m:
            files.append((int(m.group(1)), path))
    return [p for _, p in sorted(files)]

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
DEFAULT_CONFIG_PATH = APP_ROOT / "dataset_config.yaml"

STAGE_SOURCES: dict[str, tuple[str, ...]] = {
    "1a": ("primevul",),
    "1b": ("primevul", "diversevul"),
    "1c": ("primevul", "diversevul", "bigvul", "cvefixes", "secvuleval"),
}

DATASET_NAMES = ("primevul", "diversevul", "bigvul", "cvefixes", "secvuleval")

DEFAULT_CORE_COLUMNS = [
    "id",
    "code",
    "label",
    "split",
    "source_dataset",
    "commit_hash",
    "commit_date",
    "file_path",
    "func_name",
    "project",
]

NORMALIZER_EXTRA_KEYS = (
    "primevul_mapping",
    "diversevul_mapping",
    "bigvul_mapping",
    "bigvul",
    "cvefixes",
    "secvuleval",
)

PIPELINE_SCRIPT_BLOCKS = {
    "1_normalizer": "normalizer",
    "2_extractor": "extractor",
    "3_cleaner": "cleaning",
    "4_deduplicator": "dedup",
    "5_temporal_splitter": "splits",
    "6_builder": "builder",
    "7_validator": "validator",
    "8_data_balancer": "balancer",
    "9_batcher": "batcher",
}


def _build_training_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    if "training_dataset" in raw:
        return dict(raw["training_dataset"])

    cfg = dict(raw.get("training_shared", {}))

    run_pipeline = raw.get("run_pipeline", {})
    if "stage" in run_pipeline:
        cfg["stage"] = run_pipeline["stage"]

    for script_key, legacy_key in PIPELINE_SCRIPT_BLOCKS.items():
        block = raw.get(script_key)
        if not block:
            continue
        if script_key == "1_normalizer":
            cfg["normalizer"] = {
                key: value for key, value in block.items() if key not in NORMALIZER_EXTRA_KEYS
            }
            for key in NORMALIZER_EXTRA_KEYS:
                if key in block:
                    cfg[key] = block[key]
        else:
            cfg[legacy_key] = dict(block)

    embedder_block = raw.get("10_embedder")
    if embedder_block:
        cfg["10_embedder"] = dict(embedder_block)

    download_block = raw.get("download_datasets")
    if download_block:
        cfg["download_datasets"] = dict(download_block)

    return cfg


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = _build_training_cfg(raw)

    base = APP_ROOT / cfg.get("base_dir", ".")
    cfg["_app_root"] = APP_ROOT
    cfg["_base_dir"] = base.resolve()
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    rel = cfg["paths"][key]
    return (cfg["_base_dir"] / rel).resolve()


def null_placeholders(cfg: dict[str, Any]) -> set[str]:
    return set(cfg.get("null_placeholders", ["None", "null", ""]))


def mapping_for_dataset(cfg: dict[str, Any], source_dataset: str) -> dict[str, Any]:
    key = f"{source_dataset}_mapping"
    if key not in cfg:
        raise KeyError(f"No {key} in global_config.yaml")
    return cfg[key]


def sources_enabled(cfg: dict[str, Any]) -> dict[str, bool]:
    return cfg.get("sources", {})


def is_source_enabled(cfg: dict[str, Any], source_dataset: str, stage: str) -> bool:
    if source_dataset not in STAGE_SOURCES.get(stage, ()):
        return False
    return bool(sources_enabled(cfg).get(source_dataset, False))


def active_sources(cfg: dict[str, Any], stage: str) -> list[str]:
    return [n for n in DATASET_NAMES if is_source_enabled(cfg, n, stage)]


def normalizer_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("normalizer", {})


def normalized_output_dir(cfg: dict[str, Any]) -> Path:
    rel = normalizer_cfg(cfg).get("output_dir", "data/interim/normalized")
    return cfg["_base_dir"] / rel


def core_columns(cfg: dict[str, Any]) -> list[str]:
    return list(cfg.get("core_columns", DEFAULT_CORE_COLUMNS))


def normalized_chunk_prefix(cfg: dict[str, Any]) -> str:
    return str(normalizer_cfg(cfg).get("output_prefix", "normalized"))


def normalized_chunk_paths(cfg: dict[str, Any]) -> list[Path]:
    return list_chunk_files(normalized_output_dir(cfg), normalized_chunk_prefix(cfg))


def extracted_chunk_prefix(cfg: dict[str, Any]) -> str:
    return str(extractor_cfg(cfg).get("output_prefix", "extracted"))


def extracted_chunk_paths(cfg: dict[str, Any]) -> list[Path]:
    return list_chunk_files(extracted_output_dir(cfg), extracted_chunk_prefix(cfg))


def cleaned_chunk_prefix(cfg: dict[str, Any]) -> str:
    return str(cfg.get("cleaning", {}).get("output_prefix", "cleaned"))


def cleaned_chunk_paths(cfg: dict[str, Any]) -> list[Path]:
    return list_chunk_files(cleaned_output_dir(cfg), cleaned_chunk_prefix(cfg))


def deduped_chunk_prefix(cfg: dict[str, Any]) -> str:
    return str(dedup_cfg(cfg).get("output_prefix", "deduped"))


def deduped_chunk_paths(cfg: dict[str, Any]) -> list[Path]:
    return list_chunk_files(deduped_output_dir(cfg), deduped_chunk_prefix(cfg))


def split_chunk_prefix(cfg: dict[str, Any]) -> str:
    return str(splits_cfg(cfg).get("output_prefix", "split"))


def split_chunk_paths(cfg: dict[str, Any]) -> list[Path]:
    return list_chunk_files(split_output_dir(cfg), split_chunk_prefix(cfg))


def extractor_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("extractor", {})


def extracted_output_dir(cfg: dict[str, Any]) -> Path:
    rel = extractor_cfg(cfg).get("output_dir", "data/interim/extracted")
    return cfg["_base_dir"] / rel


def cleaned_output_dir(cfg: dict[str, Any]) -> Path:
    rel = cfg.get("cleaning", {}).get("output_dir", "data/interim/cleaned")
    return cfg["_base_dir"] / rel


def dedup_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("dedup", {})


def deduped_output_dir(cfg: dict[str, Any]) -> Path:
    rel = dedup_cfg(cfg).get("output_dir", "data/interim/deduped")
    return cfg["_base_dir"] / rel


def splits_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("splits", {})


def split_output_dir(cfg: dict[str, Any]) -> Path:
    rel = splits_cfg(cfg).get("output_dir", "data/interim/split")
    return cfg["_base_dir"] / rel


def split_parquet_path(cfg: dict[str, Any], dataset: str) -> Path:
    return split_output_dir(cfg) / f"{dataset}.parquet"


def builder_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("builder", {})


def processed_output_dir(cfg: dict[str, Any]) -> Path:
    """Root processed folder: data/processed/"""
    rel = cfg.get("paths", {}).get("processed_dir", "data/processed")
    return cfg["_base_dir"] / rel


def processed_whole_dir(cfg: dict[str, Any]) -> Path:
    """Whole-table parquets from builder: data/processed/whole/"""
    bcfg = builder_cfg(cfg)
    whole = str(bcfg.get("whole_output_dir", "whole"))
    return processed_output_dir(cfg) / whole


def processed_shards_dir(cfg: dict[str, Any], split_name: str) -> Path:
    """Batcher shard output per split: data/processed/train/, valid/, test/"""
    bcfg = batcher_cfg(cfg)
    subdirs = bcfg.get("shard_subdirs", {})
    sub = str(subdirs.get(split_name, split_name))
    return processed_output_dir(cfg) / sub


def validator_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("validator", {})


def balancer_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("balancer", {})


def batcher_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("batcher", {})


def builder_output_path(cfg: dict[str, Any], split_name: str) -> Path:
    """e.g. data/processed/whole/train.parquet"""
    bcfg = builder_cfg(cfg)
    filename = bcfg.get(f"{split_name}_file", f"{split_name}.parquet")
    return processed_whole_dir(cfg) / filename


def windowed_output_path(cfg: dict[str, Any], split_name: str) -> Path:
    """e.g. data/processed/whole/train_windowed.parquet (9_batcher output)."""
    bcfg = batcher_cfg(cfg)
    suffix = str(bcfg.get("windowed_suffix", "windowed"))
    return processed_whole_dir(cfg) / f"{split_name}_{suffix}.parquet"


def batch_shard_path(cfg: dict[str, Any], split_name: str, shard_idx: int) -> Path:
    """e.g. data/processed/train/train_batch_1.parquet"""
    bcfg = batcher_cfg(cfg)
    prefix = str(bcfg.get("shard_prefix", "batch"))
    return processed_shards_dir(cfg, split_name) / f"{split_name}_{prefix}_{shard_idx}.parquet"


def list_batch_shards(cfg: dict[str, Any], split_name: str) -> list[Path]:
    """Sorted shard parquets under data/processed/{split}/."""
    directory = processed_shards_dir(cfg, split_name)
    bcfg = batcher_cfg(cfg)
    prefix = str(bcfg.get("shard_prefix", "batch"))
    pattern = re.compile(rf"^{re.escape(split_name)}_{re.escape(prefix)}_(\d+)\.parquet$")
    files: list[tuple[int, Path]] = []
    if not directory.exists():
        return []
    for path in directory.glob(f"{split_name}_{prefix}_*.parquet"):
        m = pattern.match(path.name)
        if m:
            files.append((int(m.group(1)), path))
    return [p for _, p in sorted(files)]


def embedder_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("10_embedder", cfg.get("embedder", {}))


def embeddings_output_root(cfg: dict[str, Any]) -> Path:
    ecfg = embedder_cfg(cfg)
    rel = str(ecfg.get("embeddings_root", "01_Data_Processing/data/embeddings"))
    path = Path(rel)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def embedding_split_dir(cfg: dict[str, Any], split_name: str, *, pool: str) -> Path:
    return embeddings_output_root(cfg) / pool / split_name


def embedding_window_split_path(cfg: dict[str, Any], split_name: str, *, pool: str) -> Path:
    return embedding_split_dir(cfg, split_name, pool=pool) / f"{split_name}_window_embeddings.parquet"
