"""Load vulnera.yaml and resolve templated paths for the pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "vulnera.yaml"


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or DEFAULT_MANIFEST
    with manifest_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def path_values(manifest: dict[str, Any]) -> dict[str, str]:
    paths = dict(manifest.get("paths", {}))
    return {key: str(value) for key, value in paths.items()}


def format_args(args: list[str], manifest: dict[str, Any]) -> list[str]:
    values = path_values(manifest)
    pipeline = manifest.get("pipeline", {})
    values["dataset_stage"] = str(pipeline.get("dataset_stage", "1a"))
    return [arg.format(**values) for arg in args]


def resolve_script(layer_key: str, script_rel: str, manifest: dict[str, Any]) -> Path:
    layer = manifest.get("layers", {}).get(layer_key, {})
    layer_dir = layer.get("dir")
    if not layer_dir:
        raise KeyError(f"Unknown layer {layer_key!r}")
    return PROJECT_ROOT / layer_dir / script_rel


def stage_by_id(manifest: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in manifest.get("pipeline", {}).get("stages", []):
        if stage.get("id") == stage_id:
            return stage
    raise KeyError(f"Unknown pipeline stage {stage_id!r}")


def expand_stage_list(manifest: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    pipeline = manifest.get("pipeline", {})
    bundle = set(pipeline.get("train_bundle", []))
    stages: list[dict[str, Any]] = []
    for name in names:
        if name == "train":
            for stage_id in pipeline.get("train_bundle", []):
                stages.append(stage_by_id(manifest, stage_id))
            continue
        if name in bundle:
            stages.append(stage_by_id(manifest, name))
            continue
        stages.append(stage_by_id(manifest, name))
    return stages
