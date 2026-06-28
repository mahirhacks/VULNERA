from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from services.hf_hub_service import folder_name_from_repo
from services.paths import LLM_CONFIG_PATH, LLM_SCRIPTS_PATH, PROJECT_ROOT

MAX_QUICK_PRESETS = 3

MODEL_PRESETS = [
    {
        "id": "qwen2.5-coder-7b",
        "label": "Qwen2.5-Coder-7B-Instruct",
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "folder_name": "qwen2.5-coder-7b-instruct",
        "note": "Default — best quality; needs ~8GB VRAM with 4-bit.",
    },
    {
        "id": "qwen2.5-coder-3b",
        "label": "Qwen2.5-Coder-3B-Instruct",
        "repo_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "folder_name": "qwen2.5-coder-3b-instruct",
        "note": "Smaller footprint for limited GPU memory.",
    },
    {
        "id": "qwen2.5-coder-1.5b",
        "label": "Qwen2.5-Coder-1.5B-Instruct",
        "repo_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "folder_name": "qwen2.5-coder-1.5b-instruct",
        "note": "Lightweight; faster explanations, lower quality.",
    },
]

DEFAULT_MODELS_ROOT = "08_LLM/models"

QUANT_PROFILES: dict[str, dict[str, Any]] = {
    "q4_nf4": {
        "profile_id": "q4_nf4",
        "load_in_4bit": True,
        "load_in_8bit": False,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
    },
    "q8": {
        "profile_id": "q8",
        "load_in_4bit": False,
        "load_in_8bit": True,
    },
    "fp16": {
        "profile_id": "fp16",
        "load_in_4bit": False,
        "load_in_8bit": False,
    },
}


def normalize_quick_preset(entry: dict[str, Any]) -> dict[str, Any]:
    repo_id = str(entry.get("repo_id", "")).strip()
    if not repo_id or "/" not in repo_id:
        raise ValueError("Each quick preset needs a valid repo_id (e.g. Qwen/Qwen2.5-Coder-7B-Instruct).")

    folder_name = str(entry.get("folder_name") or folder_name_from_repo(repo_id)).strip()
    label = str(entry.get("label") or repo_id.split("/")[-1]).strip()
    preset_id = str(entry.get("id") or folder_name).strip()
    note = str(entry.get("note") or "").strip()
    return {
        "id": preset_id,
        "label": label,
        "repo_id": repo_id,
        "folder_name": folder_name,
        "note": note,
    }


def get_quick_presets(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = config or load_llm_config()
    raw = cfg.get("quick_presets")
    if not isinstance(raw, list) or not raw:
        return [dict(preset) for preset in MODEL_PRESETS[:MAX_QUICK_PRESETS]]
    return [normalize_quick_preset(item) for item in raw[:MAX_QUICK_PRESETS]]


def update_quick_presets(presets: list[dict[str, Any]]) -> dict[str, Any]:
    if len(presets) > MAX_QUICK_PRESETS:
        raise ValueError(f"At most {MAX_QUICK_PRESETS} quick presets are allowed.")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in presets:
        preset = normalize_quick_preset(item)
        if preset["id"] in seen_ids:
            raise ValueError(f"Duplicate quick preset id: {preset['id']}")
        seen_ids.add(preset["id"])
        normalized.append(preset)

    config = load_llm_config()
    config["quick_presets"] = normalized
    save_llm_config(config)
    return llm_settings_payload(config)


def apply_quantization_profile(config: dict[str, Any], quantization_id: str | None) -> dict[str, Any]:
    profile_id = (quantization_id or "q4_nf4").strip()
    profile = dict(QUANT_PROFILES.get(profile_id, QUANT_PROFILES["q4_nf4"]))
    config["quantization"] = profile
    return config


def _ensure_llm_scripts_on_path() -> None:
    scripts = str(LLM_SCRIPTS_PATH)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def load_llm_config() -> dict[str, Any]:
    with LLM_CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    with LLM_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
    return config


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value.strip())
    return path if path.is_absolute() else PROJECT_ROOT / path


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def _has_model_weights(model_dir: Path) -> bool:
    if not (model_dir / "config.json").is_file():
        return False
    return (
        (model_dir / "model.safetensors").is_file()
        or (model_dir / "model.safetensors.index.json").is_file()
        or (model_dir / "pytorch_model.bin").is_file()
        or any(model_dir.glob("*.safetensors"))
    )


def is_downloaded_model_dir(model_dir: Path) -> bool:
    return model_dir.is_dir() and _has_model_weights(model_dir)


def infer_repo_id(model_dir: Path, *, fallback: str = "") -> str:
    cfg = load_llm_config()
    for preset in get_quick_presets(cfg):
        if model_dir.name == preset["folder_name"]:
            return preset["repo_id"]

    for preset in MODEL_PRESETS:
        if model_dir.name == preset["folder_name"]:
            return preset["repo_id"]

    for name in ("adapter_config.json", "tokenizer_config.json"):
        meta_path = model_dir / name
        if not meta_path.is_file():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("base_model_name_or_path", "_name_or_path"):
            value = str(payload.get(key, "")).strip()
            if value and "/" in value:
                return value

    return fallback


def _preset_by_folder(folder_name: str, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    for preset in get_quick_presets(config):
        if preset["folder_name"] == folder_name:
            return preset
    for preset in MODEL_PRESETS:
        if preset["folder_name"] == folder_name:
            return preset
    return None


def get_preset(preset_id: str) -> dict[str, Any]:
    cfg = load_llm_config()
    for preset in get_quick_presets(cfg):
        if preset["id"] == preset_id:
            return dict(preset)
    for preset in MODEL_PRESETS:
        if preset["id"] == preset_id:
            return dict(preset)
    raise ValueError(f"Unknown model preset: {preset_id}")


def _models_root_from_config(cfg: dict[str, Any]) -> str:
    model_cfg = dict(cfg.get("model") or {})
    explicit = str(model_cfg.get("models_root_dir", "")).strip()
    if explicit:
        return explicit

    local_dir = str(model_cfg.get("local_model_dir", "")).strip()
    if local_dir:
        return _relative_to_project(_resolve_path(local_dir).parent)

    return DEFAULT_MODELS_ROOT


def scan_models_directory(models_root_dir: str) -> list[dict[str, Any]]:
    root = _resolve_path(models_root_dir)
    if not root.is_dir():
        return []

    cfg = load_llm_config()
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append_model(model_dir: Path) -> None:
        if not is_downloaded_model_dir(model_dir):
            return
        rel_path = _relative_to_project(model_dir)
        if rel_path in seen:
            return
        seen.add(rel_path)
        preset = _preset_by_folder(model_dir.name, cfg)
        discovered.append(
            {
                "id": model_dir.name,
                "label": preset["label"] if preset else model_dir.name,
                "relative_path": rel_path,
                "folder_name": model_dir.name,
                "downloaded": True,
                "preset_id": preset["id"] if preset else None,
            }
        )

    if is_downloaded_model_dir(root):
        append_model(root)

    try:
        children = sorted(root.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        children = []

    for child in children:
        if child.is_dir() and not child.name.startswith("."):
            append_model(child)

    discovered.sort(key=lambda item: item["label"].lower())
    return discovered


def _cached_model_path() -> str:
    try:
        _ensure_llm_scripts_on_path()
        from llm_common import cached_model_source  # noqa: PLC0415

        return cached_model_source() or ""
    except Exception:
        return ""


def _preset_status(models_root_dir: str, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = config or load_llm_config()
    root = _resolve_path(models_root_dir)
    enriched: list[dict[str, Any]] = []
    for preset in get_quick_presets(cfg):
        candidate = root / preset["folder_name"]
        downloaded = is_downloaded_model_dir(candidate) if root.is_dir() else False
        enriched.append(
            {
                **preset,
                "downloaded": downloaded,
                "relative_path": _relative_to_project(candidate),
                "target_path": str(candidate),
            }
        )
    return enriched


def llm_model_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_llm_config()
    model_cfg = dict(cfg.get("model") or {})
    models_root_dir = _models_root_from_config(cfg)
    local_dir = str(model_cfg.get("local_model_dir", "")).strip()
    resolved = _resolve_path(local_dir) if local_dir else None
    cached_path = _cached_model_path()

    downloaded = is_downloaded_model_dir(resolved) if resolved is not None else False
    selected_id = resolved.name if resolved is not None and downloaded else ""

    return {
        "models_root_dir": models_root_dir,
        "models_root_resolved": str(_resolve_path(models_root_dir)),
        "selected_model_id": selected_id,
        "local_model_dir": local_dir,
        "resolved_path": str(resolved) if resolved is not None else "",
        "downloaded": downloaded,
        "cached_in_memory": bool(cached_path),
        "loaded_model_path": cached_path,
        "config_path": str(LLM_CONFIG_PATH),
    }


def update_llm_model(
    *,
    models_root_dir: str,
    selected_model_id: str,
    quantization_id: str | None = None,
) -> dict[str, Any]:
    models_root_dir = models_root_dir.strip()
    selected_model_id = selected_model_id.strip()
    if not models_root_dir:
        raise ValueError("models_root_dir is required")
    if not selected_model_id:
        raise ValueError("selected_model is required")

    root = _resolve_path(models_root_dir)
    if not root.is_dir():
        raise ValueError(f"Models directory not found: {root}")

    available = scan_models_directory(models_root_dir)
    match = next((item for item in available if item["id"] == selected_model_id), None)
    if match is None:
        raise ValueError(
            f"Model '{selected_model_id}' was not found under {root}. "
            "Scan the directory and pick a downloaded model folder.",
        )

    local_model_dir = match["relative_path"]
    model_path = _resolve_path(local_model_dir)
    config = load_llm_config()
    preset = _preset_by_folder(model_path.name, config)
    repo_id = infer_repo_id(
        model_path,
        fallback=str((config.get("model") or {}).get("repo_id", "")),
    )
    if preset:
        repo_id = preset["repo_id"]

    model_cfg = dict(config.get("model") or {})
    model_cfg["models_root_dir"] = models_root_dir.replace("\\", "/")
    model_cfg["local_model_dir"] = local_model_dir.replace("\\", "/")
    model_cfg["repo_id"] = repo_id
    config["model"] = model_cfg
    apply_quantization_profile(config, quantization_id)
    save_llm_config(config)
    return llm_settings_payload(config)


def delete_llm_model(*, models_root_dir: str, model_id: str) -> dict[str, Any]:
    import shutil

    from services.download_job_store import download_is_running

    models_root_dir = models_root_dir.strip()
    model_id = model_id.strip()
    if not models_root_dir:
        raise ValueError("models_root_dir is required")
    if not model_id:
        raise ValueError("model_id is required")
    if download_is_running():
        raise ValueError("A model download is in progress. Wait for it to finish before deleting.")

    root = _resolve_path(models_root_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Models directory not found: {root}")

    available = scan_models_directory(models_root_dir)
    match = next((item for item in available if item["id"] == model_id), None)
    if match is None:
        raise ValueError(f"Model '{model_id}' was not found under {root}.")

    model_path = _resolve_path(match["relative_path"]).resolve()
    if model_path == root:
        raise ValueError("Cannot delete the models root directory.")
    if not _path_is_relative_to(model_path, root):
        raise ValueError("Model path is outside the configured models directory.")
    if not is_downloaded_model_dir(model_path):
        raise ValueError(f"'{model_id}' is not a valid downloaded model folder.")

    try:
        _ensure_llm_scripts_on_path()
        from llm_common import cached_model_source, release_model_cache  # noqa: PLC0415

        cached = cached_model_source()
        if cached and Path(cached).resolve() == model_path:
            release_model_cache()
    except Exception:
        pass

    shutil.rmtree(model_path)

    config = load_llm_config()
    model_cfg = dict(config.get("model") or {})
    local_dir = str(model_cfg.get("local_model_dir", "")).strip()
    was_active = bool(local_dir) and _resolve_path(local_dir).resolve() == model_path

    remaining = scan_models_directory(models_root_dir)
    if was_active and remaining:
        return update_llm_model(
            models_root_dir=models_root_dir,
            selected_model_id=remaining[0]["id"],
        )

    if was_active:
        model_cfg["local_model_dir"] = ""
        config["model"] = model_cfg
        save_llm_config(config)

    return llm_settings_payload(config)


def llm_settings_payload(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_llm_config()
    models_root_dir = _models_root_from_config(cfg)
    return {
        "model": llm_model_status(cfg),
        "available_models": scan_models_directory(models_root_dir),
        "presets": _preset_status(models_root_dir, cfg),
        "quick_presets": get_quick_presets(cfg),
        "max_quick_presets": MAX_QUICK_PRESETS,
    }
