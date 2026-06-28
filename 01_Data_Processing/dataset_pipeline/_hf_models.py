"""Resolve HuggingFace model sources for local checkpoints vs hub ids."""

from __future__ import annotations

from pathlib import Path

_WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors", "model.bin")
_TOKENIZER_FILE = "tokenizer_config.json"
_CONFIG_FILE = "config.json"
_HUB_DEFAULT = "microsoft/graphcodebert-base"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_filesystem_path(value: str | Path) -> bool:
    text = str(value)
    path = Path(text)
    if path.is_absolute():
        return True
    if len(text) > 1 and text[1] == ":":
        return True
    return "\\" in text


def is_local_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / _CONFIG_FILE).is_file():
        return False
    return any((path / name).is_file() for name in _WEIGHT_FILES)


def has_local_tokenizer(path: Path) -> bool:
    return path.is_dir() and (path / _TOKENIZER_FILE).is_file()


def resolve_pretrained_source(
    configured: str | Path,
    *,
    root_dir: Path | None = None,
    require_weights: bool = True,
) -> str:
    """Return a path or hub id suitable for ``from_pretrained``."""
    root = root_dir or project_root()
    fallbacks: list[str | Path] = [
        configured,
        "02_ML_Model/graphcodebert-base",
    ]
    seen: set[str] = set()

    for item in fallbacks:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)

        path = Path(item)
        if not path.is_absolute():
            path = (root / path).resolve()
        if require_weights and is_local_model_dir(path):
            return str(path)
        if not require_weights and has_local_tokenizer(path):
            return str(path)

    return _HUB_DEFAULT


def load_pretrained_kwargs(source: str) -> dict[str, bool]:
    path = Path(source)
    if path.is_dir() and (path / _CONFIG_FILE).is_file():
        return {"local_files_only": True}
    return {}
