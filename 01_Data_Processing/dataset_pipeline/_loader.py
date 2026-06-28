"""Load numbered pipeline steps and ``0_config`` (filenames cannot be normal imports)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_STEPS_DIR = Path(__file__).resolve().parent


def load_step(stem: str) -> ModuleType:
    """Load e.g. ``1_normalizer`` from ``dataset_pipeline/1_normalizer.py``."""
    path = _STEPS_DIR / f"{stem}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = f"dataset_pipeline.{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pipeline step: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _CfgModule:
    """Lazy access to ``0_config.py``: ``from dataset_pipeline._loader import cfg``."""

    _mod: ModuleType | None = None

    def __getattr__(self, name: str) -> Any:
        if _CfgModule._mod is None:
            _CfgModule._mod = load_step("0_config")
        return getattr(_CfgModule._mod, name)


cfg = _CfgModule()
