from __future__ import annotations

from typing import Any

import yaml

from services.paths import CONFIG_PATH


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_config(config: dict[str, Any]) -> dict[str, Any]:
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
    return config
