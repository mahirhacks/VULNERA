"""Apply window-stack tune JSONs from 03_TREE/results/window_stack_*_tune_max.json into tree_config.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from tune_window_stack_common import MODEL_CHOICES, tune_summary_path

TREE_ROOT = Path(__file__).resolve().parent.parent
RESULTS = TREE_ROOT / "results"
CONFIG_PATH = TREE_ROOT / "tree_config.yaml"

def resolve_summary_path(model: str) -> Path:
    return RESULTS / tune_summary_path(model)


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for model in MODEL_CHOICES:
        path = resolve_summary_path(model)
        if not path.exists():
            raise FileNotFoundError(f"Missing tune output: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        tuned = summary["tuned"]
        params = dict(tuned["params"])
        threshold = float(tuned["threshold"])

        section = dict(config.get(model, {}))
        for key, value in params.items():
            section[key] = value
        section["decision_threshold"] = round(threshold, 2)
        config[model] = section

        test_f1 = tuned["test"]["f1"]
        valid_f1 = tuned["valid"]["f1"]
        print(
            f"{model}: valid F1={valid_f1:.4f} test F1={test_f1:.4f} "
            f"threshold={threshold:.2f}  ({path.name})"
        )

    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"Updated {CONFIG_PATH}")


if __name__ == "__main__":
    main()
