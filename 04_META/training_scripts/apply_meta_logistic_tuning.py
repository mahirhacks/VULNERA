"""Apply meta_logistic_tune.json into meta_config.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

META_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = META_ROOT / "meta_config.yaml"
SUMMARY_PATH = META_ROOT / "results" / "meta_logistic_tune.json"


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Run tune_meta_logistic.py first: missing {SUMMARY_PATH}")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    tuned = summary["tuned"]
    params = dict(tuned["params"])
    threshold = float(tuned["threshold"])

    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    section = dict(config.get("train_meta", {}).get("logistic_regression", {}))
    section["C"] = float(params["C"])
    section["class_weight"] = params["class_weight"]
    section["max_iter"] = int(params["max_iter"])
    config.setdefault("train_meta", {})["logistic_regression"] = section
    config["train_meta"].setdefault("decision_thresholds", {})["logistic"] = round(threshold, 2)

    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(
        f"logistic meta: valid F1={tuned['valid']['f1']:.4f} test F1={tuned['test']['f1']:.4f} "
        f"threshold={threshold:.2f} C={params['C']}"
    )
    print(f"Updated {CONFIG_PATH}")


if __name__ == "__main__":
    main()
