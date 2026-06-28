"""Apply window_stack_meta_logistic_tune.json into meta_config.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

META_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = META_ROOT / "meta_config.yaml"
SUMMARY_PATH = META_ROOT / "results" / "window_stack" / "window_stack_meta_logistic_tune.json"


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Run tune_meta_logistic.py --config meta_config.yaml first: {SUMMARY_PATH}")

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
    if "solver" in params:
        section["solver"] = str(params["solver"])
    config.setdefault("train_meta", {})["logistic_regression"] = section
    config["train_meta"].setdefault("decision_thresholds", {})["logistic"] = round(threshold, 2)
    cal = config.setdefault("calibrate_threshold", {})
    cal["deployment_threshold"] = round(threshold, 2)

    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(
        f"window-stack logistic meta: valid F1={tuned['valid']['f1']:.4f} "
        f"P={tuned['valid']['precision']:.4f} R={tuned['valid']['recall']:.4f} | "
        f"test F1={tuned['test']['f1']:.4f} threshold={threshold:.2f} "
        f"meets_policy={tuned.get('meets_policy')}"
    )
    print(f"Updated {CONFIG_PATH}")


if __name__ == "__main__":
    main()
