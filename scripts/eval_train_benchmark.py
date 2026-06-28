"""One-off: score train split for VULNERA + Devign HF baselines (print only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
AGG_SCRIPTS = PROJECT_ROOT / "06_AGGREGATOR" / "training_scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(AGG_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AGG_SCRIPTS))

import numpy as np
import torch

from benchmark_transfer import BASELINES, binary_metrics, best_f1_threshold, load_split_frame, predict_probs

# VULNERA train scoring (reuse aggregation stack)
from run_aggregation import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    AGGREGATOR_ROOT,
    function_meta_from_window_frame,
    function_scores_from_windows,
    load_config,
    load_window_frame,
    load_window_stack_models,
    resolve_path,
    score_all_windows,
)
from aggregator_common import import_meta_common, import_train_trees  # noqa: E402


def vulnera_train_metrics() -> dict[str, float]:
    config = load_config(DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.is_absolute() else AGGREGATOR_ROOT / DEFAULT_CONFIG_PATH)
    cfg = config.get("run_aggregator", {})
    meta_common = import_meta_common()
    train_trees = import_train_trees()
    meta_config = meta_common.load_config(resolve_path(str(cfg["meta_config"])))
    stack = load_window_stack_models(cfg, meta_config.get("train_meta", {}))
    deployment_threshold = float(
        json.loads(resolve_path(str(cfg["calibrated_deployment"])).read_text(encoding="utf-8"))[
            "deployment_threshold_calibrated"
        ]
    )
    window_frame = score_all_windows(
        load_window_frame(cfg, "train", str(cfg.get("window_pool", cfg.get("pool", "max")))),
        stack=stack,
    )
    function_meta = function_meta_from_window_frame(window_frame)
    calibrated_scores = function_scores_from_windows(
        window_frame,
        function_meta,
        threshold=deployment_threshold,
        weight=float(cfg.get("spread_weight", 0.25)),
    )
    labels = function_meta["label"].astype(int).to_numpy()
    metrics = train_trees.compute_metrics(labels, calibrated_scores, threshold=deployment_threshold)
    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "threshold": deployment_threshold,
        "n": int(len(labels)),
    }


def main() -> None:
    train_path = PROJECT_ROOT / "01_Data_Processing" / "data" / "processed" / "whole" / "train.parquet"
    train = load_split_frame(train_path, max_samples=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_thresholds = {"codebert-devign": 0.22, "graphcodebert-devign": 0.38}

    print("=== TRAIN SPLIT (1999-2019, n=72,260 functions) ===\n", flush=True)

    print("Scoring VULNERA on train windows ...", flush=True)
    v = vulnera_train_metrics()
    print(
        f"VULNERA @ tau={v['threshold']:.2f}: "
        f"P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['n']:,})\n",
        flush=True,
    )

    for bid in ("codebert-devign", "graphcodebert-devign"):
        spec = BASELINES[bid]
        print(f"Scoring {spec['display']} ...", flush=True)
        probs = predict_probs(
            model_id=spec["model"],
            tokenizer_id=spec["tokenizer"],
            codes=train.codes,
            device=device,
            batch_size=16,
        )
        t_valid = valid_thresholds[bid]
        at_valid = binary_metrics(train.labels, probs, threshold=t_valid)
        t_train, _ = best_f1_threshold(train.labels, probs)
        at_train = binary_metrics(train.labels, probs, threshold=t_train)
        print(spec["display"])
        print(
            f"  @ valid-tuned t={t_valid:.2f}: "
            f"P={at_valid['precision']:.3f} R={at_valid['recall']:.3f} F1={at_valid['f1']:.3f}"
        )
        print(
            f"  @ train max-F1 t={t_train:.2f}: "
            f"P={at_train['precision']:.3f} R={at_train['recall']:.3f} F1={at_train['f1']:.3f}\n"
        )


if __name__ == "__main__":
    main()
