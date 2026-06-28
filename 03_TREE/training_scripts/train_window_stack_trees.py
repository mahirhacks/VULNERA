"""
Train the four tree architectures on window-level embeddings.

Each row is one token window; labels are propagated from the parent function.
Models are saved under each model's output_dir in subdir ``window/final``.

Usage:
    python 03_TREE/training_scripts/train_window_stack_trees.py
    python 03_TREE/training_scripts/train_window_stack_trees.py --smoke-test
"""

from __future__ import annotations

import importlib.util
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent
TREE_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = TREE_ROOT.parent
DEFAULT_CONFIG = TREE_ROOT / "tree_config.yaml"
WINDOW_STACK_SUBDIR = "window/final"

_spec = importlib.util.spec_from_file_location("train_trees", SCRIPTS_ROOT / "train_trees.py")
_train_trees = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train_trees)


def main() -> None:
    parser = ArgumentParser(description="Train four tree models on window embeddings.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pool", type=str, default=None, choices=["mean", "max", "attention"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["xgboost", "lightgbm", "random_forest", "extra_trees"],
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else TREE_ROOT / args.config
    config = _train_trees.load_config(config_path)

    settings = config.get("train_window_trees", {})
    pool = args.pool or str(settings.get("default_pool", "max"))
    label_column = str(settings.get("label_column", "label"))
    embedding_column = str(settings.get("embedding_column", "embedding"))
    window_id_column = str(settings.get("window_id_column", "window_id"))
    function_group_column = str(settings.get("function_group_column", "function_group_id"))
    embeddings_root = _train_trees.embeddings_root_from_config(config)

    train_path = _train_trees.resolve_path(
        _train_trees.resolve_window_embedding_path("train", pool, embeddings_root)
    )
    valid_path = _train_trees.resolve_path(
        _train_trees.resolve_window_embedding_path("valid", pool, embeddings_root)
    )
    test_path = _train_trees.resolve_path(
        _train_trees.resolve_window_embedding_path("test", pool, embeddings_root)
    )
    logs_dir = _train_trees.resolve_path(settings.get("output_dir", "03_TREE/results"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, _ = _train_trees.load_window_split(
        train_path,
        label_column=label_column,
        embedding_column=embedding_column,
        window_id_column=window_id_column,
        function_group_column=function_group_column,
    )
    x_valid, y_valid, _ = _train_trees.load_window_split(
        valid_path,
        label_column=label_column,
        embedding_column=embedding_column,
        window_id_column=window_id_column,
        function_group_column=function_group_column,
    )
    x_test, y_test, _ = _train_trees.load_window_split(
        test_path,
        label_column=label_column,
        embedding_column=embedding_column,
        window_id_column=window_id_column,
        function_group_column=function_group_column,
    )

    if args.smoke_test:
        x_train, y_train = x_train[:5000], y_train[:5000]
        x_valid, y_valid = x_valid[:1000], y_valid[:1000]
        x_test, y_test = x_test[:1000], y_test[:1000]

    scale_pos_weight = _train_trees.auto_scale_pos_weight(y_train)
    xgb_cfg = config.get("xgboost", {})
    lgbm_cfg = config.get("lightgbm", {})
    xgb_threshold = float(xgb_cfg.get("decision_threshold", 0.5))
    lgbm_threshold = float(lgbm_cfg.get("decision_threshold", 0.5))
    rf_threshold = float(config.get("random_forest", {}).get("decision_threshold", 0.5))
    et_threshold = float(config.get("extra_trees", {}).get("decision_threshold", 0.5))

    print(f"\nWindow-stack tree training [{pool}-pool]")
    print(f"Train windows: {x_train.shape[0]:,} x {x_train.shape[1]} dims")
    print(f"Valid windows: {x_valid.shape[0]:,}")
    print(f"Test windows:  {x_test.shape[0]:,}")
    print(f"scale_pos_weight: {scale_pos_weight:.3f}")

    model_entries = config.get("train_trees", {}).get("models", [])
    if args.only:
        model_entries = [entry for entry in model_entries if str(entry.get("name")) == args.only]
    builders_by_name: dict[str, tuple[Any, bool]] = {
        "xgboost": (_train_trees.build_xgb(config.get("xgboost", {}), scale_pos_weight), True),
        "lightgbm": (_train_trees.build_lgbm(config.get("lightgbm", {}), scale_pos_weight), True),
        "random_forest": (_train_trees.build_random_forest(config.get("random_forest", {})), False),
        "extra_trees": (_train_trees.build_extra_trees(config.get("extra_trees", {})), False),
    }

    results: list[dict[str, Any]] = []
    for entry in model_entries:
        if not entry.get("enabled", True):
            continue
        name = str(entry["name"])
        if name not in builders_by_name:
            continue
        builder, early_stop = builders_by_name[name]
        base_dir = _train_trees.resolve_path(str(entry["output_dir"]))
        if name == "xgboost":
            threshold = xgb_threshold
        elif name == "lightgbm":
            threshold = lgbm_threshold
        elif name == "random_forest":
            threshold = rf_threshold
        else:
            threshold = et_threshold

        print(f"\n--- {name} (window stack) ---")
        result = _train_trees.train_and_evaluate(
            name,
            builder,
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test,
            base_dir,
            trained_subdir=WINDOW_STACK_SUBDIR,
            use_early_stopping=early_stop,
            decision_threshold=threshold,
            early_stopping_rounds=int(lgbm_cfg.get("early_stopping_rounds", 50)),
        )
        results.append(result)
        print(
            f"Valid F1: {result['valid']['f1']:.4f} | "
            f"Test F1: {result['test']['f1']:.4f} | "
            f"Test AUC: {result['test']['roc_auc']:.4f}"
        )

    summary = {
        "granularity": "window",
        "pool_method": pool,
        "trained_subdir": WINDOW_STACK_SUBDIR,
        "train_windows": int(x_train.shape[0]),
        "valid_windows": int(x_valid.shape[0]),
        "test_windows": int(x_test.shape[0]),
        "scale_pos_weight": scale_pos_weight,
        "models": results,
    }
    summary_path = logs_dir / f"window_stack_tree_summary_{pool}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
