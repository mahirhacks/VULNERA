"""
Run a split through final base tree models and build a stacking table.

Output columns: xgb, lightgbm, random_forest, extra_trees, label

Usage:
    python 04_META/training_scripts/build_meta_table.py
    python 04_META/training_scripts/build_meta_table.py --split valid
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from meta_common import META_ROOT, build_split_table, load_config, resolve_path


def main() -> None:
    parser = ArgumentParser(description="Build meta-learner input table from base tree predictions.")
    parser.add_argument("--config", type=Path, default=META_ROOT / "meta_config.yaml")
    parser.add_argument("--split", type=str, default=None, choices=["train", "valid", "test"])
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else META_ROOT / args.config
    config = load_config(config_path)
    settings = config.get("build_meta_table", {})
    build_settings = {**config.get("train_meta", {}), **settings}

    split = args.split or str(settings.get("split", "valid"))
    feature_columns = [str(col) for col in build_settings.get("feature_columns", ["xgb", "lightgbm"])]
    label_column = str(build_settings.get("label_column", "label"))
    output_path = resolve_path(str(settings.get("output_path")))

    if split != "valid":
        output_path = output_path.with_name(f"meta_{split}_predictions.parquet")

    table = build_split_table(build_settings, split, feature_columns, label_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)

    print(f"\nMeta table [{split}-pool | {build_settings.get('pool', 'max')}]")
    print(f"Rows: {len(table):,}")
    print(f"Columns: {', '.join(feature_columns + [label_column])}")
    print(f"Positive rate: {table[label_column].mean():.4f}")
    for col in feature_columns:
        print(f"{col} mean prob: {table[col].mean():.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
