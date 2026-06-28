"""
Build window_id -> code lookup index from processed batch shards.

Usage:
    python 07_XAI/training_scripts/build_code_index.py
    python 07_XAI/training_scripts/build_code_index.py --split test
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from xai_common import DEFAULT_CONFIG_PATH, XAI_ROOT, load_config, resolve_path


def main() -> None:
    parser = ArgumentParser(description="Build window code index for XAI/webapp.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", type=str, default=None, choices=["train", "valid", "test"])
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else XAI_ROOT / args.config
    config = load_config(config_path)
    cfg = config.get("build_code_index", {})

    split = args.split or str(cfg.get("split", "test"))
    processed_dir = resolve_path(str(cfg.get("processed_dir")))
    split_dir = processed_dir / split
    output_path = resolve_path(str(cfg.get("output_path")))

    shard_paths = sorted(split_dir.glob(f"{split}_batch_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No batch shards found in {split_dir}")

    columns = ["id", "code", "function_group_id", "window_index", "label", "cwe", "cve"]
    frames = [pd.read_parquet(path, columns=columns) for path in shard_paths]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.rename(columns={"id": "window_id"})
    frame["window_id"] = frame["window_id"].astype(str)
    frame["function_group_id"] = frame["function_group_id"].astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    print(f"\nWindow code index [{split}]")
    print(f"Rows: {len(frame):,}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
