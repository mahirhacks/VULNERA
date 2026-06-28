"""
Build webapp-ready UI payload from disagreement artifacts + window code.

Usage:
    python 07_XAI/training_scripts/build_ui_payload.py
    python 07_XAI/training_scripts/build_ui_payload.py --all
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pandas as pd

from xai_common import (
    DEFAULT_CONFIG_PATH,
    XAI_ROOT,
    attach_window_code,
    load_config,
    reconstruct_window_lists,
    resolve_path,
    write_jsonl,
)


def load_disagreement_records(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    parquet_path = resolve_path(str(cfg["aggregation_parquet"]))
    frame = pd.read_parquet(parquet_path)
    return frame.to_dict(orient="records")


def load_window_lookup(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    predictions_path = resolve_path(
        str(cfg.get("window_predictions", "06_AGGREGATOR/artifacts/test_window_predictions.parquet"))
    )
    frame = pd.read_parquet(predictions_path)
    return {str(row["window_id"]): row.to_dict() for _, row in frame.iterrows()}


def main() -> None:
    parser = ArgumentParser(description="Build UI payload for disagreement surfacing.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--all", action="store_true", help="Export all functions (default: sample per status).")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else XAI_ROOT / args.config
    config = load_config(config_path)
    cfg = config.get("build_ui_payload", {})

    code_index_path = resolve_path(str(cfg["code_index"]))
    if not code_index_path.exists():
        raise FileNotFoundError(
            f"Code index not found: {code_index_path}. Run build_code_index.py first."
        )

    code_frame = pd.read_parquet(code_index_path)
    code_index = {str(row["window_id"]): row.to_dict() for _, row in code_frame.iterrows()}
    window_lookup = load_window_lookup(cfg)

    records = load_disagreement_records(cfg)
    sample_per_status = int(cfg.get("sample_per_status", 25))

    selected: list[dict[str, Any]] = []
    if args.all:
        selected = records
    else:
        by_status: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_status.setdefault(str(record["agreement_status"]), []).append(record)
        for group in by_status.values():
            selected.extend(group[:sample_per_status])

    payload_rows = [
        attach_window_code(reconstruct_window_lists(record, window_lookup), code_index) for record in selected
    ]
    output_path = resolve_path(str(cfg["output_path"]))
    write_jsonl(output_path, payload_rows)

    summary = {
        "split": cfg.get("split", "test"),
        "rows_exported": len(payload_rows),
        "all_functions": args.all,
        "counts": {
            status: sum(1 for row in payload_rows if row["agreement_status"] == status)
            for status in sorted({str(row["agreement_status"]) for row in payload_rows})
        },
        "output_path": str(output_path),
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nUI payload built")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
