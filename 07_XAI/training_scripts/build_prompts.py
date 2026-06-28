"""
Build status-aware LLM prompts from UI payload.

Usage:
    python 07_XAI/training_scripts/build_prompts.py
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

from xai_common import DEFAULT_CONFIG_PATH, XAI_ROOT, build_prompt, iter_jsonl, load_config, resolve_path, write_jsonl


def main() -> None:
    parser = ArgumentParser(description="Build status-aware LLM prompts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else XAI_ROOT / args.config
    config = load_config(config_path)
    cfg = config.get("build_prompts", {})

    ui_payload_path = resolve_path(str(cfg["ui_payload"]))
    sample_per_status = int(cfg.get("sample_per_status", 5))
    output_path = resolve_path(str(cfg["output_path"]))

    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(ui_payload_path):
        status = str(record["agreement_status"])
        if len(by_status[status]) < sample_per_status:
            by_status[status].append(record)

    prompt_rows: list[dict[str, Any]] = []
    for status, records in sorted(by_status.items()):
        for record in records:
            prompt_rows.append(
                {
                    "function_group_id": record["function_group_id"],
                    "agreement_status": status,
                    "label": record.get("label"),
                    "function_score_calibrated": record.get("function_score_calibrated"),
                    "function_flagged": record.get("function_flagged"),
                    "max_window_prob": record.get("max_window_prob"),
                    "window_count": record.get("window_count"),
                    "highlight_window_indices": record.get("highlight_window_indices"),
                    "status_display": record.get("status_display"),
                    "prompt": build_prompt(record),
                }
            )

    write_jsonl(output_path, prompt_rows)
    summary = {
        "prompt_count": len(prompt_rows),
        "counts": {status: len(records) for status, records in by_status.items()},
        "output_path": str(output_path),
    }
    output_path.with_name(output_path.stem + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nPrompts built")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
