"""
Run local HuggingFace explainer on XAI prompts (no Ollama / no HTTP server).

Usage:
    python 08_LLM/training_scripts/download_model.py
    python 08_LLM/training_scripts/run_explainer.py
    python 08_LLM/training_scripts/run_explainer.py --limit 3
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from llm_common import DEFAULT_CONFIG_PATH, LLM_ROOT, ensure_xai_on_path, generate_explanation, load_config, resolve_path

ensure_xai_on_path()
from xai_common import iter_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    parser = ArgumentParser(description="Run local Qwen2.5-Coder explainer on XAI prompts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Max prompts to explain (overrides config).")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else LLM_ROOT / args.config
    config = load_config(config_path)
    cfg = config.get("run_explainer", {})

    prompts_path = resolve_path(str(cfg["prompts_path"]))
    output_path = resolve_path(str(cfg["output_path"]))
    limit = args.limit if args.limit is not None else cfg.get("limit")

    rows: list[dict[str, Any]] = []
    for index, prompt_row in enumerate(iter_jsonl(prompts_path)):
        if limit is not None and index >= int(limit):
            break

        explanation = generate_explanation(str(prompt_row["prompt"]), config)
        row: dict[str, Any] = {
            "function_group_id": prompt_row["function_group_id"],
            "agreement_status": prompt_row["agreement_status"],
            "label": prompt_row.get("label"),
            "function_score_calibrated": prompt_row.get("function_score_calibrated"),
            "highlight_window_indices": prompt_row.get("highlight_window_indices"),
            "provider": "huggingface",
            "model": config["model"]["repo_id"],
            "local_model_dir": str(resolve_path(config["model"]["local_model_dir"])),
            "explanation": explanation,
        }
        grounded_cfg = config.get("grounded_explanation") or {}
        if bool(grounded_cfg.get("enabled", True)) and prompt_row.get("prompt_windows"):
            from grounded_explain import build_grounded_window_prompt  # noqa: PLC0415

            window_index = int((prompt_row.get("highlight_window_indices") or [0])[0])
            record = dict(prompt_row)
            g_prompt, g_ctx = build_grounded_window_prompt(
                record,
                window_index,
                chain_of_thought=bool(grounded_cfg.get("chain_of_thought", True)),
            )
            if bool(grounded_cfg.get("verification_pass", True)):
                from llm_common import generate_verified_explanation  # noqa: PLC0415

                verified = generate_verified_explanation(
                    g_prompt,
                    window_code=str(g_ctx.get("window_code") or ""),
                    config=config,
                )
                row["explanation"] = verified["explanation"]
                row["explanation_grounding"] = {
                    "detected_cwe": g_ctx.get("detected_cwe"),
                    "top_tokens": g_ctx.get("top_tokens"),
                    "verified": verified.get("verified"),
                }
            else:
                row["explanation"] = generate_explanation(g_prompt, config)
                row["explanation_grounding"] = {
                    "detected_cwe": g_ctx.get("detected_cwe"),
                    "top_tokens": g_ctx.get("top_tokens"),
                }
        rows.append(row)
        print(f"[{index + 1}] {prompt_row['function_group_id']} ({prompt_row['agreement_status']})")

    write_jsonl(output_path, rows)
    summary = {
        "provider": "huggingface",
        "model": config["model"]["repo_id"],
        "explanation_count": len(rows),
        "output_path": str(output_path),
    }
    output_path.with_name(output_path.stem + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nExplanations generated")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
