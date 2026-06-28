"""

Run LLM explainer on status-aware prompts.



Providers:

  mock         — template explanation for demo/offline (default)

  huggingface  — local Qwen2.5-Coder-7B 4-bit via 08_LLM (no Ollama/server)

  openai       — calls OpenAI API when OPENAI_API_KEY is set



Usage:

    python 07_XAI/training_scripts/run_explainer.py

    python 07_XAI/training_scripts/run_explainer.py --provider huggingface

    python 08_LLM/training_scripts/run_explainer.py --limit 5

"""



from __future__ import annotations



import json

import os

import sys

from argparse import ArgumentParser

from pathlib import Path

from typing import Any



from xai_common import DEFAULT_CONFIG_PATH, PROJECT_ROOT, XAI_ROOT, iter_jsonl, load_config, resolve_path, write_jsonl





def _ensure_llm_on_path() -> None:

    llm_scripts = PROJECT_ROOT / "08_LLM" / "training_scripts"

    if str(llm_scripts) not in sys.path:

        sys.path.insert(0, str(llm_scripts))





def mock_explanation(prompt_row: dict[str, Any]) -> str:

    status = str(prompt_row["agreement_status"])

    score = float(prompt_row.get("function_score_calibrated", 0.0))

    windows = prompt_row.get("highlight_window_indices") or []

    window_hint = f" Window(s) {windows}." if windows else ""

    if status == "agree_positive":

        return (

            f"The function-level calibrated risk is {score:.1%} and at least one window also crossed the window threshold.{window_hint} "

            "Review the highlighted memory, bounds, or input-handling operations in the flagged window(s)."

        )

    if status == "review_suggested":

        return (

            f"Although the pooled function score is only {score:.1%}, an individual window exceeded the window detector threshold.{window_hint} "

            "Inspect that window in isolation — localized unsafe logic may be diluted by max-pooling across the full function."

        )

    if status == "diffuse_risk":

        return (

            f"The function-level score is elevated ({score:.1%}) but no single window crossed the window threshold.{window_hint} "

            "Risk may be distributed across max-pool contributors rather than concentrated in one segment."

        )

    return f"Models agree this function is not flagged ({score:.1%} calibrated risk). No immediate review required."





def huggingface_explanation(prompt_row: dict[str, Any], *, llm_config_path: Path | None = None) -> str:

    _ensure_llm_on_path()

    from llm_common import DEFAULT_CONFIG_PATH as LLM_DEFAULT_CONFIG  # noqa: PLC0415

    from llm_common import generate_explanation, load_config as load_llm_config  # noqa: PLC0415



    config_path = llm_config_path or LLM_DEFAULT_CONFIG

    llm_config = load_llm_config(config_path)

    return generate_explanation(str(prompt_row["prompt"]), llm_config)





def openai_explanation(prompt_row: dict[str, Any], *, model: str, max_tokens: int, temperature: float) -> str:

    try:

        from openai import OpenAI  # noqa: PLC0415

    except ImportError as exc:

        raise ImportError("Install openai package: pip install openai") from exc



    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.responses.create(

        model=model,

        input=str(prompt_row["prompt"]),

        max_output_tokens=max_tokens,

        temperature=temperature,

    )

    return response.output_text.strip()





def main() -> None:

    parser = ArgumentParser(description="Run LLM explainer on prompts.")

    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    parser.add_argument("--provider", type=str, default=None, choices=["mock", "huggingface", "openai"])

    parser.add_argument("--limit", type=int, default=None, help="Max prompts to explain.")

    parser.add_argument("--llm-config", type=Path, default=None, help="Path to 08_LLM/llm_config.yaml")

    args = parser.parse_args()



    config_path = args.config if args.config.is_absolute() else XAI_ROOT / args.config

    config = load_config(config_path)

    cfg = config.get("run_explainer", {})



    provider = args.provider or str(cfg.get("provider", "mock"))

    prompts_path = resolve_path(str(cfg["prompts_path"]))

    output_path = resolve_path(str(cfg["output_path"]))

    model = str(cfg.get("model", "gpt-4o-mini"))

    max_tokens = int(cfg.get("max_tokens", 500))

    temperature = float(cfg.get("temperature", 0.2))

    limit = args.limit if args.limit is not None else cfg.get("limit")



    llm_config_path = args.llm_config

    if llm_config_path is None and cfg.get("llm_config"):

        llm_config_path = resolve_path(str(cfg["llm_config"]))



    rows: list[dict[str, Any]] = []

    for index, prompt_row in enumerate(iter_jsonl(prompts_path)):

        if limit is not None and index >= int(limit):

            break



        if provider == "huggingface":

            explanation = huggingface_explanation(prompt_row, llm_config_path=llm_config_path)

            model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

        elif provider == "openai":

            if not os.environ.get("OPENAI_API_KEY"):

                raise EnvironmentError("OPENAI_API_KEY is not set.")

            explanation = openai_explanation(

                prompt_row,

                model=model,

                max_tokens=max_tokens,

                temperature=temperature,

            )

            model_name = model

        else:

            explanation = mock_explanation(prompt_row)

            model_name = "mock"



        rows.append(

            {

                "function_group_id": prompt_row["function_group_id"],

                "agreement_status": prompt_row["agreement_status"],

                "label": prompt_row.get("label"),

                "function_score_calibrated": prompt_row.get("function_score_calibrated"),

                "highlight_window_indices": prompt_row.get("highlight_window_indices"),

                "provider": provider,

                "model": model_name,

                "explanation": explanation,

            }

        )



    write_jsonl(output_path, rows)

    summary = {

        "provider": provider,

        "model": model_name if rows else model,

        "explanation_count": len(rows),

        "output_path": str(output_path),

    }

    output_path.with_name(output_path.stem + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nExplanations generated")

    print(json.dumps(summary, indent=2))





if __name__ == "__main__":

    main()


