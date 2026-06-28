"""
Download Qwen2.5-Coder-7B-Instruct into 08_LLM/models/ for offline 4-bit inference.

Usage:
    python 08_LLM/training_scripts/download_model.py
    python 08_LLM/training_scripts/download_model.py --verify-load
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from llm_common import DEFAULT_CONFIG_PATH, LLM_ROOT, load_config, local_model_dir, model_is_downloaded, resolve_path


def download_weights(config_path: Path) -> Path:
    from huggingface_hub import snapshot_download

    config = load_config(config_path)
    model_cfg = config["model"]
    repo_id = str(model_cfg["repo_id"])
    target_dir = local_model_dir(config)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
    )
    print(f"Saved model files under {target_dir}")
    return target_dir


def verify_load(config_path: Path) -> dict:
    from llm_common import generate_explanation, load_model_and_tokenizer

    config = load_config(config_path)
    load_model_and_tokenizer(config, force_reload=True)
    sample = generate_explanation(
        "Reply with exactly: VULNERA LLM ready.",
        config,
        max_new_tokens=32,
        temperature=0.0,
    )
    return {"verified": True, "sample_output": sample}


def main() -> None:
    parser = ArgumentParser(description="Download Qwen2.5-Coder-7B-Instruct into 08_LLM/models/.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--verify-load", action="store_true", help="Load 4-bit model and run a one-line smoke test.")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else LLM_ROOT / args.config
    config = load_config(config_path)
    target_dir = local_model_dir(config)

    if not model_is_downloaded(config):
        download_weights(config_path)
    else:
        print(f"Model already present at {target_dir}")

    summary = {
        "repo_id": config["model"]["repo_id"],
        "local_model_dir": str(target_dir),
        "downloaded": model_is_downloaded(config),
    }

    if args.verify_load:
        print("Running 4-bit load + generation smoke test (requires CUDA GPU)...")
        summary["smoke_test"] = verify_load(config_path)

    summary_path = resolve_path("08_LLM/artifacts/download_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
