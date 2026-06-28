"""HuggingFace 4-bit local inference for Qwen2.5-Coder-7B-Instruct."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_ROOT = Path(__file__).resolve().parent
LLM_ROOT = SCRIPTS_ROOT.parent
PROJECT_ROOT = LLM_ROOT.parent
DEFAULT_CONFIG_PATH = LLM_ROOT / "llm_config.yaml"

_MODEL_CACHE: dict[str, Any] = {}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def local_model_dir(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    return resolve_path(str(cfg["model"]["local_model_dir"]))


def model_is_downloaded(config: dict[str, Any] | None = None) -> bool:
    model_dir = local_model_dir(config)
    return (model_dir / "config.json").exists() and (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "model.safetensors.index.json").exists()
        or any(model_dir.glob("*.safetensors"))
    )


def _dtype_from_name(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[str(name).lower()]


def _system_ram_budget_gb() -> float:
    try:
        import psutil

        return max(1.0, float(psutil.virtual_memory().available) / (1024**3) * 0.9)
    except Exception:
        return 16.0


def _gpu_memory_budget_gb() -> tuple[float, float]:
    """Return (free_gb, total_gb) for CUDA device 0 after empty_cache."""
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return free_bytes / (1024**3), total_bytes / (1024**3)


def _build_max_memory(reserve_gb: float = 0.75) -> dict[str | int, str] | None:
    """Build accelerate max_memory map using free VRAM + available system RAM."""
    import torch

    if not torch.cuda.is_available():
        return None

    free_gb, _total_gb = _gpu_memory_budget_gb()
    gpu_gb = max(0.5, free_gb - reserve_gb)
    cpu_gb = _system_ram_budget_gb()
    return {0: f"{int(gpu_gb)}GiB", "cpu": f"{int(cpu_gb)}GiB"}


def _bitsandbytes_config(quant_cfg: dict[str, Any], *, allow_cpu_offload: bool):
    from transformers import BitsAndBytesConfig

    common = {"llm_int8_enable_fp32_cpu_offload": allow_cpu_offload}
    if quant_cfg.get("load_in_8bit"):
        return BitsAndBytesConfig(load_in_8bit=True, **common)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=_dtype_from_name(str(quant_cfg.get("bnb_4bit_compute_dtype", "float16"))),
        bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
        **common,
    )


def _model_input_device(model) -> "torch.device":
    import torch

    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    for param in model.parameters():
        if param.device.type == "cuda":
            return param.device
    return next(model.parameters()).device


def _resolve_model_source(config: dict[str, Any]) -> str:
    model_cfg = config["model"]
    local_dir = local_model_dir(config)
    if model_is_downloaded(config):
        return str(local_dir)
    return str(model_cfg["repo_id"])


def model_is_cached(config: dict[str, Any] | None = None) -> bool:
    cfg = config or load_config()
    return str(_resolve_model_source(cfg)) in _MODEL_CACHE


def cached_model_source() -> str:
    """Return cache key/path for the model currently loaded in memory, if any."""
    if not _MODEL_CACHE:
        return ""
    return next(iter(_MODEL_CACHE.keys()), "")


def load_model_and_tokenizer(config: dict[str, Any] | None = None, *, force_reload: bool = False):
    """Load (or return cached) 4-bit quantized model + tokenizer."""
    cfg = config or load_config()
    cache_key = str(_resolve_model_source(cfg))
    if not force_reload and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Defer switching until scan time: release a different loaded model only when loading anew.
    if _MODEL_CACHE and cache_key not in _MODEL_CACHE:
        release_model_cache()

    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_cfg = cfg["model"]
    quant_cfg = cfg.get("quantization", {})
    model_source = _resolve_model_source(cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    use_quant = bool(quant_cfg.get("load_in_8bit") or quant_cfg.get("load_in_4bit", True))
    max_memory = _build_max_memory() if use_quant and torch.cuda.is_available() else None
    allow_cpu_offload = max_memory is not None and "cpu" in max_memory

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

    if quant_cfg.get("load_in_8bit"):
        model_kwargs["quantization_config"] = _bitsandbytes_config(
            quant_cfg, allow_cpu_offload=allow_cpu_offload
        )
    elif quant_cfg.get("load_in_4bit", True):
        model_kwargs["quantization_config"] = _bitsandbytes_config(
            quant_cfg, allow_cpu_offload=allow_cpu_offload
        )
        if allow_cpu_offload:
            free_gb, total_gb = _gpu_memory_budget_gb()
            print(
                f"LLM: GPU headroom {free_gb:.1f}/{total_gb:.1f} GB — "
                "CPU offload enabled for layers that do not fit on GPU (slower inference).",
                flush=True,
            )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.eval()

    bundle = {"model": model, "tokenizer": tokenizer, "config": cfg}
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def build_chat_prompt(user_prompt: str, config: dict[str, Any] | None = None) -> str:
    cfg = config or load_config()
    system_prompt = str(cfg.get("system_prompt", "")).strip()
    bundle = load_model_and_tokenizer(cfg)
    tokenizer = bundle["tokenizer"]

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def release_model_cache() -> None:
    """Drop cached Qwen weights and free GPU memory."""
    import gc

    import torch

    for cached in list(_MODEL_CACHE.values()):
        model = cached.pop("model", None)
        if model is not None:
            del model
        tokenizer = cached.pop("tokenizer", None)
        if tokenizer is not None:
            del tokenizer
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def generate_explanation(
    user_prompt: str,
    config: dict[str, Any] | None = None,
    *,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    web_scan: bool = False,
    clear_cache_after: bool | None = None,
) -> str:
    """Generate an explanation string from a status-aware XAI prompt."""
    return _generate_text(
        user_prompt,
        config,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        web_scan=web_scan,
        clear_cache_after=clear_cache_after,
    )


def _generate_text(
    user_prompt: str,
    config: dict[str, Any] | None = None,
    *,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    web_scan: bool = False,
    clear_cache_after: bool | None = None,
) -> str:
    """Low-level text generation (used by verification pass)."""
    import gc

    import torch

    cfg = config or load_config()
    gen_cfg = cfg.get("generation", {})
    web_cfg = cfg.get("web_scan", {}) if web_scan else {}
    use_web = bool(web_cfg)

    bundle = load_model_and_tokenizer(cfg)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    chat_text = build_chat_prompt(user_prompt, cfg)
    max_prompt_tokens = int(web_cfg.get("max_prompt_tokens", 3072)) if use_web else None
    tokenize_kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if max_prompt_tokens:
        tokenize_kwargs["truncation"] = True
        tokenize_kwargs["max_length"] = max_prompt_tokens

    inputs = tokenizer(chat_text, **tokenize_kwargs)
    input_device = _model_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    resolved_max_tokens = max_new_tokens
    if resolved_max_tokens is None and use_web:
        resolved_max_tokens = web_cfg.get("max_new_tokens")
    resolved_max_tokens = int(resolved_max_tokens or gen_cfg.get("max_new_tokens", 384))

    do_sample = bool(web_cfg.get("do_sample", False)) if use_web else bool(gen_cfg.get("do_sample", True))

    generation_kwargs = {
        "max_new_tokens": resolved_max_tokens,
        "temperature": float(temperature if temperature is not None else gen_cfg.get("temperature", 0.2)),
        "top_p": float(top_p if top_p is not None else gen_cfg.get("top_p", 0.9)),
        "do_sample": do_sample,
        "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.05)),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if not do_sample:
        generation_kwargs.pop("temperature", None)
        generation_kwargs.pop("top_p", None)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    del inputs
    del output_ids
    should_clear = clear_cache_after
    if should_clear is None and use_web:
        should_clear = bool(web_cfg.get("clear_cuda_cache_each_call", True))
    if should_clear:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return text


def generate_verified_explanation(
    analysis_prompt: str,
    *,
    window_code: str,
    config: dict[str, Any] | None = None,
    max_new_tokens: int | None = None,
    verification_max_tokens: int | None = None,
    web_scan: bool = False,
    clear_cache_after: bool | None = None,
) -> dict[str, Any]:
    """
    Draft explanation + verification pass (Suggestion 3).
    Returns {explanation, verified, verification_raw, draft}.
    """
    cfg = config or load_config()
    grounded_cfg = cfg.get("grounded_explanation") or {}

    draft = _generate_text(
        analysis_prompt,
        cfg,
        max_new_tokens=max_new_tokens,
        web_scan=web_scan,
        clear_cache_after=False,
    )

    if not bool(grounded_cfg.get("verification_pass", True)):
        return {"explanation": draft, "verified": None, "verification_raw": "", "draft": draft}

    ensure_xai_on_path()
    from grounded_explain import build_verification_prompt, parse_verification_response  # noqa: PLC0415

    verify_prompt = build_verification_prompt(window_code=window_code, proposed_explanation=draft)
    verify_tokens = verification_max_tokens
    if verify_tokens is None:
        verify_tokens = int(grounded_cfg.get("verification_max_tokens", 256))

    verification_raw = _generate_text(
        verify_prompt,
        cfg,
        max_new_tokens=verify_tokens,
        web_scan=web_scan,
        clear_cache_after=clear_cache_after,
    )
    parsed = parse_verification_response(verification_raw)
    return {
        "explanation": parsed["explanation"],
        "verified": parsed["verified"],
        "verification_raw": verification_raw,
        "draft": draft,
    }


def ensure_xai_on_path() -> None:
    xai_scripts = PROJECT_ROOT / "07_XAI" / "training_scripts"
    if str(xai_scripts) not in sys.path:
        sys.path.insert(0, str(xai_scripts))
