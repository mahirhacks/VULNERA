from __future__ import annotations

import re
import urllib.error
import urllib.request
from typing import Any

from services.hardware_service import (
    estimate_required_memory_gb,
    get_hardware_profile,
    rate_model_fit,
)

HF_FAMILIES = [
    {"id": "all", "label": "All families", "search": ""},
    {"id": "qwen", "label": "Qwen", "search": "qwen"},
    {"id": "llama", "label": "Llama", "search": "llama"},
    {"id": "mistral", "label": "Mistral", "search": "mistral"},
    {"id": "phi", "label": "Phi", "search": "phi"},
    {"id": "gemma", "label": "Gemma", "search": "gemma"},
    {"id": "codellama", "label": "Code Llama", "search": "codellama"},
    {"id": "deepseek", "label": "DeepSeek", "search": "deepseek"},
    {"id": "starcoder", "label": "StarCoder", "search": "starcoder"},
]

PARAM_SIZES = [
    {"id": "any", "label": "Any size"},
    {"id": "1b", "label": "~1B", "min": 0.5, "max": 2.0},
    {"id": "3b", "label": "~3B", "min": 2.0, "max": 4.5},
    {"id": "7b", "label": "~7B", "min": 4.5, "max": 9.0},
    {"id": "8b", "label": "~8B", "min": 7.0, "max": 10.0},
    {"id": "13b", "label": "~13B", "min": 10.0, "max": 16.0},
    {"id": "32b", "label": "~32B+", "min": 20.0, "max": 200.0},
]

PURPOSES = [
    {"id": "code", "label": "Code / instruct"},
    {"id": "general", "label": "General text"},
]

RUNTIME_QUANTS = [
    {
        "id": "q4_nf4",
        "label": "Q4 NF4",
        "short_label": "Q4",
        "description": "4-bit NF4 at load time — default for 8 GB GPUs.",
        "bytes_per_param": 0.5,
        "default": True,
    },
    {
        "id": "q8",
        "label": "Q8",
        "short_label": "Q8",
        "description": "8-bit at load time — better quality, roughly 2× the VRAM of Q4.",
        "bytes_per_param": 1.0,
    },
    {
        "id": "fp16",
        "label": "FP16",
        "short_label": "FP16",
        "description": "Full half-precision weights — highest quality, largest footprint.",
        "bytes_per_param": 2.0,
    },
]

TIER_RANK = {
    "high_speed": 5,
    "comfortable": 4,
    "acceptable": 3,
    "tight": 2,
    "not_recommended": 1,
}


def runtime_quant_options() -> list[dict[str, Any]]:
    return [dict(item) for item in RUNTIME_QUANTS]


def model_runtime_variants(
    param_b: float | None,
    hardware: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for quant in RUNTIME_QUANTS:
        fit = rate_model_fit(
            param_b=param_b,
            bytes_per_param=float(quant["bytes_per_param"]),
            hardware=hardware,
        )
        variants.append(
            {
                **quant,
                "needed_gb": estimate_required_memory_gb(param_b, float(quant["bytes_per_param"])),
                "fit": fit,
            }
        )
    return variants


def best_variant_for_model(
    param_b: float | None,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    variants = model_runtime_variants(param_b, hardware)
    if not variants:
        return None
    ranked = sorted(
        variants,
        key=lambda item: TIER_RANK.get(item["fit"]["tier"], 0),
        reverse=True,
    )
    for variant in ranked:
        if variant["fit"]["tier"] != "not_recommended":
            return variant
    return ranked[-1]


def enrich_model_with_variants(
    model: dict[str, Any],
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    param_b = model.get("param_b")
    variants = model_runtime_variants(param_b, hardware)
    best = best_variant_for_model(param_b, hardware)
    enriched = dict(model)
    enriched["variants"] = variants
    enriched["recommended_variant_id"] = best["id"] if best else "q4_nf4"
    enriched["fit"] = best["fit"] if best else rate_model_fit(
        param_b=param_b,
        bytes_per_param=0.5,
        hardware=hardware,
    )
    return enriched


def folder_name_from_repo(repo_id: str) -> str:
    name = repo_id.split("/")[-1].strip()
    name = re.sub(r"[^\w.\-]+", "-", name).strip("-").lower()
    return name or "model"


def model_loader_info(repo_id: str) -> dict[str, Any]:
    lower = repo_id.lower()
    if "gguf" in lower or re.search(r"[-_]q\d+_k", lower):
        return {
            "format": "gguf",
            "compatible": False,
            "label": "GGUF",
            "note": (
                "GGUF checkpoints are not supported by VULNERA's Transformers + "
                "BitsAndBytes loader. Pick a standard HuggingFace transformers repo instead."
            ),
        }
    if any(token in lower for token in ("-gptq", "-awq", "-bnb-4bit", "w4a16")):
        return {
            "format": "prequant",
            "compatible": False,
            "label": "Pre-quantized",
            "note": (
                "This repo looks pre-quantized for a different runtime. "
                "Use a base instruct checkpoint (e.g. Qwen/Qwen2.5-Coder-7B-Instruct)."
            ),
        }
    return {
        "format": "transformers",
        "compatible": True,
        "label": "Transformers",
        "note": "",
    }


def extract_param_b(model_id: str) -> float | None:
    text = model_id.replace("_", "-")

    active_moe = re.search(r"-A(\d+(?:\.\d+)?)[bB](?:\b|-|_|$)", text, re.I)
    if active_moe:
        return float(active_moe.group(1))

    active_suffix = re.search(r"(\d+(?:\.\d+)?)[bB]-active", text, re.I)
    if active_suffix:
        return float(active_suffix.group(1))

    matches = re.findall(r"(\d+(?:\.\d+)?)[bB]", text)
    if not matches:
        return None
    try:
        values = [float(value) for value in matches]
    except ValueError:
        return None
    if len(values) >= 2:
        return min(values)
    return values[0]


def _matches_param_size(model_id: str, param_size: str) -> bool:
    if param_size in {"", "any"}:
        return True
    spec = next((item for item in PARAM_SIZES if item["id"] == param_size), None)
    if spec is None:
        return True
    params_b = extract_param_b(model_id)
    if params_b is None:
        return False
    return float(spec["min"]) <= params_b <= float(spec["max"])


def check_huggingface_connectivity(timeout: float = 5.0) -> bool:
    request = urllib.request.Request(
        "https://huggingface.co/api/models?limit=1",
        headers={"User-Agent": "VULNERA/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def hf_filter_options() -> dict[str, Any]:
    return {
        "families": HF_FAMILIES,
        "param_sizes": [{"id": item["id"], "label": item["label"]} for item in PARAM_SIZES],
        "purposes": PURPOSES,
    }


def _build_search_query(*, query: str, family: str, purpose: str) -> str:
    terms: list[str] = []
    if purpose == "code":
        terms.append("instruct coder")
    if query.strip():
        terms.append(query.strip())
    if family and family != "all":
        family_entry = next((item for item in HF_FAMILIES if item["id"] == family), None)
        if family_entry and family_entry["search"]:
            terms.append(str(family_entry["search"]))
    return " ".join(terms).strip() or "text-generation instruct"


def _serialize_model(model: Any) -> dict[str, Any]:
    repo_id = str(getattr(model, "modelId", "") or getattr(model, "id", ""))
    param_b = extract_param_b(repo_id)
    tags = list(getattr(model, "tags", None) or [])[:6]
    loader = model_loader_info(repo_id)
    return {
        "repo_id": repo_id,
        "label": repo_id.split("/")[-1] if "/" in repo_id else repo_id,
        "folder_name": folder_name_from_repo(repo_id),
        "downloads": int(getattr(model, "downloads", 0) or 0),
        "likes": int(getattr(model, "likes", 0) or 0),
        "tags": tags,
        "gated": bool(getattr(model, "gated", False)),
        "param_b": param_b,
        "param_label": f"{param_b:g}B" if param_b is not None else "—",
        "loader": loader,
    }


def search_hf_models(
    *,
    query: str = "",
    family: str = "all",
    param_size: str = "any",
    purpose: str = "code",
    page: int = 0,
    limit: int = 20,
    include_fit: bool = False,
) -> dict[str, Any]:
    if not check_huggingface_connectivity():
        raise ConnectionError("No internet connection to Hugging Face.")

    from huggingface_hub import HfApi

    api = HfApi()
    search = _build_search_query(query=query, family=family, purpose=purpose)
    fetch_limit = min(max(limit * 5, 60), 150)

    raw_models = list(
        api.list_models(
            search=search,
            task="text-generation",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=fetch_limit,
        )
    )

    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model in raw_models:
        repo_id = str(getattr(model, "modelId", "") or "")
        if not repo_id or "/" not in repo_id or repo_id in seen:
            continue
        if not _matches_param_size(repo_id, param_size):
            continue
        seen.add(repo_id)
        filtered.append(_serialize_model(model))

    start = max(0, page) * limit
    end = start + limit
    page_models = filtered[start:end]
    hardware = get_hardware_profile() if include_fit else None
    if include_fit:
        page_models = [enrich_model_with_variants(model, hardware) for model in page_models]

    payload: dict[str, Any] = {
        "models": page_models,
        "total": len(filtered),
        "page": max(0, page),
        "limit": limit,
        "has_more": end < len(filtered),
        "search": search,
    }
    if include_fit:
        payload["hardware"] = hardware
        payload["runtime_quants"] = runtime_quant_options()
    return payload
