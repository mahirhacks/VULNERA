from __future__ import annotations

import shutil
import subprocess
from typing import Any

OVERHEAD_FACTOR = 1.2
RAM_HEADROOM_FACTOR = 0.9

FIT_TIERS: dict[str, dict[str, str]] = {
    "not_recommended": {
        "label": "Not recommended",
        "summary": "Combined RAM + VRAM is well below what this model needs.",
    },
    "tight": {
        "label": "Tight fit",
        "summary": "May load with heavy CPU offload; expect slow scans and possible OOM.",
    },
    "acceptable": {
        "label": "Acceptable",
        "summary": "Should run, but most weights may spill to system RAM.",
    },
    "comfortable": {
        "label": "Comfortable",
        "summary": "Dedicated GPU VRAM covers the model with modest headroom.",
    },
    "high_speed": {
        "label": "High speed",
        "summary": "Plenty of dedicated VRAM for fast GPU inference.",
    },
}


def _nvidia_smi_vram_gb() -> tuple[float, str | None]:
    if shutil.which("nvidia-smi") is None:
        return 0.0, None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0, None
        line = result.stdout.strip().splitlines()[0]
        if "," not in line:
            return 0.0, None
        name, memory_mb = [part.strip() for part in line.split(",", 1)]
        lower = name.lower()
        if "intel" in lower and "arc" not in lower:
            return 0.0, name
        return float(memory_mb) / 1024.0, name
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0, None


def _torch_dedicated_vram_gb() -> tuple[float, str | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, None
        props = torch.cuda.get_device_properties(0)
        name = str(props.name)
        lower = name.lower()
        if "intel" in lower and "arc" not in lower:
            return 0.0, name
        return float(props.total_memory) / (1024**3), name
    except Exception:
        return 0.0, None


def get_hardware_profile() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        available_ram_gb = float(vm.available) / (1024**3)
    except Exception:
        available_ram_gb = 0.0

    usable_ram_gb = round(available_ram_gb * RAM_HEADROOM_FACTOR, 2)

    gpu_vram_gb, gpu_name = _nvidia_smi_vram_gb()
    if gpu_vram_gb <= 0:
        gpu_vram_gb, gpu_name = _torch_dedicated_vram_gb()

    return {
        "available_ram_gb": round(available_ram_gb, 2),
        "usable_ram_gb": usable_ram_gb,
        "gpu_vram_gb": round(gpu_vram_gb, 2),
        "gpu_name": gpu_name,
        "has_dedicated_gpu": gpu_vram_gb > 0,
        "overhead_factor": OVERHEAD_FACTOR,
        "ram_headroom_factor": RAM_HEADROOM_FACTOR,
    }


def estimate_required_memory_gb(param_b: float | None, bytes_per_param: float) -> float | None:
    if param_b is None or param_b <= 0:
        return None
    return round(float(param_b) * float(bytes_per_param) * OVERHEAD_FACTOR, 2)


def rate_model_fit(
    *,
    param_b: float | None,
    bytes_per_param: float,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = hardware or get_hardware_profile()
    needed_gb = estimate_required_memory_gb(param_b, bytes_per_param)
    usable_ram = float(profile.get("usable_ram_gb") or 0.0)
    gpu_vram = float(profile.get("gpu_vram_gb") or 0.0)
    total_pool = usable_ram + gpu_vram

    if needed_gb is None:
        tier = "acceptable"
        detail = "Parameter size unknown — fit estimate is approximate."
    elif total_pool < needed_gb:
        tier = "not_recommended"
        detail = (
            f"Needs ~{needed_gb:g} GB effective memory; your pool is only ~{total_pool:g} GB "
            f"({usable_ram:g} GB RAM + {gpu_vram:g} GB VRAM) — it will not fit."
        )
    elif total_pool < needed_gb * 1.1:
        tier = "tight"
        detail = (
            f"Needs ~{needed_gb:g} GB; total headroom is under 10% "
            f"({total_pool:g} GB pool vs {needed_gb:g} GB required)."
        )
    elif gpu_vram >= needed_gb * 1.5:
        tier = "high_speed"
        detail = (
            f"Needs ~{needed_gb:g} GB; dedicated VRAM ({gpu_vram:g} GB) exceeds that by 50%+."
        )
    elif gpu_vram >= needed_gb * 1.1:
        tier = "comfortable"
        detail = (
            f"Needs ~{needed_gb:g} GB; dedicated VRAM ({gpu_vram:g} GB) has at least 10% headroom."
        )
    else:
        tier = "acceptable"
        detail = (
            f"Needs ~{needed_gb:g} GB; fits across RAM+VRAM ({total_pool:g} GB) "
            f"but GPU alone ({gpu_vram:g} GB) is tight."
        )

    meta = FIT_TIERS[tier]
    return {
        "tier": tier,
        "label": meta["label"],
        "summary": meta["summary"],
        "detail": detail,
        "needed_gb": needed_gb,
        "usable_ram_gb": usable_ram,
        "gpu_vram_gb": gpu_vram,
        "total_pool_gb": round(total_pool, 2),
    }
