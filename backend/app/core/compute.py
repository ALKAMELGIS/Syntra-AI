"""
Compute device manager — Phase 2 of the roadmap.

Exposes four execution options to the rest of the engine and to the API:

    CPU0        a configurable CPU profile (see Settings.CPU0_THREADS)
    CPU1        a second configurable CPU profile (defaults to all cores)
    GPU         CUDA, validated against the installed torch/onnxruntime build
    AUTO        picks the best available device for the current model +
                workload, and transparently falls back when a device is
                unavailable

Torch is imported lazily so this module (and anything that only needs device
*selection*, not actual tensor execution) stays importable in environments
that don't have torch installed yet — e.g. this build sandbox, or a thin
API-only process.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.core.config import settings


class DeviceOption(str, Enum):
    CPU0 = "cpu0"
    CPU1 = "cpu1"
    GPU = "gpu"
    AUTO = "auto"


@dataclass
class DeviceProfile:
    option: DeviceOption
    resolved: str          # the concrete device string handed to the model runtime, e.g. "cuda:0" / "cpu"
    threads: int | None    # CPU thread count, when applicable
    gpu_index: int | None  # CUDA device index, when applicable
    reason: str            # human-readable explanation, surfaced to the UI


def _cuda_devices() -> list[dict]:
    """Return available CUDA devices with free/total VRAM, or [] if none / torch missing."""
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    devices = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        devices.append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "free_vram_mb": free // (1024 * 1024),
                "total_vram_mb": total // (1024 * 1024),
            }
        )
    return devices


def list_devices() -> dict:
    """Inventory used by the /api/compute endpoint and the dashboard's device badge."""
    gpus = _cuda_devices()
    return {
        "cpu0": {"threads": settings.CPU0_THREADS},
        "cpu1": {"threads": settings.CPU1_THREADS},
        "gpu": gpus,
        "gpu_count": len(gpus),
    }


def resolve_device(
    option: DeviceOption,
    *,
    model_precision_supports_fp16: bool = True,
    estimated_vram_mb: int | None = None,
    allowed_gpu_indices: list[int] | None = None,
) -> DeviceProfile:
    """
    Resolve a requested execution option into a concrete device profile.

    AUTO SELECT evaluates, in order: hardware availability -> model/precision
    compatibility -> available memory / GPU VRAM -> processing workload, and
    falls back to the next-best option rather than failing outright.

    `allowed_gpu_indices` backs the UI's per-GPU checkboxes (the top-bar GPU
    dropdown): when given, only those physical CUDA device indices are ever
    considered a candidate — a real restriction, not cosmetic. An empty list
    is treated the same as None (no restriction) rather than "no GPU allowed",
    since an accidentally-empty selection shouldn't silently force CPU.
    """
    gpus = _cuda_devices()
    if allowed_gpu_indices:
        gpus = [g for g in gpus if g["index"] in allowed_gpu_indices]

    if option == DeviceOption.CPU0:
        return DeviceProfile(option, "cpu", settings.CPU0_THREADS, None, "User-selected CPU profile 0.")

    if option == DeviceOption.CPU1:
        return DeviceProfile(option, "cpu", settings.CPU1_THREADS, None, "User-selected CPU profile 1.")

    if option == DeviceOption.GPU:
        if not gpus:
            reason = (
                "GPU requested but no CUDA device is available on this host — fell back to CPU1."
                if not _cuda_devices() else
                "GPU requested but none of the enabled GPUs are available — fell back to CPU1."
            )
            return DeviceProfile(DeviceOption.CPU1, "cpu", settings.CPU1_THREADS, None, reason)
        best = max(gpus, key=lambda g: g["free_vram_mb"])
        return DeviceProfile(DeviceOption.GPU, f"cuda:{best['index']}", None, best["index"],
                              f"Using {best['name']} ({best['free_vram_mb']} MB free).")

    # AUTO SELECT
    if gpus:
        candidates = gpus
        if estimated_vram_mb is not None:
            candidates = [g for g in gpus if g["free_vram_mb"] >= estimated_vram_mb] or gpus
        best = max(candidates, key=lambda g: g["free_vram_mb"])
        reason = f"AUTO selected GPU {best['index']} ({best['name']}, {best['free_vram_mb']} MB free)."
        if estimated_vram_mb is not None and best["free_vram_mb"] < estimated_vram_mb:
            reason += " Warning: estimated workload may exceed free VRAM; expect possible OOM or slow fallback."
        return DeviceProfile(DeviceOption.GPU, f"cuda:{best['index']}", None, best["index"], reason)

    return DeviceProfile(
        DeviceOption.CPU1, "cpu", settings.CPU1_THREADS, None,
        "AUTO selected CPU1 — no CUDA device detected on this host.",
    )
