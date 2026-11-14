"""
Which hardware onnxruntime runs a model on, decided in one place.

Every model in the catalog runs through onnxruntime, so the GPU question is
"which onnxruntime execution providers are installed", not "is torch built
with CUDA". Before this module, GPUs were found through torch only: a machine
with an NVIDIA card and onnxruntime-gpu (no torch) showed "CPU only" and ran
on the processor, and AMD / Intel GPUs (DirectML) were never used.

Order tried on a GPU device: TensorRT (FP16, engine cached on disk) ->
CUDA -> DirectML -> CPU. onnxruntime falls back down the list by itself
when a provider cannot take a node, so the list is safe on any machine.
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

GPU_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider")


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def gpu_providers() -> list[str]:
    have = set(available_providers())
    return [p for p in GPU_PROVIDERS if p in have]


@lru_cache(maxsize=1)
def nvidia_gpus() -> tuple[dict, ...]:
    """NVIDIA cards as the driver reports them (nvidia-smi ships with the driver).

    Works without torch. Free memory is read once per process; it is only used
    to pick the emptiest card and to show the card in the device menu.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return ()
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "total_vram_mb": int(float(parts[2])), "free_vram_mb": int(float(parts[3]))})
    return tuple(gpus)


def gpu_status() -> dict:
    """What the Settings panel shows: the card, whether Syntra can use it, and the fix."""
    cards = list(nvidia_gpus())
    provs = gpu_providers()
    nvidia_ready = any(p in provs for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider"))
    status = {"cards": cards, "providers": provs, "usable": bool(provs), "fix": None}
    if cards and not nvidia_ready:
        status["fix"] = ("An NVIDIA card is installed but onnxruntime has no CUDA support. In the Syntra "
                         "environment run: pip uninstall -y onnxruntime && pip install \"onnxruntime-gpu[cuda,cudnn]\"")
    return status


def session_providers(device: str, trt_cache: Path | None = None) -> list:
    """Providers for an InferenceSession on `device` ("cpu", "cuda:0", "gpu:1", "dml:0")."""
    if not device or device == "cpu":
        return ["CPUExecutionProvider"]
    idx = int(device.split(":")[1]) if ":" in device and device.split(":")[1].isdigit() else 0
    have = set(available_providers())
    chosen: list = []
    if "TensorrtExecutionProvider" in have:
        opts = {"device_id": idx, "trt_fp16_enable": True}
        if trt_cache is not None:
            trt_cache.mkdir(parents=True, exist_ok=True)
            opts.update(trt_engine_cache_enable=True, trt_engine_cache_path=str(trt_cache))
        chosen.append(("TensorrtExecutionProvider", opts))
    if "CUDAExecutionProvider" in have:
        chosen.append(("CUDAExecutionProvider", {"device_id": idx}))
    if "DmlExecutionProvider" in have:
        chosen.append(("DmlExecutionProvider", {"device_id": idx}))
    chosen.append("CPUExecutionProvider")
    return chosen


def trt_cache_dir() -> Path:
    from app.core.config import settings

    return settings.DATA_DIR / "trt_cache"
