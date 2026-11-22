"""Compute device inventory — backs the top-bar GPU/CPU status widget."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.compute import list_devices

router = APIRouter(prefix="/api/compute", tags=["compute"])


@router.get("")
def devices():
    return list_devices()


@router.get("/hosting")
def hosting():
    """Where this engine runs, for Train > Setup.

    space_id   the Hugging Face Space it runs in (Spaces set SPACE_ID), else null
    zero_gpu   the Space is on ZeroGPU: a GPU lent for short calls (60 s by default,
               5 min a day on a free account), too little to train; Spaces set SPACES_ZERO_GPU
    train_gpu  a GPU this engine can train on (CUDA, and not ZeroGPU)
    """
    import os

    inv = list_devices()
    space = os.getenv("SPACE_ID") or None
    zero = bool(os.getenv("SPACES_ZERO_GPU"))
    gpus = inv.get("gpu") or []
    return {
        "space_id": space,
        "space_host": os.getenv("SPACE_HOST") or None,
        "zero_gpu": zero,
        "gpu_name": gpus[0].get("name") if gpus else None,
        "train_gpu": bool(gpus) and not zero,
    }
