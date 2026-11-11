"""Compute device inventory — backs the top-bar GPU/CPU status widget."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.compute import list_devices

router = APIRouter(prefix="/api/compute", tags=["compute"])


@router.get("")
def devices():
    return list_devices()
