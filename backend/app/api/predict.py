"""
Run-prediction endpoint.

Kicks off a detection run for one model against one uploaded image, tracks
progress in the persistent job store (app.core.jobs), and reports NOT_PRODUCTION_READY models honestly instead of
pretending to run them.
"""
from __future__ import annotations


import re
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.core.compute import DeviceOption
from app.core.config import settings
from app.core.jobs import get_store
from app.core.image_library import find_entry_by_image_id
from app.core.security import confine_path
from app.models.base import ModelStatus
from app.models.registry import get_metadata
from app.models.stats import record_run

router = APIRouter(prefix="/api/predict", tags=["predict"])

KIND = "predict"


class PredictRequest(BaseModel):
    # Preferred: the library id returned by POST /api/images. The server maps
    # it to a file inside UPLOAD_DIR, so clients never send raw paths.
    image_id: str | None = None
    # Legacy: a path, accepted only if it resolves inside an allowed data root.
    image_path: str | None = None
    model_id: str
    device: DeviceOption = DeviceOption.AUTO
    confidence_threshold: float | None = None
    tile_size: int = settings.DEFAULT_TILE_SIZE
    overlap: float = settings.DEFAULT_TILE_OVERLAP
    gpu_indices: list[int] | None = None  # restricts AUTO/GPU to these physical CUDA device indices


def _run_job(job_id: str, req: PredictRequest):
    jobs = get_store()
    jobs.update(job_id, status="processing")
    try:
        from app.engine.pipeline import DetectionEngine
        from app.engine.raster_io import iter_tiles
        from app.models.factory import build_model

        model = build_model(req.model_id)
        engine = DetectionEngine(model)

        def progress(done, total):
            jobs.update(job_id, tiles_processed=done, tiles_total=total)

        tiles = iter_tiles(req.image_path, tile_size=req.tile_size, overlap=req.overlap)
        out_dir = settings.RESULTS_DIR / job_id
        result = engine.run(
            tiles, out_dir, device_option=req.device,
            confidence_threshold=req.confidence_threshold, progress_cb=progress,
            allowed_gpu_indices=req.gpu_indices,
        )
        jobs.update(job_id, status="completed", result=result.__dict__)
        record_run(
            settings.DATA_DIR, req.model_id, status="completed",
            tiles_processed=result.tiles_processed, detection_count=result.final_detection_count,
            duration_seconds=result.duration_seconds, device=result.device_used,
        )
    except Exception as exc:
        jobs.update(job_id, status="error", error=str(exc))
        record_run(settings.DATA_DIR, req.model_id, status="error")


def _display_name(path: str) -> str:
    # Uploads are stored as "<uuid>_<original name>"; show the original.
    return re.sub(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_", "", Path(path).name)


def _resolve_image_path(req: PredictRequest) -> str:
    if req.image_id:
        entry = find_entry_by_image_id(settings.UPLOAD_DIR, req.image_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"No library image with id '{req.image_id}'.")
        return str(confine_path(entry["path"], field="image_id", must_exist=True))
    if req.image_path:
        return str(confine_path(req.image_path, field="image_path", must_exist=True))
    raise HTTPException(status_code=400, detail="Provide image_id (preferred) or image_path.")


@router.post("")
def start_prediction(req: PredictRequest, background_tasks: BackgroundTasks):
    meta = get_metadata(req.model_id)
    if meta.status == ModelStatus.NOT_PRODUCTION_READY:
        raise HTTPException(
            status_code=409,
            detail=(
                f"'{meta.name}' is not production-ready yet (no validated weights/benchmark). "
                f"Refusing to generate fake detections. Details: {meta.notes}"
            ),
        )
    req = req.model_copy(update={"image_path": _resolve_image_path(req)})

    job_id = get_store().create(
        KIND, model_id=req.model_id, image_id=req.image_id,
        image_name=_display_name(req.image_path), tiles_processed=0,
    )
    background_tasks.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}")
def job_status(job_id: str):
    job = get_store().get(job_id, KIND)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job
