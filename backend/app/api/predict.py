"""
Run-prediction endpoint.

Kicks off a detection run for one model against one uploaded image, tracks
progress in an in-memory job store (swap for Redis/DB in a multi-worker
deployment), and reports NOT_PRODUCTION_READY models honestly instead of
pretending to run them.
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.core.compute import DeviceOption
from app.core.config import settings
from app.models.base import ModelStatus
from app.models.registry import get_metadata
from app.models.stats import record_run

router = APIRouter(prefix="/api/predict", tags=["predict"])

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


class PredictRequest(BaseModel):
    image_path: str
    model_id: str
    device: DeviceOption = DeviceOption.AUTO
    confidence_threshold: float | None = None
    tile_size: int = settings.DEFAULT_TILE_SIZE
    overlap: float = settings.DEFAULT_TILE_OVERLAP


def _run_job(job_id: str, req: PredictRequest):
    with _LOCK:
        _JOBS[job_id]["status"] = "processing"
    try:
        from app.engine.pipeline import DetectionEngine
        from app.engine.raster_io import iter_tiles
        from app.models.factory import build_model

        model = build_model(req.model_id)
        engine = DetectionEngine(model)

        def progress(done, total):
            with _LOCK:
                _JOBS[job_id]["tiles_processed"] = done

        tiles = iter_tiles(req.image_path, tile_size=req.tile_size, overlap=req.overlap)
        out_dir = settings.RESULTS_DIR / job_id
        result = engine.run(
            tiles, out_dir, device_option=req.device,
            confidence_threshold=req.confidence_threshold, progress_cb=progress,
        )
        with _LOCK:
            _JOBS[job_id].update(status="completed", result=result.__dict__)
        record_run(
            settings.DATA_DIR, req.model_id, status="completed",
            tiles_processed=result.tiles_processed, detection_count=result.final_detection_count,
            duration_seconds=result.duration_seconds, device=result.device_used,
        )
    except Exception as exc:
        with _LOCK:
            _JOBS[job_id].update(status="error", error=str(exc))
        record_run(settings.DATA_DIR, req.model_id, status="error")


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
    if not Path(req.image_path).exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    job_id = str(uuid.uuid4())
    with _LOCK:
        _JOBS[job_id] = {"status": "queued", "model_id": req.model_id, "tiles_processed": 0}
    background_tasks.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}")
def job_status(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job
