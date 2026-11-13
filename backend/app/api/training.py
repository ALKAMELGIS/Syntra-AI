"""
Dataset export endpoint — first API step of the Model Setup wizard's
"Dataset" stage.

Tiles a large orthomosaic + pixel-space annotations into a YOLO dataset
(images/labels/train+val, data.yaml) using app.training.dataset_builder,
ported from the GeoSyntra AI QGIS plugin's own exporter. Runs as a
background job tracked in-memory, same pattern as /api/predict.

NOTE: `points_px` are expected in the raster's own pixel coordinate space
(column, row). Passing `image_id` instead of `image_path`/`class_names`/
`annotations` pulls them from app.annotations.store — the per-image
annotations saved by the map's Annotate tool — converted to pixel space
already; this is the normal way to call this endpoint from the wizard now.
Pixel-space annotations can still be given directly (e.g. a script, or a
test) when there's no saved image annotation set to draw from.
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import confine_path

router = APIRouter(prefix="/api/training", tags=["training"])

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


class AnnotationIn(BaseModel):
    class_id: int
    points_px: list[list[float]]   # polygon ring in raster pixel coords, e.g. [[x0,y0], [x1,y1], ...]


class ExportDatasetRequest(BaseModel):
    image_id: str | None = None      # shortcut: pulls image_path/class_names/annotations from the Annotate store
    image_path: str | None = None
    class_names: list[str] | None = None
    annotations: list[AnnotationIn] | None = None
    out_dir: str | None = None       # defaults to DATA_DIR/datasets/<job_id>
    tile: int = 640
    overlap: int = 64
    val_split: float = 0.2
    neg_ratio: float = 0.1
    task: str = "detect"             # "detect" | "obb" | "segment"
    augment: bool = True
    img_fmt: str = "jpg"


def _run_job(job_id: str, req: ExportDatasetRequest):
    import numpy as np

    with _LOCK:
        _JOBS[job_id]["status"] = "processing"
    try:
        from app.training.dataset_builder import DatasetBuilder
        from app.training.raster_window import RasterReader

        reader = RasterReader(req.image_path)
        anns = [{"cls": a.class_id, "pts": np.asarray(a.points_px, dtype=float)} for a in req.annotations]
        out_dir = req.out_dir or str(settings.DATA_DIR / "datasets" / job_id)

        def progress(pct, message=None):
            with _LOCK:
                _JOBS[job_id]["progress"] = round(pct, 1)
                if message:
                    _JOBS[job_id]["message"] = message

        def log(message):
            with _LOCK:
                _JOBS[job_id].setdefault("log", []).append(message)

        def canceled():
            with _LOCK:
                return _JOBS[job_id].get("cancel_requested", False)

        builder = DatasetBuilder(
            reader, anns, req.class_names, out_dir,
            tile=req.tile, overlap=req.overlap, val_split=req.val_split, neg_ratio=req.neg_ratio,
            task=req.task, augment=req.augment, img_fmt=req.img_fmt,
            progress=progress, canceled=canceled, log=log,
        )
        yaml_path, info = builder.run()
        reader.close()
        with _LOCK:
            _JOBS[job_id].update(status="completed", data_yaml=yaml_path, info=info)
    except Exception as exc:
        with _LOCK:
            _JOBS[job_id].update(status="error", error=str(exc))


def _resolve_from_image_id(req: ExportDatasetRequest) -> ExportDatasetRequest:
    from app.annotations import store
    from app.core.image_library import find_entry_by_image_id

    entry = find_entry_by_image_id(settings.UPLOAD_DIR, req.image_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No library image with id '{req.image_id}'.")
    anns = store.load_annotations(entry["path"])
    if not anns:
        raise HTTPException(status_code=409, detail="No annotations saved for this image yet — use the Annotate step first.")
    class_names, annotations = store.to_export_payload(anns)
    return req.model_copy(update={
        "image_path": entry["path"],
        "class_names": class_names,
        "annotations": [AnnotationIn(**a) for a in annotations],
    })


@router.post("/export-dataset")
def export_dataset(req: ExportDatasetRequest, background_tasks: BackgroundTasks):
    if req.image_id:
        req = _resolve_from_image_id(req)
    if not req.image_path:
        raise HTTPException(status_code=400, detail="Provide either image_id or image_path.")
    req = req.model_copy(update={
        "image_path": str(confine_path(req.image_path, field="image_path", must_exist=True)),
        "out_dir": str(confine_path(req.out_dir, field="out_dir")) if req.out_dir else None,
    })
    if not req.annotations:
        raise HTTPException(status_code=400, detail="At least one annotation is required to export a dataset.")
    if req.overlap >= req.tile:
        raise HTTPException(status_code=400, detail="overlap must be smaller than tile.")

    job_id = str(uuid.uuid4())
    with _LOCK:
        _JOBS[job_id] = {"status": "queued", "progress": 0.0}
    background_tasks.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@router.get("/export-dataset/{job_id}")
def export_dataset_status(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job


@router.post("/export-dataset/{job_id}/cancel")
def export_dataset_cancel(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        job["cancel_requested"] = True
    return {"job_id": job_id, "cancel_requested": True}


# --------------------------------------------------------------------- Train
_TRAIN_JOBS: dict[str, dict] = {}
_TRAIN_LOCK = threading.Lock()


class TrainRequest(BaseModel):
    data_yaml: str
    out_dir: str | None = None       # defaults to DATA_DIR/training_runs/<job_id>
    device: str = "cpu"              # "cpu" | "cuda:<n>" — resolve via /api/compute first
    model_size: str = "nano"         # "nano" | "small" | "medium"
    tile_size: int = 640
    epochs: int | None = None        # None -> auto-suggested from dataset size
    batch: int | None = None
    workers: int | None = None
    imgsz: int | None = None
    lr0: float | None = None
    patience: int | None = None


def _count_train_images(data_yaml: str) -> int:
    # app.training.dataset_builder.write_data_yaml always writes a
    # "images/train" subfolder next to data.yaml — no YAML parsing needed.
    train_dir = Path(data_yaml).parent / "images" / "train"
    try:
        return sum(1 for p in train_dir.iterdir() if p.is_file()) or 500
    except OSError:
        return 500  # unknown -> a reasonable middle-of-the-road default


def _run_train_job(job_id: str, req: TrainRequest):
    with _TRAIN_LOCK:
        _TRAIN_JOBS[job_id]["status"] = "processing"
    try:
        from app.training.trainer import build_job_config, run_training

        out_dir = req.out_dir or str(settings.DATA_DIR / "training_runs" / job_id)
        cancel_flag = str(settings.DATA_DIR / "training_runs" / job_id / "cancel.flag")
        cfg = build_job_config(
            req.data_yaml, out_dir,
            num_train_images=_count_train_images(req.data_yaml),
            device=req.device, model_size=req.model_size, tile_size=req.tile_size,
            overrides={
                "epochs": req.epochs, "batch": req.batch, "workers": req.workers,
                "imgsz": req.imgsz, "lr0": req.lr0, "patience": req.patience,
            },
            cancel_flag_path=cancel_flag,
        )
        with _TRAIN_LOCK:
            _TRAIN_JOBS[job_id]["config"] = cfg
            _TRAIN_JOBS[job_id]["cancel_flag_path"] = cancel_flag

        def on_event(event):
            with _TRAIN_LOCK:
                if event.get("type") == "epoch":
                    _TRAIN_JOBS[job_id]["progress"] = event
                    _TRAIN_JOBS[job_id].setdefault("history", []).append(event)
                elif event.get("type") in ("done", "canceled", "error"):
                    _TRAIN_JOBS[job_id]["status"] = event["type"]
                    _TRAIN_JOBS[job_id]["result"] = event

        final = run_training(cfg, on_event)
        with _TRAIN_LOCK:
            if _TRAIN_JOBS[job_id]["status"] == "processing":
                _TRAIN_JOBS[job_id]["status"] = final.get("type", "error")
                _TRAIN_JOBS[job_id]["result"] = final
    except Exception as exc:
        with _TRAIN_LOCK:
            _TRAIN_JOBS[job_id].update(status="error", error=str(exc))


@router.post("/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    req = req.model_copy(update={
        "data_yaml": str(confine_path(req.data_yaml, field="data_yaml", must_exist=True)),
        "out_dir": str(confine_path(req.out_dir, field="out_dir")) if req.out_dir else None,
    })
    job_id = str(uuid.uuid4())
    with _TRAIN_LOCK:
        _TRAIN_JOBS[job_id] = {"status": "queued", "data_yaml": req.data_yaml}
    background_tasks.add_task(_run_train_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@router.get("/train/{job_id}")
def train_status(job_id: str):
    with _TRAIN_LOCK:
        job = _TRAIN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job


@router.post("/train/{job_id}/cancel")
def train_cancel(job_id: str):
    from app.training.trainer import request_cancel

    with _TRAIN_LOCK:
        job = _TRAIN_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        flag_path = job.get("cancel_flag_path")
    if not flag_path:
        raise HTTPException(status_code=409, detail="Training hasn't started yet — nothing to cancel.")
    request_cancel(flag_path)
    return {"job_id": job_id, "cancel_requested": True}


class RegisterModelRequest(BaseModel):
    name: str
    notes: str | None = None


@router.post("/train/{job_id}/register-model")
def register_trained_model(job_id: str, req: RegisterModelRequest):
    """Closes the Train -> Detect loop: turns this run's best.pt into a
    real, selectable catalog entry (app.models.custom_registry), with its
    class names read straight back from the run's own data.yaml — not
    retyped by hand, so they can't drift from what was actually trained."""
    with _TRAIN_LOCK:
        job = _TRAIN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job.get("status") not in ("done", "completed"):
        raise HTTPException(status_code=409, detail=f"Training isn't finished yet (status: {job.get('status')}).")

    best = (job.get("result") or {}).get("best")
    if not best:
        raise HTTPException(status_code=409, detail="No best.pt weights were produced by this run.")

    data_yaml = job.get("data_yaml")
    class_names: dict[int, str] = {}
    if data_yaml and Path(data_yaml).exists():
        from app.training.dataset_builder import read_data_yaml_names
        try:
            class_names = dict(enumerate(read_data_yaml_names(data_yaml)))
        except Exception:
            class_names = {}
    if not class_names:
        raise HTTPException(status_code=409, detail="Could not read class names from this run's data.yaml.")

    from app.models.custom_registry import register_custom_model
    try:
        entry = register_custom_model(
            settings.DATA_DIR,
            name=req.name,
            weights_path=best,
            class_names=class_names,
            source_job_id=job_id,
            data_yaml=data_yaml,
            notes=req.notes or "",
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return entry.to_dict()
