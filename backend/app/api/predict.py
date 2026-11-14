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
    rgb_bands: list[int] | None = None    # 1-based R,G,B band numbers; default chosen per file (see raster_prep)
    aoi: dict | None = None               # WGS84 GeoJSON Polygon/Feature drawn on the map; analyse only inside it
    index_bands: list[int] | None = None  # vegetation: [red, nir] for NDVI or [red, green, blue] for VARI
    index_threshold: float | None = None  # vegetation: index value above which a pixel counts as vegetation


def _run_job(job_id: str, req: PredictRequest):
    jobs = get_store()
    jobs.update(job_id, status="processing")
    try:
        if req.model_id in INDEX_MODELS:
            _run_index_job(job_id, req)
            return
        from app.models.packs import pack_for_model
        pack = pack_for_model(req.model_id)
        if pack is not None and pack.runtime == "geodeep_segmentor":
            _run_segmentation_job(job_id, req, pack)
            return
        from app.engine.pipeline import DetectionEngine
        from app.engine.raster_io import count_tiles, iter_tiles, raster_crs_and_size
        from app.engine.raster_prep import plan_rgb
        from app.models.factory import build_model

        model = build_model(req.model_id)
        engine = DetectionEngine(model)

        processed, skipped = {"n": 0}, {"n": 0}

        def progress(done, total):
            # done is None for an empty tile skipped before inference
            if done is not None:
                processed["n"] = done
            jobs.update(job_id, tiles_processed=processed["n"] + skipped["n"], tiles_empty=skipped["n"])

        source_crs, width, height = raster_crs_and_size(req.image_path)
        if hasattr(model, "preferred_tile_size"):
            # read tiles at the scale the model was trained at
            from app.engine.segmentation_run import ground_gsd_m
            req = req.model_copy(update={"tile_size": model.preferred_tile_size(ground_gsd_m(req.image_path))})
            jobs.update(job_id, tile_size=req.tile_size)
        aoi = None
        if req.aoi:
            from app.engine.aoi import prepare_aoi
            aoi = prepare_aoi(req.image_path, req.aoi)
            width, height = aoi.window[2], aoi.window[3]
            jobs.update(job_id, aoi_area_m2=aoi.area_m2)
        tiles_total = count_tiles(width, height, req.tile_size, req.overlap)
        jobs.update(job_id, tiles_total=tiles_total)
        plan = plan_rgb(req.image_path, override=req.rgb_bands)
        jobs.update(job_id, rgb_plan=plan.to_dict())

        def on_skip():
            skipped["n"] += 1
            progress(None, None)

        tiles = iter_tiles(req.image_path, tile_size=req.tile_size, overlap=req.overlap,
                           rgb_plan=plan, on_skip=on_skip,
                           window=aoi.window if aoi else None,
                           tile_filter=aoi.tile_intersects if aoi else None)
        out_dir = settings.RESULTS_DIR / job_id
        result = engine.run(
            tiles, out_dir, device_option=req.device,
            confidence_threshold=req.confidence_threshold, progress_cb=progress,
            allowed_gpu_indices=req.gpu_indices,
            source_crs=source_crs, tiles_total=tiles_total,
            keep_detection=(lambda d: aoi.contains_point(*d["centroid_geo"])) if aoi else None,
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


INDEX_MODELS = {"vegetation_index"}


def _run_index_job(job_id: str, req: PredictRequest) -> None:
    """Spectral-index analyses: whole-window raster maths, not tiled detection."""
    from app.engine.vegetation import run_vegetation

    jobs = get_store()
    aoi = None
    if req.aoi:
        from app.engine.aoi import prepare_aoi
        aoi = prepare_aoi(req.image_path, req.aoi)
        jobs.update(job_id, aoi_area_m2=aoi.area_m2)

    def progress(done, total):
        jobs.update(job_id, tiles_processed=done, tiles_total=total)

    result = run_vegetation(
        req.image_path, settings.RESULTS_DIR / job_id, req.model_id, aoi=aoi,
        threshold=req.index_threshold, index_bands=req.index_bands, progress_cb=progress,
    )
    jobs.update(job_id, status="completed", result=result)
    record_run(
        settings.DATA_DIR, req.model_id, status="completed", tiles_processed=1,
        detection_count=result["final_detection_count"], duration_seconds=result["duration_seconds"], device="cpu",
    )


def _run_segmentation_job(job_id: str, req: PredictRequest, pack) -> None:
    from app.engine.segmentation_run import run_segmentation
    from app.models.packs import pack_path

    jobs = get_store()
    aoi = None
    if req.aoi:
        from app.engine.aoi import prepare_aoi
        aoi = prepare_aoi(req.image_path, req.aoi)
        jobs.update(job_id, aoi_area_m2=aoi.area_m2)

    def progress(done, total):
        jobs.update(job_id, tiles_processed=done, tiles_total=total)

    meta = get_metadata(req.model_id)
    result = run_segmentation(
        req.image_path, settings.RESULTS_DIR / job_id, req.model_id, str(pack_path(pack)),
        class_names=meta.class_names, aoi=aoi, rgb_bands=req.rgb_bands, progress_cb=progress,
    )
    jobs.update(job_id, status="completed", result=result)
    record_run(settings.DATA_DIR, req.model_id, status="completed", tiles_processed=result["tiles_processed"],
               detection_count=result["final_detection_count"], duration_seconds=result["duration_seconds"], device="cpu")


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


def _refuse_wrong_scale(meta, image_path: str) -> None:
    """Drone-only models are refused on imagery far coarser than they were trained on
    (a 4 cm model on a 50 cm satellite scene would see every object as a few pixels)."""
    if meta.supported_image_sources != ["drone"]:
        return
    try:
        from app.engine.segmentation_run import ground_gsd_m
        gsd = ground_gsd_m(image_path)
    except Exception:
        return
    if gsd and gsd > 1.5 * meta.supported_gsd.max_m:
        raise HTTPException(status_code=409, detail=(
            f"'{meta.name}' is a drone model ({meta.supported_gsd.min_m * 100:g}-{meta.supported_gsd.max_m * 100:g} cm per pixel); "
            f"this image is {gsd * 100:.0f} cm per pixel. Use the satellite models for it."))


@router.post("")
def start_prediction(req: PredictRequest, background_tasks: BackgroundTasks):
    meta = get_metadata(req.model_id)
    if meta.status == ModelStatus.NOT_INSTALLED:
        raise HTTPException(status_code=409, detail=f"'{meta.name}' is not installed yet. Install its model first.")
    if meta.status == ModelStatus.NOT_PRODUCTION_READY:
        raise HTTPException(
            status_code=409,
            detail=(
                f"'{meta.name}' is not production-ready yet (no validated weights/benchmark). "
                f"Refusing to generate fake detections. Details: {meta.notes}"
            ),
        )
    req = req.model_copy(update={"image_path": _resolve_image_path(req)})
    try:
        from app.engine.raster_io import raster_crs_and_size
        no_crs = raster_crs_and_size(req.image_path)[0] is None
    except Exception:
        no_crs = False   # unreadable or rasterio missing: the job itself records the error
    if no_crs:
        raise HTTPException(status_code=409, detail=(
            "This image has no coordinate system, so results can't be placed on the map. "
            "Assign one (QGIS: Raster > Projections > Assign Projection) and upload it again."))
    _refuse_wrong_scale(meta, req.image_path)
    if req.aoi is not None:
        # Fail fast with a clear message instead of a background error.
        try:
            from app.engine.aoi import AoiError, prepare_aoi
            prepare_aoi(req.image_path, req.aoi)
        except ImportError:
            raise HTTPException(status_code=503, detail="AOI needs rasterio, shapely and pyproj installed.")
        except AoiError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    job_id = get_store().create(
        KIND, model_id=req.model_id, image_id=req.image_id,
        image_name=_display_name(req.image_path), tiles_processed=0, has_aoi=req.aoi is not None,
    )
    background_tasks.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}")
def job_status(job_id: str):
    job = get_store().get(job_id, KIND)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job
