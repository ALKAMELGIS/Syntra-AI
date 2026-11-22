"""Click to segment and outline geometry, for labelling.

    GET  /api/segment/status                 is the click tool installed, last timings
    POST /api/segment/install                download SAM 2.1 tiny once (job)
    GET  /api/segment/install/{job_id}
    POST /api/images/{image_id}/segment      clicks + the visible view -> fitted outline
    POST /api/images/{image_id}/outline      a drawn or traced ring -> fitted outline
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.annotations import store
from app.core.config import settings
from app.core.image_library import find_entry_by_image_id
from app.core.jobs import get_store

router = APIRouter(tags=["segment"])
INSTALL_KIND = "segment_install"


class Refine(BaseModel):
    shape: str | None = None              # None: from the class name
    simplify_m: float = Field(default=0.6, ge=0, le=50)
    grow_m: float = Field(default=0.0, ge=-50, le=50)
    fill_holes_m2: float = Field(default=6.0, ge=0, le=1e6)
    min_area_m2: float = Field(default=0.0, ge=0, le=1e9)


class SegmentRequest(Refine):
    pick: str = "large"                    # one click: "large" = the whole object, "best" = SAM's top mask (a crown, a part)
    clicks: list[list[float]]              # [lon, lat, 1 add | 0 remove | 2 box top-left | 3 box bottom-right]
    view: list[float]                      # visible map: [west, south, east, north]
    class_name: str = ""
    device: str = "cpu"


class OutlineRequest(Refine):
    points_lonlat: list[list[float]]
    class_name: str = ""


class InstallRequest(BaseModel):
    model: str = "sam2.1_tiny"


def _entry(image_id: str) -> dict:
    settings.ensure_dirs()
    entry = find_entry_by_image_id(settings.UPLOAD_DIR, image_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No library image with id '{image_id}'.")
    if not entry.get("bounds_wgs84") or not entry.get("width") or not entry.get("height"):
        raise HTTPException(status_code=409, detail="This image has no geographic bounds yet")
    return entry


@lru_cache(maxsize=32)
def _gsd(path: str, mtime: float) -> float | None:
    try:
        from app.engine.segmentation_run import ground_gsd_m
        return ground_gsd_m(path)
    except Exception:  # noqa: BLE001 - no CRS / unreadable: sizes fall back to pixels
        return None


def _opts(req: Refine, class_name: str, path: str):
    from app.engine.shapes import RefineOptions, shape_for_class

    gsd = _gsd(path, Path(path).stat().st_mtime) if Path(path).exists() else None
    return RefineOptions(shape=req.shape or shape_for_class(class_name), px_m=gsd, simplify_m=req.simplify_m,
                         grow_m=req.grow_m, fill_holes_m2=req.fill_holes_m2, min_area_m2=req.min_area_m2)


def _answer(res: dict, entry: dict) -> dict:
    b, w, h = entry["bounds_wgs84"], entry["width"], entry["height"]
    ring = [list(map(float, p)) for p in list(res.pop("polygon").exterior.coords)[:-1]]
    res["points_px"] = ring
    res["points_lonlat"] = [list(store.pixel_to_lonlat(x, y, b, w, h)) for x, y in ring]
    return res


@router.get("/api/segment/status")
def segment_status():
    from app.engine import click_segment
    return click_segment.status()


def _run_install(job_id: str, model: str) -> None:
    from app.engine import click_segment

    jobs = get_store()
    jobs.update(job_id, status="processing")
    try:
        result = click_segment.install(model, progress=lambda p: jobs.update(job_id, progress=round(p, 1)))
        jobs.update(job_id, status="completed", progress=100.0, result=result)
    except Exception as exc:  # noqa: BLE001 - shown to the user as it is
        jobs.update(job_id, status="error", error=str(exc))


@router.post("/api/segment/install")
def segment_install(body: InstallRequest, background_tasks: BackgroundTasks):
    from app.engine import click_segment

    if body.model not in click_segment.MODELS:
        raise HTTPException(status_code=400, detail="Unknown model")
    job_id = get_store().create(INSTALL_KIND, progress=0.0, model=body.model)
    background_tasks.add_task(_run_install, job_id, body.model)
    return {"job_id": job_id, "status": "queued"}


@router.get("/api/segment/install/{job_id}")
def segment_install_status(job_id: str):
    job = get_store().get(job_id)
    if not job or job.get("kind") not in (None, INSTALL_KIND):
        raise HTTPException(status_code=404, detail="No such install")
    return job


@router.post("/api/images/{image_id}/segment")
async def segment_click(image_id: str, req: SegmentRequest):
    from app.engine import click_segment

    entry = _entry(image_id)
    b, w, h = entry["bounds_wgs84"], entry["width"], entry["height"]
    if not req.clicks or any(len(c) != 3 or int(c[2]) not in (0, 1, 2, 3) for c in req.clicks) or len(req.view) != 4:
        raise HTTPException(status_code=400, detail="Send clicks as [lon, lat, 1|0|2|3] and the view as [west, south, east, north]")
    clicks = [(*store.lonlat_to_pixel(c[0], c[1], b, w, h), int(c[2])) for c in req.clicks]
    x0, y0 = store.lonlat_to_pixel(req.view[0], req.view[3], b, w, h)
    x1, y1 = store.lonlat_to_pixel(req.view[2], req.view[1], b, w, h)
    try:
        window = click_segment.view_window((x0, y0, x1, y1), w, h)
        opts = _opts(req, req.class_name, entry["path"])
        res = await run_in_threadpool(click_segment.segment, entry["path"], window, clicks, opts, req.device, None, req.pick != "best")
    except click_segment.SegmentUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _answer(res, entry)


class PrepareRequest(BaseModel):
    view: list[float]
    device: str = "cpu"


@router.post("/api/images/{image_id}/segment/prepare")
async def segment_prepare(image_id: str, req: PrepareRequest):
    """Encode the visible view now, so the first click on it answers at once."""
    from app.engine import click_segment

    entry = _entry(image_id)
    b, w, h = entry["bounds_wgs84"], entry["width"], entry["height"]
    if len(req.view) != 4:
        raise HTTPException(status_code=400, detail="Send the view as [west, south, east, north]")
    x0, y0 = store.lonlat_to_pixel(req.view[0], req.view[3], b, w, h)
    x1, y1 = store.lonlat_to_pixel(req.view[2], req.view[1], b, w, h)
    try:
        window = click_segment.view_window((x0, y0, x1, y1), w, h)
        return await run_in_threadpool(click_segment.prepare, entry["path"], window, req.device)
    except click_segment.SegmentUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class SegmentAllRequest(Refine):
    view: list[float]                      # visible map: [west, south, east, north]
    class_name: str = ""
    device: str = "cpu"
    grid: int = Field(default=16, ge=4, le=32)


@router.post("/api/images/{image_id}/segment-all")
async def segment_all(image_id: str, req: SegmentAllRequest):
    """Every object in the visible part of the image (SAM over a grid of points):
    suggestions to keep or drop, fitted to the class's shape."""
    from app.engine import click_segment

    entry = _entry(image_id)
    b, w, h = entry["bounds_wgs84"], entry["width"], entry["height"]
    if len(req.view) != 4:
        raise HTTPException(status_code=400, detail="Send the view as [west, south, east, north]")
    x0, y0 = store.lonlat_to_pixel(req.view[0], req.view[3], b, w, h)
    x1, y1 = store.lonlat_to_pixel(req.view[2], req.view[1], b, w, h)
    try:
        window = click_segment.view_window((x0, y0, x1, y1), w, h)
        opts = _opts(req, req.class_name, entry["path"])
        res = await run_in_threadpool(click_segment.segment_everything, entry["path"], window, opts, req.device, None, req.grid)
    except click_segment.SegmentUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    res["objects"] = [_answer(o, entry) for o in res["objects"]]
    return res


@router.post("/api/images/{image_id}/outline")
def outline(image_id: str, req: OutlineRequest):
    """Fit a shape to a ring someone drew or traced (no model needed)."""
    from shapely.geometry import Polygon

    from app.engine.shapes import refine

    entry = _entry(image_id)
    b, w, h = entry["bounds_wgs84"], entry["width"], entry["height"]
    if len(req.points_lonlat) < 3:
        raise HTTPException(status_code=400, detail="An outline needs at least 3 points")
    ring = [store.lonlat_to_pixel(lon, lat, b, w, h) for lon, lat in req.points_lonlat]
    try:
        poly = Polygon(ring).buffer(0)
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        res = refine(poly, _opts(req, req.class_name, entry["path"]))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _answer(res, entry)
