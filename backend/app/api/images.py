"""Source-image management: upload, list, and metadata inspection."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import safe_filename
from app.core.image_library import build_entry, find_entry_by_image_id, list_library, write_entry

router = APIRouter(prefix="/api/images", tags=["images"])


@router.post("")
async def upload_image(file: UploadFile, background_tasks: BackgroundTasks, metadata: UploadFile | None = None):
    """Upload a GeoTIFF, optionally with the product's metadata XML (e.g. the
    SuperView/WorldView sidecar). The XML carries calibration gains and the
    sensor id, which vegetation indices and band detection use."""
    settings.ensure_dirs()
    filename = safe_filename(file.filename)
    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only GeoTIFF (.tif/.tiff) uploads are supported.")

    image_id = str(uuid.uuid4())
    dest = settings.UPLOAD_DIR / f"{image_id}_{filename}"
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    # Both the disk copy and the rasterio metadata reads below are blocking
    # (synchronous file I/O / C-extension calls). On a multi-GB GeoTIFF the
    # copy alone can run for a long time; running it directly in this `async
    # def` handler would stall FastAPI's single event loop for that whole
    # duration, freezing *every other* request the frontend is making in the
    # background (job/usage/environment polling) — which is what made the
    # whole app, not just the upload, look hung on a large upload. Running it
    # in a worker thread keeps the server responsive while the copy proceeds.
    def _copy() -> bool:
        written = 0
        with dest.open("wb") as f:
            while chunk := file.file.read(8 * 1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    return False
                f.write(chunk)
        return True

    if metadata is not None and metadata.filename:
        if not safe_filename(metadata.filename).lower().endswith(".xml"):
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Product metadata must be an .xml file.")
        xml_bytes = await metadata.read(2 * 1024 * 1024 + 1)
        if len(xml_bytes) > 2 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Product metadata XML is larger than 2 MB.")

    if not await run_in_threadpool(_copy):
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit (SYNTRA_MAX_UPLOAD_MB).",
        )

    if metadata is not None and metadata.filename:
        # stored next to the image under the same stem, where detect_sensor()
        # and read_calibration_gains() look for it
        dest.with_suffix(".xml").write_bytes(xml_bytes)

    raster_meta = None
    raster_meta_note = None
    preview_bounds = None
    bounds_wgs84 = None
    health = None
    georef = None
    try:
        from app.engine.georef import ensure_georeferenced
        georef = await run_in_threadpool(ensure_georeferenced, dest)
    except ImportError:
        pass
    except Exception as exc:
        georef = {"method": "failed", "error": str(exc)}

    try:
        from app.engine.raster_io import compute_wgs84_bounds, read_raster_info
        info = await run_in_threadpool(read_raster_info, dest)
        raster_meta = {
            "width": info.width, "height": info.height, "bands": info.count,
            "dtype": info.dtype, "crs": info.crs, "gsd_x_m": info.gsd_x, "gsd_y_m": info.gsd_y,
        }
        bounds_wgs84 = await run_in_threadpool(compute_wgs84_bounds, dest)
        from app.engine.raster_io import preview_bounds_wgs84
        preview_bounds = await run_in_threadpool(preview_bounds_wgs84, dest)
        from app.engine.raster_health import check_raster
        health = (await run_in_threadpool(check_raster, dest)).to_dict()
        from app.engine.georef import georef_finding
        if georef and georef.get("method") != "failed" and (finding := georef_finding(georef)):
            health["findings"].insert(0, finding)
            if finding["level"] == "warning" and health["status"] == "ok":
                health["status"] = "warning"
        if health.get("gsd_m") and not health.get("projected"):
            # geographic CRS: pixel size is in degrees; store the ground size in metres
            raster_meta["gsd_x_m"] = raster_meta["gsd_y_m"] = health["gsd_m"]
    except ImportError:
        raster_meta_note = "rasterio not installed in this environment — metadata unavailable until deployed with full requirements.txt"
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read raster: {exc}")

    entry = build_entry(
        image_id=image_id,
        filename=filename,
        path=dest,
        size_bytes=dest.stat().st_size,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        raster_meta=raster_meta,
        raster_meta_note=raster_meta_note,
        bounds_wgs84=bounds_wgs84,
        health=health,
        preview_bounds_wgs84=preview_bounds,
    )
    out = entry.to_dict()
    # Large stripped files or files without overviews are made fast to read in the
    # background (tiles + overviews); the image is usable meanwhile.
    # A file with no image data (a mask) is not worth retiling: nothing will run on it.
    empty = any(f.get("code") == "no_data" for f in (health or {}).get("findings", []))
    try:
        from app.engine.optimize import needs_optimizing
        plan = None if empty else await run_in_threadpool(needs_optimizing, dest)
    except Exception:
        plan = None
    if plan:
        from app.core.jobs import get_store
        job_id = get_store().create("optimize", image_id=image_id, progress=0.0, plan=plan)
        out["optimize_job_id"] = job_id
        background_tasks.add_task(_run_optimize, job_id, str(dest))
    entry_dict = out
    _save_dict(dest, entry_dict)
    return entry_dict


def _save_dict(dest: Path, entry: dict) -> None:
    import json
    from app.core.image_library import sidecar_path
    sidecar_path(dest).write_text(json.dumps(entry, indent=2))


def _run_optimize(job_id: str, path: str) -> None:
    from app.core.jobs import get_store
    from app.engine.optimize import optimize

    jobs = get_store()
    jobs.update(job_id, status="processing")
    try:
        res = optimize(path, progress=lambda pct: jobs.update(job_id, progress=round(pct, 1)))
        # the raster check (layout, overviews) changed: redo it on next listing
        import json
        from app.core.image_library import sidecar_path
        sc = sidecar_path(Path(path))
        entry = json.loads(sc.read_text())
        if entry.get("health"):
            entry["health"]["version"] = 0
        entry.pop("optimize_job_id", None)
        sc.write_text(json.dumps(entry, indent=2))
        jobs.update(job_id, status="completed", result=res, progress=100.0)
    except Exception as exc:
        jobs.update(job_id, status="error", error=str(exc))


def _fresh(entry: dict) -> dict:
    """Bring a saved library entry up to date once, and save it: the Web Mercator
    preview bounds (missing on images uploaded before the preview was reprojected),
    and a raster check saved by an older version of the rules."""
    if entry.get("bounds_wgs84") and not entry.get("preview_bounds_wgs84"):
        try:
            from app.engine.raster_io import preview_bounds_wgs84
            entry = {**entry, "preview_bounds_wgs84": preview_bounds_wgs84(entry["path"])}
            _save(entry)
        except Exception:
            pass
    health = entry.get("health")
    if not health:
        return entry
    try:
        from app.engine.raster_health import HEALTH_VERSION, check_raster
    except ImportError:
        return entry
    if health.get("version") == HEALTH_VERSION:
        return entry
    try:
        new = check_raster(entry["path"]).to_dict()
    except Exception:
        return entry
    kept = [f for f in health.get("findings", []) if str(f.get("code", "")).startswith("georef")]
    new["findings"] = kept + new["findings"]
    if any(f["level"] == "warning" for f in kept) and new["status"] == "ok":
        new["status"] = "warning"
    entry = {**entry, "health": new}
    _save(entry)
    return entry


def _save(entry: dict) -> None:
    try:
        import json
        from app.core.image_library import sidecar_path
        sidecar_path(Path(entry["path"])).write_text(json.dumps(entry, indent=2))
    except OSError:
        pass


@router.get("")
def list_images():
    settings.ensure_dirs()
    return [_fresh(e) for e in list_library(settings.UPLOAD_DIR)]


@router.get("/{image_id}")
def get_image(image_id: str):
    settings.ensure_dirs()
    entry = find_entry_by_image_id(settings.UPLOAD_DIR, image_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No library image with id '{image_id}'.")
    return _fresh(entry)


@router.get("/{image_id}/health")
def get_image_health(image_id: str):
    """Re-run the raster health check (CRS, resolution, bit depth, layout...)."""
    settings.ensure_dirs()
    entry = find_entry_by_image_id(settings.UPLOAD_DIR, image_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No library image with id '{image_id}'.")
    try:
        from app.engine.raster_health import check_raster
        return check_raster(entry["path"]).to_dict()
    except ImportError:
        raise HTTPException(status_code=503, detail="rasterio not installed — health check unavailable in this environment.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read raster: {exc}")


_EMPTY_TILE = None


def _empty_png() -> bytes:
    global _EMPTY_TILE
    if _EMPTY_TILE is None:
        import io
        from PIL import Image
        b = io.BytesIO()
        Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(b, format="PNG")
        _EMPTY_TILE = b.getvalue()
    return _EMPTY_TILE


@router.get("/{image_id}/tiles/{z}/{x}/{y}.png")
def get_image_tile(image_id: str, z: int, x: int, y: int):
    """Full-resolution XYZ tile (Web Mercator, 256 px) cut from the GeoTIFF."""
    if not (0 <= z <= 26 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise HTTPException(status_code=400, detail="Tile out of range.")
    entry = find_entry_by_image_id(settings.UPLOAD_DIR, image_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No library image with id '{image_id}'.")
    try:
        from app.engine.tiles import render_tile
        png = render_tile(entry["path"], z, x, y)
    except ImportError:
        raise HTTPException(status_code=503, detail="rasterio not installed.")
    headers = {"Cache-Control": "private, max-age=3600"}
    return Response(content=png or _empty_png(), media_type="image/png", headers=headers)


@router.get("/{image_id}/preview.png")
def get_image_preview(image_id: str):
    """A real, decimated true-color PNG of the raster for the map overlay —
    not a placeholder box. See app.engine.raster_io.read_preview_png."""
    settings.ensure_dirs()
    entry = find_entry_by_image_id(settings.UPLOAD_DIR, image_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No library image with id '{image_id}'.")
    try:
        from app.engine.raster_io import read_preview_png
        png_bytes = read_preview_png(entry["path"])
    except ImportError:
        raise HTTPException(status_code=503, detail="rasterio/Pillow not installed — preview unavailable in this environment.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not render preview: {exc}")
    return Response(content=png_bytes, media_type="image/png")
