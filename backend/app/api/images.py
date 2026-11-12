"""Source-image management: upload, list, and metadata inspection."""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.image_library import build_entry, find_entry_by_image_id, list_library, write_entry

router = APIRouter(prefix="/api/images", tags=["images"])


@router.post("")
async def upload_image(file: UploadFile):
    settings.ensure_dirs()
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only GeoTIFF (.tif/.tiff) uploads are supported.")

    image_id = str(uuid.uuid4())
    dest = settings.UPLOAD_DIR / f"{image_id}_{file.filename}"

    # Both the disk copy and the rasterio metadata reads below are blocking
    # (synchronous file I/O / C-extension calls). On a multi-GB GeoTIFF the
    # copy alone can run for a long time; running it directly in this `async
    # def` handler would stall FastAPI's single event loop for that whole
    # duration, freezing *every other* request the frontend is making in the
    # background (job/usage/environment polling) — which is what made the
    # whole app, not just the upload, look hung on a large upload. Running it
    # in a worker thread keeps the server responsive while the copy proceeds.
    def _copy():
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f, length=8 * 1024 * 1024)

    await run_in_threadpool(_copy)

    raster_meta = None
    raster_meta_note = None
    bounds_wgs84 = None
    try:
        from app.engine.raster_io import compute_wgs84_bounds, read_raster_info
        info = await run_in_threadpool(read_raster_info, dest)
        raster_meta = {
            "width": info.width, "height": info.height, "bands": info.count,
            "dtype": info.dtype, "crs": info.crs, "gsd_x_m": info.gsd_x, "gsd_y_m": info.gsd_y,
        }
        bounds_wgs84 = await run_in_threadpool(compute_wgs84_bounds, dest)
    except ImportError:
        raster_meta_note = "rasterio not installed in this environment — metadata unavailable until deployed with full requirements.txt"
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read raster: {exc}")

    entry = build_entry(
        image_id=image_id,
        filename=file.filename,
        path=dest,
        size_bytes=dest.stat().st_size,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        raster_meta=raster_meta,
        raster_meta_note=raster_meta_note,
        bounds_wgs84=bounds_wgs84,
    )
    write_entry(dest, entry)

    return entry.to_dict()


@router.get("")
def list_images():
    settings.ensure_dirs()
    return list_library(settings.UPLOAD_DIR)


@router.get("/{image_id}")
def get_image(image_id: str):
    settings.ensure_dirs()
    entry = find_entry_by_image_id(settings.UPLOAD_DIR, image_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No library image with id '{image_id}'.")
    return entry


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
