"""Source-image management: upload, list, and metadata inspection."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import safe_filename
from app.core.image_library import build_entry, find_entry_by_image_id, list_library, write_entry

router = APIRouter(prefix="/api/images", tags=["images"])


@router.post("")
async def upload_image(file: UploadFile, metadata: UploadFile | None = None):
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
    bounds_wgs84 = None
    health = None
    try:
        from app.engine.raster_io import compute_wgs84_bounds, read_raster_info
        info = await run_in_threadpool(read_raster_info, dest)
        raster_meta = {
            "width": info.width, "height": info.height, "bands": info.count,
            "dtype": info.dtype, "crs": info.crs, "gsd_x_m": info.gsd_x, "gsd_y_m": info.gsd_y,
        }
        bounds_wgs84 = await run_in_threadpool(compute_wgs84_bounds, dest)
        from app.engine.raster_health import check_raster
        health = (await run_in_threadpool(check_raster, dest)).to_dict()
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
