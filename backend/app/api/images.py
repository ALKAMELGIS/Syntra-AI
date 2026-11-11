"""Source-image management: upload, list, and metadata inspection."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.config import settings

router = APIRouter(prefix="/api/images", tags=["images"])


@router.post("")
async def upload_image(file: UploadFile):
    settings.ensure_dirs()
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only GeoTIFF (.tif/.tiff) uploads are supported.")

    image_id = str(uuid.uuid4())
    dest = settings.UPLOAD_DIR / f"{image_id}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from app.engine.raster_io import read_raster_info
        info = read_raster_info(dest)
        meta = {
            "width": info.width, "height": info.height, "bands": info.count,
            "dtype": info.dtype, "crs": info.crs, "gsd_x_m": info.gsd_x, "gsd_y_m": info.gsd_y,
        }
    except ImportError:
        meta = {"note": "rasterio not installed in this environment — metadata unavailable until deployed with full requirements.txt"}
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read raster: {exc}")

    return {"image_id": image_id, "filename": file.filename, "path": str(dest), **meta}


@router.get("")
def list_images():
    settings.ensure_dirs()
    out = []
    for p in sorted(settings.UPLOAD_DIR.glob("*")):
        if p.suffix.lower() in (".tif", ".tiff"):
            out.append({"filename": p.name, "path": str(p), "size_bytes": p.stat().st_size})
    return out
