"""
Source-image library: sidecar metadata for uploaded GeoTIFFs.

This module is deliberately kept free of rasterio/GDAL imports so it is
fully unit-testable in any Python environment (the raster-reading step
itself still happens in app/api/images.py via app.engine.raster_io, which
*does* need rasterio and is lazy-imported there as elsewhere in this repo).

Each upload gets one sidecar JSON file next to it on disk
(`<dest>.meta.json`). The sidecar is the single source of truth the
library list/search/filter UI reads from, so listing uploaded images never
needs to re-open large GeoTIFFs just to show a thumbnail row.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(image_path.suffix + ".meta.json")


@dataclass
class LibraryEntry:
    image_id: str
    filename: str
    path: str
    size_bytes: int
    uploaded_at: str  # ISO 8601
    width: int | None = None
    height: int | None = None
    bands: int | None = None
    dtype: str | None = None
    crs: str | None = None
    gsd_x_m: float | None = None
    gsd_y_m: float | None = None
    area_km2: float | None = None
    bounds_wgs84: dict[str, float] | None = None  # {south, west, north, east} — for map ImageOverlay placement
    raster_meta_note: str | None = None  # set when raster metadata couldn't be read
    health: dict[str, Any] | None = None  # app.engine.raster_health.check_raster(...).to_dict()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_area_km2(width: int | None, height: int | None, gsd_x_m: float | None, gsd_y_m: float | None) -> float | None:
    """Ground area from pixel dimensions + ground sample distance — real geometry, no guessing."""
    if not width or not height or not gsd_x_m or not gsd_y_m:
        return None
    area_m2 = width * gsd_x_m * height * gsd_y_m
    return round(area_m2 / 1_000_000, 3)


def build_entry(
    image_id: str,
    filename: str,
    path: Path,
    size_bytes: int,
    uploaded_at: str,
    raster_meta: dict[str, Any] | None = None,
    raster_meta_note: str | None = None,
    bounds_wgs84: dict[str, float] | None = None,
    health: dict[str, Any] | None = None,
) -> LibraryEntry:
    raster_meta = raster_meta or {}
    width = raster_meta.get("width")
    height = raster_meta.get("height")
    gsd_x_m = raster_meta.get("gsd_x_m")
    gsd_y_m = raster_meta.get("gsd_y_m")
    return LibraryEntry(
        image_id=image_id,
        filename=filename,
        path=str(path),
        size_bytes=size_bytes,
        uploaded_at=uploaded_at,
        width=width,
        height=height,
        bands=raster_meta.get("bands"),
        dtype=raster_meta.get("dtype"),
        crs=raster_meta.get("crs"),
        gsd_x_m=gsd_x_m,
        gsd_y_m=gsd_y_m,
        area_km2=compute_area_km2(width, height, gsd_x_m, gsd_y_m),
        bounds_wgs84=bounds_wgs84,
        raster_meta_note=raster_meta_note,
        health=health,
    )


def write_entry(image_path: Path, entry: LibraryEntry) -> Path:
    sc = sidecar_path(image_path)
    sc.write_text(json.dumps(entry.to_dict(), indent=2))
    return sc


def read_entry(image_path: Path) -> dict[str, Any] | None:
    sc = sidecar_path(image_path)
    if not sc.exists():
        return None
    try:
        return json.loads(sc.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def list_library(upload_dir: Path) -> list[dict[str, Any]]:
    """All uploaded images with sidecar metadata, newest first.

    An uploaded .tif with no sidecar yet (e.g. uploaded by an older version
    of this API, or a future partial-write race) still shows up with
    whatever we can tell from the filesystem alone, rather than vanishing
    from the library silently.
    """
    out: list[dict[str, Any]] = []
    if not upload_dir.exists():
        return out
    for p in sorted(upload_dir.glob("*")):
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue
        entry = read_entry(p)
        if entry is None:
            stat = p.stat()
            entry = {
                "image_id": None,
                "filename": p.name,
                "path": str(p),
                "size_bytes": stat.st_size,
                "uploaded_at": None,
                "width": None, "height": None, "bands": None, "dtype": None, "crs": None,
                "gsd_x_m": None, "gsd_y_m": None, "area_km2": None,
                "raster_meta_note": "No metadata sidecar found for this file.",
            }
        out.append(entry)
    out.sort(key=lambda e: e.get("uploaded_at") or "", reverse=True)
    return out


def find_entry_by_image_id(upload_dir: Path, image_id: str) -> dict[str, Any] | None:
    for entry in list_library(upload_dir):
        if entry.get("image_id") == image_id:
            return entry
    return None
