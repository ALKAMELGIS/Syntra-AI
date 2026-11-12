"""
Persistence + lon/lat <-> pixel conversion for per-image object annotations.

Pure stdlib/json — no rasterio dependency — so this is importable and fully
unit-testable anywhere, same as app.core.image_library.

Coordinate convention: the frontend displays the raster via a Leaflet
`<ImageOverlay>` stretched linearly across the image's `bounds_wgs84`
rectangle (south/west/north/east) — see app/engine/raster_io.compute_wgs84_bounds
and frontend/src/components/MapView.jsx. Annotations are drawn as shapes on
top of that same overlay, so converting a drawn point back to the raster's
pixel grid must use the *same* linear stretch the overlay used, not a full
CRS reprojection — otherwise a box drawn flush against the image edge on
screen would not land on the image edge in pixel space. This assumes a
north-up, non-rotated raster, which is what compute_wgs84_bounds already
assumes by using `rasterio.warp.transform_bounds` (an axis-aligned
bounding box, not a General rotation-aware remap).
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def sidecar_path(image_path: str | Path) -> Path:
    p = Path(image_path)
    return p.with_suffix(p.suffix + ".annotations.json")


def lonlat_to_pixel(lon: float, lat: float, bounds: dict, width: int, height: int) -> tuple[float, float]:
    lon_span = bounds["east"] - bounds["west"]
    lat_span = bounds["north"] - bounds["south"]
    x = ((lon - bounds["west"]) / lon_span) * width if lon_span else 0.0
    y = ((bounds["north"] - lat) / lat_span) * height if lat_span else 0.0  # north is row 0
    return x, y


def pixel_to_lonlat(x: float, y: float, bounds: dict, width: int, height: int) -> tuple[float, float]:
    lon_span = bounds["east"] - bounds["west"]
    lat_span = bounds["north"] - bounds["south"]
    lon = bounds["west"] + (x / width) * lon_span if width else bounds["west"]
    lat = bounds["north"] - (y / height) * lat_span if height else bounds["north"]
    return lon, lat


def load_annotations(image_path: str | Path) -> list[dict]:
    sc = sidecar_path(image_path)
    if not sc.exists():
        return []
    try:
        return json.loads(sc.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_annotations(image_path: str | Path, annotations: list[dict]) -> None:
    sidecar_path(image_path).write_text(json.dumps(annotations, indent=2, ensure_ascii=False), encoding="utf-8")


def add_annotation(image_path: str | Path, class_name: str, points_lonlat: list[list[float]],
                    bounds: dict, width: int, height: int) -> dict:
    if len(points_lonlat) < 3:
        raise ValueError("An annotation needs at least 3 points (a closed ring).")
    points_px = [list(lonlat_to_pixel(lon, lat, bounds, width, height)) for lon, lat in points_lonlat]
    entry = {
        "id": str(uuid.uuid4()),
        "class_name": class_name.strip() or "object",
        "points_lonlat": points_lonlat,
        "points_px": points_px,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    anns = load_annotations(image_path)
    anns.append(entry)
    save_annotations(image_path, anns)
    return entry


def delete_annotation(image_path: str | Path, annotation_id: str) -> bool:
    anns = load_annotations(image_path)
    kept = [a for a in anns if a["id"] != annotation_id]
    if len(kept) == len(anns):
        return False
    save_annotations(image_path, kept)
    return True


def clear_annotations(image_path: str | Path) -> None:
    save_annotations(image_path, [])


def class_names_used(annotations: list[dict]) -> list[str]:
    seen: list[str] = []
    for a in annotations:
        if a["class_name"] not in seen:
            seen.append(a["class_name"])
    return seen


def to_export_payload(annotations: list[dict]) -> tuple[list[str], list[dict[str, Any]]]:
    """Shape expected by POST /api/training/export-dataset: a stable
    class_id per class name (first-seen order) + {class_id, points_px}."""
    names = class_names_used(annotations)
    index = {n: i for i, n in enumerate(names)}
    out = [{"class_id": index[a["class_name"]], "points_px": a["points_px"]} for a in annotations]
    return names, out
