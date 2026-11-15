"""
Persistence + lon/lat <-> pixel conversion for per-image object annotations.

No rasterio dependency, so this is importable and fully unit-testable
anywhere. Labels are stored per class as GeoJSON in the data folder's
`classes/` library (see app.core.layout).

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
    """The file labels lived in before the per-class library (read once, then moved)."""
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


# ---------------------------------------------------------------- the library
# Labels are kept per class, as GeoJSON with real coordinates (app.core.layout):
#   classes/<class>/labels/<key>.geojson   objects of that class on image <key>
#   classes/_areas/<key>.geojson           training ("complete") and accuracy areas
# The pixel ring is kept in each feature's properties, so nothing has to be
# re-projected to tile the image.

_ROLE = {"__train_area__": "complete", "__check_area__": "accuracy"}
_ROLE_CLASS = {v: k for k, v in _ROLE.items()}


def image_key_for(image_path: str | Path) -> str:
    import hashlib
    import re

    from app.core.layout import image_key

    p = Path(image_path)
    if re.fullmatch(r"[0-9a-f]{8}", p.parent.name):
        return p.parent.name
    m = re.match(r"^([0-9a-f]{8})-?[0-9a-f]{4}", p.name)
    if m:
        return image_key(m.group(1))
    return hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:8]


def _files(image_path) -> list[Path]:
    from app.core.layout import classes_dir

    key = image_key_for(image_path)
    root = classes_dir()
    if not root.is_dir():
        return []
    return sorted(root.glob(f"*/labels/{key}.geojson")) + sorted(root.glob(f"_areas/{key}.geojson"))


def _to_feature(a: dict) -> dict:
    ring = [list(map(float, p)) for p in a["points_lonlat"]]
    if ring and ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    props = {k: v for k, v in a.items() if k not in ("points_lonlat",)}
    if a["class_name"] in _ROLE:
        props["role"] = _ROLE[a["class_name"]]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": props}


def _from_feature(f: dict) -> dict | None:
    props = dict(f.get("properties") or {})
    try:
        ring = f["geometry"]["coordinates"][0]
    except (KeyError, IndexError, TypeError):
        return None
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if "class_name" not in props:
        props["class_name"] = _ROLE_CLASS.get(props.get("role"), props.get("class", "object"))
    props.pop("role", None)
    props["points_lonlat"] = ring
    return props


def load_annotations(image_path: str | Path) -> list[dict]:
    sc = sidecar_path(image_path)
    if sc.exists():                                          # labels from before the library: move them once
        try:
            legacy = json.loads(sc.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            legacy = None
        if isinstance(legacy, list):
            have = {a["id"] for a in _load_library(image_path)}
            save_annotations(image_path, _load_library(image_path) + [a for a in legacy if a.get("id") not in have])
            sc.rename(sc.with_name(sc.name + ".moved"))
    return _load_library(image_path)


def _load_library(image_path) -> list[dict]:
    out: list[dict] = []
    for f in _files(image_path):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for feat in data.get("features", []):
            a = _from_feature(feat)
            if a is not None:
                out.append(a)
    out.sort(key=lambda a: (a.get("order", 1e9), a.get("created_at") or ""))
    return out


def save_annotations(image_path: str | Path, annotations: list[dict]) -> None:
    from app.core.layout import AREAS, class_dir, write_json
    from app.engine.shapes import shape_for_class

    key = image_key_for(image_path)
    groups: dict[str, list[dict]] = {}
    for i, a in enumerate(annotations):
        a = {**a, "order": i}                                 # keeps the drawing order across class files
        folder = AREAS if a["class_name"] in _ROLE else a["class_name"]
        groups.setdefault(folder, []).append(a)
    written = set()
    for folder, items in groups.items():
        d = class_dir(folder)
        target = d / f"{key}.geojson" if folder == AREAS else d / "labels" / f"{key}.geojson"
        write_json(target, {"type": "FeatureCollection", "image": key, "source": Path(image_path).name,
                            "features": [_to_feature(a) for a in items]})
        written.add(target)
        if folder != AREAS and not (d / "class.json").exists():
            write_json(d / "class.json", {"name": folder, "shape": shape_for_class(folder), "target": 1500})
    for f in _files(image_path):                             # classes with no labels left on this image
        if f not in written:
            f.unlink(missing_ok=True)


def add_annotation(image_path: str | Path, class_name: str, points_lonlat: list[list[float]],
                    bounds: dict, width: int, height: int, extra: dict | None = None) -> dict:
    if len(points_lonlat) < 3:
        raise ValueError("An annotation needs at least 3 points (a closed ring).")
    points_px = [list(lonlat_to_pixel(lon, lat, bounds, width, height)) for lon, lat in points_lonlat]
    xs, ys = [p[0] for p in points_px], [p[1] for p in points_px]
    if max(xs) <= 0 or min(xs) >= width or max(ys) <= 0 or min(ys) >= height:
        raise ValueError("This shape is outside the image. Draw it on the image")
    entry = {
        "id": str(uuid.uuid4()),
        "class_name": class_name.strip() or "object",
        "points_lonlat": points_lonlat,
        "points_px": points_px,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **{k: str(v)[:32] for k, v in (extra or {}).items()},
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


# Two reserved "classes" mark areas rather than objects:
#   training area  everything inside is labelled; tiles are cut only from here,
#                  so an unlabelled object is never taught as background
#   accuracy area  labelled, but never trained on; models are scored against it
TRAIN_AREA = "__train_area__"
CHECK_AREA = "__check_area__"
AREA_CLASSES = {TRAIN_AREA, CHECK_AREA}


def is_area(a: dict) -> bool:
    return a.get("class_name") in AREA_CLASSES


def areas(annotations: list[dict], kind: str) -> list[dict]:
    return [a for a in annotations if a.get("class_name") == kind]


def _inside_any(points_px: list, polys: list[list]) -> bool:
    """Centroid of the shape inside any of the polygons (pixel coordinates)."""
    import numpy as np

    pts = np.asarray(points_px, dtype=float)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    for poly in polys:
        p = np.asarray(poly, dtype=float)
        x, y = p[:, 0], p[:, 1]
        inside = False
        j = len(p) - 1
        for i in range(len(p)):
            if (y[i] > cy) != (y[j] > cy) and cx < (x[j] - x[i]) * (cy - y[i]) / (y[j] - y[i] + 1e-12) + x[i]:
                inside = not inside
            j = i
        if inside:
            return True
    return False


def objects_for_training(annotations: list[dict]) -> list[dict]:
    """Object labels that may be trained on: not areas, not inside an accuracy area."""
    checks = [a["points_px"] for a in areas(annotations, CHECK_AREA)]
    return [a for a in annotations if not is_area(a) and not (checks and _inside_any(a["points_px"], checks))]


def objects_for_test(annotations: list[dict], names: list[str]) -> list[dict]:
    """Objects inside accuracy areas, as {class_id, points_px}: the test split's labels."""
    checks = [a["points_px"] for a in areas(annotations, CHECK_AREA)]
    index = {n: i for i, n in enumerate(names)}
    return [{"class_id": index[a["class_name"]], "points_px": a["points_px"]} for a in annotations
            if not is_area(a) and a["class_name"] in index and checks and _inside_any(a["points_px"], checks)]


def class_names_used(annotations: list[dict]) -> list[str]:
    seen: list[str] = []
    for a in annotations:
        if is_area(a):
            continue
        if a["class_name"] not in seen:
            seen.append(a["class_name"])
    return seen


def to_export_payload(annotations: list[dict]) -> tuple[list[str], list[dict[str, Any]]]:
    """Shape expected by POST /api/training/export-dataset: a stable
    class_id per class name (first-seen order) + {class_id, points_px}."""
    objs = objects_for_training(annotations)
    names = class_names_used(objs)
    index = {n: i for i, n in enumerate(names)}
    out = [{"class_id": index[a["class_name"]], "points_px": a["points_px"]} for a in objs]
    return names, out
