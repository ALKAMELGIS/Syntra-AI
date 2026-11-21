"""
Detection merging + result export.

Converts raw per-tile model outputs into mosaic-global detections, applies
cross-tile NMS, and serializes to GeoJSON (points for tree/palm-style counts,
polygons for boxes and field-boundary segmentation).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.engine.geo import GeoTransform, box_pixel_to_geo, point_pixel_to_geo, polygon_pixel_to_geo
from app.engine.nms import merge_tile_duplicates


def tile_detections_to_global(
    tile_detections: list[dict],
    tile_transform: GeoTransform,
    class_names: dict[int, str],
) -> list[dict]:
    """
    Convert one tile's raw detections (pixel-space, tile-local) into
    georeferenced records ready for cross-tile merging and export.

    Each input detection: {"box": (x1,y1,x2,y2), "score": float, "class_id": int}
    plus an optional "polygon": [(x, y), ...] in tile pixels (e.g. an oriented
    bounding box). With a polygon, that rotated outline is what gets exported
    and box_geo is its envelope, so cross-tile NMS keeps working unchanged.
    """
    out = []
    for det in tile_detections:
        x1, y1, x2, y2 = det["box"]
        polygon_px = det.get("polygon")
        if polygon_px:
            ring_px = [tuple(map(float, pt)) for pt in polygon_px]
            if ring_px[0] != ring_px[-1]:
                ring_px.append(ring_px[0])
            ring_geo = polygon_pixel_to_geo(ring_px, tile_transform)
            cx = sum(p[0] for p in ring_px[:-1]) / (len(ring_px) - 1)
            cy = sum(p[1] for p in ring_px[:-1]) / (len(ring_px) - 1)
            centroid_geo = point_pixel_to_geo(cx, cy, tile_transform)
        else:
            ring_geo = box_pixel_to_geo((x1, y1, x2, y2), tile_transform)
            centroid_geo = point_pixel_to_geo((x1 + x2) / 2, (y1 + y2) / 2, tile_transform)
        # NOTE: do not assume ring_geo[0] is the min corner and ring_geo[2] is
        # the max corner. box_pixel_to_geo() maps pixel-space corners in
        # pixel order (x1,y1)->(x2,y2), but a standard north-up affine
        # transform has a NEGATIVE pixel-height coefficient (e < 0), which
        # flips the y-axis: the geo-y of pixel-row y1 (top) ends up LARGER
        # than the geo-y of pixel-row y2 (bottom). Taking ring_geo[0]/[2]
        # verbatim then produces (xmin, ymax, xmax, ymin) instead of
        # (xmin, ymin, xmax, ymax) — silently breaking the axis-aligned-box
        # contract that nms.py's IoU math assumes (negative height -> broken
        # area/intersection -> cross-tile duplicate suppression never fires
        # for any real north-up GeoTIFF). Always derive the true min/max
        # from the actual coordinates instead of positional indices.
        geo_xs = [pt[0] for pt in ring_geo]
        geo_ys = [pt[1] for pt in ring_geo]
        out.append(
            {
                "box_geo": (min(geo_xs), min(geo_ys), max(geo_xs), max(geo_ys)),
                "polygon_geo": ring_geo,
                "centroid_geo": centroid_geo,
                "score": float(det["score"]),
                "class_id": int(det["class_id"]),
                "class_name": class_names.get(int(det["class_id"]), f"class_{det['class_id']}"),
            }
        )
    return out


def merge_and_dedupe(all_tile_detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    return merge_tile_duplicates(all_tile_detections, iou_threshold=iou_threshold)


def to_geojson_points(detections: list[dict]) -> dict:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": list(d["centroid_geo"])},
            "properties": {"class": d["class_name"], "score": round(d["score"], 4)},
        }
        for d in detections
    ]
    return {"type": "FeatureCollection", "features": features}


def to_geojson_boxes(detections: list[dict]) -> dict:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [list(map(list, d["polygon_geo"]))]},
            "properties": {"class": d["class_name"], "score": round(d["score"], 4)},
        }
        for d in detections
    ]
    return {"type": "FeatureCollection", "features": features}


def to_geojson_polygons(polygons: list[dict]) -> dict:
    """For segmentation results: polygons carry area_sqm / area_ha / perimeter_m instead of score."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [list(map(list, p["ring_geo"]))]},
            "properties": {
                "class": p.get("class_name", "field"),
                "area_sqm": round(p["area_sqm"], 2),
                "area_ha": round(p["area_sqm"] / 10_000, 4),
                "perimeter_m": round(p["perimeter_m"], 2),
            },
        }
        for p in polygons
    ]
    return {"type": "FeatureCollection", "features": features}


def export_results(
    out_dir: str | Path,
    model_id: str,
    detections: list[dict],
    geometry: str = "point",
    extra_meta: dict[str, Any] | None = None,
) -> dict:
    """Write annotated GeoJSON + a run-manifest JSON to `out_dir`; returns file paths + summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if geometry == "point":
        geojson = to_geojson_points(detections)
    elif geometry == "polygon_detection":
        geojson = to_geojson_boxes(detections)
    elif geometry == "polygon_segmentation":
        geojson = to_geojson_polygons(detections)
    else:
        raise ValueError(f"Unknown geometry kind: {geometry}")

    from app.engine.measure import measure_features
    measure_features(geojson)
    geojson_path = out_dir / f"{model_id}.geojson"
    geojson_path.write_text(json.dumps(geojson, indent=2))

    manifest = {
        "model_id": model_id,
        "count": len(detections),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "geojson_file": geojson_path.name,
        **(extra_meta or {}),
    }
    manifest_path = out_dir / f"{model_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {"geojson_path": str(geojson_path), "manifest_path": str(manifest_path), "manifest": manifest}
