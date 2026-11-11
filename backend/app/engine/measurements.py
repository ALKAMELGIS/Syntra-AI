"""
Geometry measurements for field-boundary segmentation (spec Section 1,
Model 06: "Calculate field area in square meters and hectares. Extract
perimeter.") and any other polygon output.

Pure NumPy + the shoelace formula for planar area — correct for any
projected (metric) CRS, which is what GeoTIFFs for this kind of GSD range
are normally supplied in. If a raster is in a geographic (lon/lat) CRS
instead, these functions will silently return values in degree-units
squared rather than square meters — `assert_projected_hint()` below is a
best-effort sanity check, not a substitute for the caller knowing its CRS.

`mask_to_geo_polygons` uses OpenCV's contour finder, which — unlike
rasterio/torch/ultralytics — IS installed in this build sandbox, so it is
genuinely exercised in tests/test_measurements_offline.py, not just a stub.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.engine.geo import GeoTransform, polygon_pixel_to_geo


def shoelace_area(ring: list[tuple[float, float]]) -> float:
    """Unsigned planar area of a simple polygon ring (closed or open)."""
    pts = np.asarray(ring, dtype=np.float64)
    if pts.shape[0] >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]  # drop the closing duplicate point, shoelace handles wraparound itself
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0)


def polygon_perimeter(ring: list[tuple[float, float]]) -> float:
    """Sum of edge lengths of a simple polygon ring (closed or open)."""
    pts = np.asarray(ring, dtype=np.float64)
    if pts.shape[0] >= 2 and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])  # close the ring for perimeter purposes
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))


def assert_projected_hint(transform: GeoTransform) -> None:
    """
    Heuristic only: pixel sizes under ~0.0001 (as degrees-per-pixel would be
    for sub-meter GSD) strongly suggest a geographic CRS slipped in where a
    projected one was expected. Not a reliable CRS check — just a guardrail
    against a silent unit mistake.
    """
    if abs(transform.a) < 1e-4:
        raise ValueError(
            f"Pixel size {transform.a} looks like degrees, not meters — area/perimeter here assume a "
            "projected (metric) CRS. Reproject the raster or pass gsd in meters explicitly."
        )


@dataclass
class FieldPolygon:
    ring_geo: list[tuple[float, float]]
    area_sqm: float
    perimeter_m: float
    area_px: int


def mask_to_geo_polygons(
    mask: np.ndarray,
    transform: GeoTransform,
    min_area_px: int = 16,
    simplify_epsilon_px: float = 1.0,
) -> list[FieldPolygon]:
    """
    Convert a binary segmentation mask (H, W), values in {0, 1} or bool,
    tile-local pixel space, into georeferenced field-boundary polygons with
    real-world area (sqm) and perimeter (m).

    Requires opencv (cv2) for contour extraction — NOT torch/rasterio, so
    this runs even in environments that can't install the full ML stack.
    """
    import cv2  # available in this sandbox; kept as a local import for symmetry with the rest of the codebase

    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[FieldPolygon] = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px:
            continue
        if simplify_epsilon_px > 0:
            cnt = cv2.approxPolyDP(cnt, simplify_epsilon_px, closed=True)

        ring_px = [(float(p[0][0]), float(p[0][1])) for p in cnt]
        if len(ring_px) < 3:
            continue
        ring_px.append(ring_px[0])  # close the ring

        ring_geo = polygon_pixel_to_geo(ring_px, transform)
        polygons.append(
            FieldPolygon(
                ring_geo=ring_geo,
                area_sqm=shoelace_area(ring_geo),
                perimeter_m=polygon_perimeter(ring_geo),
                area_px=int(area_px),
            )
        )
    return polygons
