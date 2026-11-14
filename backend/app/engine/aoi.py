"""
Area of interest (AOI): run an analysis on part of a scene only.

The map's drawing tools produce the AOI as a WGS84 GeoJSON polygon. Here it
is transformed into the raster's CRS, clipped to the raster, and turned into
three things the engine uses:

* a pixel **window** (the AOI's bounding box) so tiling never even reads the
  rest of a multi-GB scene,
* a **tile filter** that skips tiles inside the window but outside the
  polygon (a diagonal AOI has empty corners),
* a **detection filter** that keeps only detections whose centre falls
  inside the polygon, so overlap at the edge doesn't leak results.

shapely and pyproj are imported lazily.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _apply(tf, col: float, row: float) -> tuple[float, float]:
    """Affine pixel -> map, written out (affine's `*` on tuples is deprecated)."""
    return tf.a * col + tf.b * row + tf.c, tf.d * col + tf.e * row + tf.f


def _invert_apply(tf, x: float, y: float) -> tuple[float, float]:
    det = tf.a * tf.e - tf.b * tf.d
    dx, dy = x - tf.c, y - tf.f
    return (tf.e * dx - tf.b * dy) / det, (-tf.d * dx + tf.a * dy) / det


class AoiError(ValueError):
    """The AOI is malformed, or doesn't overlap the image."""


@dataclass
class AoiInRaster:
    geom: Any                                  # shapely geometry in the raster CRS
    window: tuple[int, int, int, int]          # col_off, row_off, width, height (pixels)
    area_m2: float | None                      # AOI area on the ground, when computable

    def tile_intersects(self, transform, col_off: int, row_off: int, w: int, h: int) -> bool:
        from shapely.geometry import box

        x0, y0 = _apply(transform, col_off, row_off)
        x1, y1 = _apply(transform, col_off + w, row_off + h)
        return self.geom.intersects(box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    def contains_point(self, x: float, y: float) -> bool:
        from shapely.geometry import Point

        return self.geom.covers(Point(x, y))


def _geometry_of(aoi: dict) -> dict:
    if not isinstance(aoi, dict):
        raise AoiError("AOI must be a GeoJSON object.")
    t = aoi.get("type")
    if t == "Feature":
        return _geometry_of(aoi.get("geometry") or {})
    if t == "FeatureCollection":
        feats = aoi.get("features") or []
        if len(feats) != 1:
            raise AoiError("AOI FeatureCollection must hold exactly one feature.")
        return _geometry_of(feats[0])
    if t in ("Polygon", "MultiPolygon"):
        return aoi
    raise AoiError(f"AOI must be a Polygon or MultiPolygon, got {t!r}.")


def prepare_aoi(raster_path: str, aoi_geojson: dict) -> AoiInRaster:
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import box, shape
    from shapely.ops import transform as shp_transform

    geom_wgs = shape(_geometry_of(aoi_geojson))
    if geom_wgs.is_empty:
        raise AoiError("AOI is empty.")
    if not geom_wgs.is_valid:
        geom_wgs = geom_wgs.buffer(0)   # repair self-intersections from freehand drawing

    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise AoiError("This image has no CRS, so a map-drawn AOI can't be placed on it.")
        tf = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        geom = shp_transform(lambda x, y, z=None: tf.transform(x, y), geom_wgs)
        b = ds.bounds
        geom = geom.intersection(box(b.left, b.bottom, b.right, b.top))
        if geom.is_empty:
            raise AoiError("The drawn area doesn't overlap this image.")

        minx, miny, maxx, maxy = geom.bounds
        cols, rows = zip(*(_invert_apply(ds.transform, x, y) for x, y in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy))))
        c0 = max(0, int(math.floor(min(cols))))
        r0 = max(0, int(math.floor(min(rows))))
        c1 = min(ds.width, int(math.ceil(max(cols))))
        r1 = min(ds.height, int(math.ceil(max(rows))))
        if c1 <= c0 or r1 <= r0:
            raise AoiError("The drawn area is smaller than one pixel.")

        area = None
        if ds.crs.is_projected:
            try:
                area = float(geom.area) * float(ds.crs.linear_units_factor[1]) ** 2
            except Exception:
                area = float(geom.area)
        else:
            from pyproj import Geod
            area = abs(Geod(ellps="WGS84").geometry_area_perimeter(geom)[0])

    return AoiInRaster(geom=geom, window=(c0, r0, c1 - c0, r1 - r0), area_m2=area)
