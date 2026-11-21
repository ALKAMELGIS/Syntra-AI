"""
Geospatial coordinate reconstruction.

Implements the standard GDAL/rasterio affine convention so detections made in
pixel space on a tile can be mapped back to real-world (CRS) coordinates:

    x_geo = a * col + b * row + c
    y_geo = d * col + e * row + f

Deliberately implemented with plain tuples (no hard dependency on the
`affine` package) so this module — and anything built on it — is importable
and unit-testable even where rasterio/affine aren't installed. When rasterio
*is* available, `from_rasterio_transform` adapts its `Affine` object directly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeoTransform:
    a: float  # pixel width
    b: float  # row rotation (usually 0)
    c: float  # x-origin (top-left)
    d: float  # column rotation (usually 0)
    e: float  # pixel height (negative, north-up)
    f: float  # y-origin (top-left)

    def pixel_to_geo(self, col: float, row: float) -> tuple[float, float]:
        x = self.a * col + self.b * row + self.c
        y = self.d * col + self.e * row + self.f
        return x, y

    def geo_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        det = self.a * self.e - self.b * self.d
        if det == 0:
            raise ValueError("Degenerate affine transform (zero determinant).")
        dx, dy = x - self.c, y - self.f
        col = (self.e * dx - self.b * dy) / det
        row = (-self.d * dx + self.a * dy) / det
        return col, row

    def offset(self, col_off: int, row_off: int) -> "GeoTransform":
        """Transform for a sub-window (tile) that starts at (col_off, row_off) in the parent raster."""
        new_c, new_f = self.pixel_to_geo(col_off, row_off)
        return GeoTransform(self.a, self.b, new_c, self.d, self.e, new_f)

    @classmethod
    def from_rasterio_transform(cls, transform) -> "GeoTransform":
        """Build from a rasterio/affine `Affine` instance (has attrs a..f, or is a 6/9-tuple)."""
        if hasattr(transform, "a"):
            return cls(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
        a, b, c, d, e, f = transform[:6]
        return cls(a, b, c, d, e, f)

    @classmethod
    def identity_with_gsd(cls, gsd: float, origin_x: float = 0.0, origin_y: float = 0.0) -> "GeoTransform":
        """North-up transform with a given ground sample distance (meters/px), used by offline tests."""
        return cls(gsd, 0.0, origin_x, 0.0, -gsd, origin_y)


def box_pixel_to_geo(box_xyxy: tuple[float, float, float, float], transform: GeoTransform) -> list[tuple[float, float]]:
    """Convert a pixel-space bbox (x1, y1, x2, y2) to a geo-space polygon ring (closed)."""
    x1, y1, x2, y2 = box_xyxy
    corners_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    return [transform.pixel_to_geo(c, r) for c, r in corners_px]


def point_pixel_to_geo(x: float, y: float, transform: GeoTransform) -> tuple[float, float]:
    return transform.pixel_to_geo(x, y)


def polygon_pixel_to_geo(ring_px: list[tuple[float, float]], transform: GeoTransform) -> list[tuple[float, float]]:
    return [transform.pixel_to_geo(c, r) for c, r in ring_px]


WGS84 = "EPSG:4326"


def reproject_detections_to_wgs84(detections: list[dict], source_crs: str | None) -> tuple[list[dict], str | None]:
    """Reproject merged detections from the raster's CRS to WGS84 lon/lat.

    GeoJSON (RFC 7946), KML and Leaflet all expect WGS84. Cross-tile NMS runs
    *before* this, in the raster's own CRS, where projected units are metres
    and IoU is meaningful.

    Returns (detections, output_crs). With no source CRS the coordinates are
    left untouched and output_crs is None: there is no honest way to place an
    unreferenced raster on Earth, so we don't pretend to.
    """
    if not source_crs:
        return detections, None
    from pyproj import CRS, Transformer  # lazy: GDAL/PROJ stack

    src = CRS.from_user_input(source_crs)
    if src == CRS.from_epsg(4326):
        return detections, WGS84
    tf = Transformer.from_crs(src, CRS.from_epsg(4326), always_xy=True)

    def ring(points):
        xs, ys = zip(*points)
        lons, lats = tf.transform(xs, ys)
        return [(float(lo), float(la)) for lo, la in zip(lons, lats)]

    out = []
    for d in detections:
        d = dict(d)
        if d.get("polygon_geo"):
            d["polygon_geo"] = ring(d["polygon_geo"])
            lons = [pt[0] for pt in d["polygon_geo"]]
            lats = [pt[1] for pt in d["polygon_geo"]]
            d["box_geo"] = (min(lons), min(lats), max(lons), max(lats))
        if d.get("centroid_geo"):
            d["centroid_geo"] = ring([d["centroid_geo"]])[0]
        out.append(d)
    return out, WGS84
