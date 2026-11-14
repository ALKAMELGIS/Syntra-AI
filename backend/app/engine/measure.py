"""
Measurements written onto every result object, so a GIS layer arrives with
the numbers an analyst would otherwise compute by hand:

  area_m2, perimeter_m   geodesic, on the WGS84 ellipsoid
  length_m, width_m      sides of the smallest rotated rectangle around it
  orientation_deg        bearing of the long side, 0-180, clockwise from north
  lon, lat               centroid

Lengths are measured in a local equidistant projection centred on each
object, so they are true metres anywhere on Earth.
"""
from __future__ import annotations

import math


def _measure_polygon(coords: list) -> dict:
    from pyproj import Geod, Transformer
    from shapely.geometry import Polygon
    from shapely.ops import transform

    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return {}
    c = poly.centroid
    geod = Geod(ellps="WGS84")
    area, perim = geod.geometry_area_perimeter(poly)
    local = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +datum=WGS84 +units=m",
                                 always_xy=True)
    p = transform(local.transform, poly)
    rect = p.minimum_rotated_rectangle
    pts = list(rect.exterior.coords) if rect.geom_type == "Polygon" else []
    length = width = orient = None
    if len(pts) >= 4:
        e1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        e2 = (pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
        l1, l2 = math.hypot(*e1), math.hypot(*e2)
        long_e = e1 if l1 >= l2 else e2
        length, width = max(l1, l2), min(l1, l2)
        orient = math.degrees(math.atan2(long_e[0], long_e[1])) % 180      # from north, clockwise
    out = {"area_m2": round(abs(area), 2), "perimeter_m": round(perim, 2), "lon": round(c.x, 7), "lat": round(c.y, 7)}
    if length is not None:
        out.update(length_m=round(length, 2), width_m=round(width, 2), orientation_deg=round(orient, 1))
    return out


def measure_features(fc: dict) -> dict:
    """Adds the measurements to each feature's properties (in place) and returns fc.
    Values the producer already wrote (e.g. a segmentation area) are kept."""
    for f in fc.get("features", []):
        g = f.get("geometry") or {}
        props = f.setdefault("properties", {})
        try:
            if g.get("type") == "Polygon":
                m = _measure_polygon(g["coordinates"][0])
            elif g.get("type") == "MultiPolygon":
                parts = [_measure_polygon(pg[0]) for pg in g["coordinates"]]
                big = max(parts, key=lambda d: d.get("area_m2", 0)) if parts else {}
                m = {**big, "area_m2": round(sum(d.get("area_m2", 0) for d in parts), 2),
                     "perimeter_m": round(sum(d.get("perimeter_m", 0) for d in parts), 2)}
            elif g.get("type") == "Point":
                m = {"lon": round(g["coordinates"][0], 7), "lat": round(g["coordinates"][1], 7)}
            else:
                m = {}
        except Exception:
            m = {}
        for k, v in m.items():
            props.setdefault(k, v)
    return fc
