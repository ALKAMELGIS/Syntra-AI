"""
GIS export formats beyond GeoJSON — spec Phase 5.

KML is plain XML, so it's implemented here with the stdlib only (no
simplekml/fiona/GDAL dependency) and is fully offline-testable. Shapefile
export genuinely needs a binary-format writer (.shp/.shx/.dbf) or a library
like `pyshp`/`fiona`; that's left as a documented follow-up (see bottom of
this file) rather than hand-rolling a binary format without a way to verify
it against a real GIS tool in this sandbox.
"""
from __future__ import annotations

from xml.sax.saxutils import escape


def _coords_to_kml(ring: list) -> str:
    # KML wants "lon,lat[,alt]" triples separated by whitespace.
    return " ".join(f"{pt[0]},{pt[1]},0" for pt in ring)


def _placemark_point(name: str, lon: float, lat: float, props: dict) -> str:
    extended = "".join(
        f'<Data name="{escape(str(k))}"><value>{escape(str(v))}</value></Data>' for k, v in props.items()
    )
    return (
        f"<Placemark><name>{escape(name)}</name>"
        f"<ExtendedData>{extended}</ExtendedData>"
        f"<Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>"
    )


def _placemark_polygon(name: str, ring: list, props: dict) -> str:
    extended = "".join(
        f'<Data name="{escape(str(k))}"><value>{escape(str(v))}</value></Data>' for k, v in props.items()
    )
    coords = _coords_to_kml(ring)
    return (
        f"<Placemark><name>{escape(name)}</name>"
        f"<ExtendedData>{extended}</ExtendedData>"
        f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates>"
        f"</LinearRing></outerBoundaryIs></Polygon></Placemark>"
    )


def geojson_to_kml(geojson: dict, document_name: str = "Syntra AI Export") -> str:
    """
    Convert a FeatureCollection of Point or Polygon features (as produced by
    app/engine/postprocess.py) into a KML document string. Each feature's
    `properties` dict is carried over as KML ExtendedData so class/score/
    area/perimeter survive the round trip into Google Earth / QGIS.
    """
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("Expected a GeoJSON FeatureCollection.")

    placemarks = []
    for i, feat in enumerate(geojson.get("features", [])):
        geom = feat.get("geometry", {})
        props = feat.get("properties", {}) or {}
        label = str(props.get("class", f"feature_{i}"))

        if geom.get("type") == "Point":
            lon, lat = geom["coordinates"]
            placemarks.append(_placemark_point(label, lon, lat, props))
        elif geom.get("type") == "Polygon":
            ring = geom["coordinates"][0]
            placemarks.append(_placemark_polygon(label, ring, props))
        else:
            raise ValueError(f"Unsupported geometry type for KML export: {geom.get('type')}")

    body = "".join(placemarks)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>{escape(document_name)}</name>{body}</Document></kml>"
    )


# --- Shapefile export (not implemented here) ---------------------------------
#
# Writing .shp/.shx/.dbf by hand is error-prone to get byte-exact without a
# real GIS tool to validate against, which this sandbox doesn't have. Use
# `pyshp` (pure Python, no GDAL) once deployed with network access:
#
#     import shapefile
#     with shapefile.Writer(out_path, shapeType=shapefile.POINT) as w:
#         w.field("class", "C")
#         w.field("score", "N", decimal=4)
#         for feat in geojson["features"]:
#             lon, lat = feat["geometry"]["coordinates"]
#             w.point(lon, lat)
#             w.record(feat["properties"]["class"], feat["properties"]["score"])
