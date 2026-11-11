"""
GIS export formats beyond GeoJSON — spec Phase 5.

KML is plain XML, so it's implemented here with the stdlib only (no
simplekml/fiona/GDAL dependency) and is fully offline-testable.

Shapefile export (.shp/.shx/.dbf) is implemented separately in
app/engine/shapefile_export.py — it's a binary format, so it's kept apart
from this XML-based module. See that file's docstring for how its
correctness was validated without a real GIS tool available in this
sandbox (a from-scratch independent reader, plus a spec-derived invariant
check on the file's self-declared length).
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


# Shapefile export: see app/engine/shapefile_export.write_shapefile(),
# exposed via GET /api/results/{job_id}/{model_id}/shapefile (zipped
# .shp+.shx+.dbf). If a real GIS stack (pyshp/fiona/GDAL) is available in
# your deployment, that remains a fine alternative — this hand-rolled
# version exists so shapefile export doesn't require installing one.
