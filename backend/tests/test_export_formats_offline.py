"""Offline test for GeoJSON -> KML export: round-trips coordinates/properties
through real XML parsing (stdlib xml.etree), not just string matching."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.export_formats import geojson_to_kml

KML_NS = "{http://www.opengis.net/kml/2.2}"


def test_point_and_polygon_roundtrip():
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [46.7, 24.7]},
                "properties": {"class": "palm_tree", "score": 0.91},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                "properties": {"class": "field", "area_ha": 2.5},
            },
        ],
    }

    kml_str = geojson_to_kml(geojson, document_name="Test Export")
    root = ET.fromstring(kml_str)  # raises if XML is malformed

    placemarks = root.findall(f".//{KML_NS}Placemark")
    assert len(placemarks) == 2

    point_pm = placemarks[0]
    assert point_pm.find(f"{KML_NS}name").text == "palm_tree"
    coords_text = point_pm.find(f"{KML_NS}Point/{KML_NS}coordinates").text
    lon, lat, alt = (float(x) for x in coords_text.split(","))
    assert abs(lon - 46.7) < 1e-9 and abs(lat - 24.7) < 1e-9

    data_values = {
        d.get("name"): d.find(f"{KML_NS}value").text
        for d in point_pm.findall(f"{KML_NS}ExtendedData/{KML_NS}Data")
    }
    assert data_values["class"] == "palm_tree"
    assert data_values["score"] == "0.91"

    poly_pm = placemarks[1]
    ring_text = poly_pm.find(f"{KML_NS}Polygon/{KML_NS}outerBoundaryIs/{KML_NS}LinearRing/{KML_NS}coordinates").text
    points = [tuple(float(v) for v in triplet.split(",")) for triplet in ring_text.split()]
    assert points[0][:2] == (0.0, 0.0)
    assert points[2][:2] == (1.0, 1.0)

    print(f"KML export: {len(placemarks)} placemarks round-tripped through real XML parsing — OK")


def test_rejects_unsupported_geometry():
    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": []}, "properties": {}}]}
    try:
        geojson_to_kml(geojson)
        raise AssertionError("expected ValueError for unsupported geometry type")
    except ValueError as exc:
        assert "LineString" in str(exc)
        print("KML export correctly rejects unsupported geometry:", exc)


if __name__ == "__main__":
    test_point_and_polygon_roundtrip()
    test_rejects_unsupported_geometry()
    print("test_export_formats_offline: PASSED")
