"""Results must leave the engine in WGS84 lon/lat (GeoJSON, KML and Leaflet
all assume it), whatever projection the source GeoTIFF used."""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from app.engine.geo import reproject_detections_to_wgs84
from app.engine.raster_io import count_tiles

pyproj = pytest.importorskip("pyproj")


def _det(ring, centroid):
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return {"polygon_geo": ring, "centroid_geo": centroid, "box_geo": (min(xs), min(ys), max(xs), max(ys)),
            "score": 0.9, "class_id": 0, "class_name": "ship"}


def test_utm_40n_detection_lands_in_dubai():
    # 10 m square near Dubai Marina in UTM zone 40N
    ring = [(323000, 2774000), (323010, 2774000), (323010, 2773990), (323000, 2773990), (323000, 2774000)]
    [d], crs = reproject_detections_to_wgs84([_det(ring, (323005, 2773995))], "EPSG:32640")
    assert crs == "EPSG:4326"
    lon, lat = d["centroid_geo"]
    assert 55.24 < lon < 55.26 and 25.06 < lat < 25.08
    xmin, ymin, xmax, ymax = d["box_geo"]
    assert xmin < xmax and ymin < ymax
    # ~10 m wide at this latitude is ~1e-4 degrees of longitude
    assert 0.8e-4 < xmax - xmin < 1.2e-4
    assert d["polygon_geo"][0] == d["polygon_geo"][-1]


def test_wgs84_input_is_left_alone():
    ring = [(55.0, 25.0), (55.1, 25.0), (55.1, 25.1), (55.0, 25.0)]
    dets = [_det(ring, (55.05, 25.05))]
    out, crs = reproject_detections_to_wgs84(dets, "EPSG:4326")
    assert crs == "EPSG:4326" and out[0]["polygon_geo"] == ring


def test_unreferenced_raster_is_not_faked():
    ring = [(0, 0), (10, 0), (10, 10), (0, 0)]
    out, crs = reproject_detections_to_wgs84([_det(ring, (5, 5))], None)
    assert crs is None and out[0]["polygon_geo"] == ring


def test_count_tiles_matches_grid():
    # 1920x1080 at 1024 tiles with 20% overlap -> stride 819 -> 3 cols x 2 rows
    assert count_tiles(1920, 1080, 1024, 0.2) == 6
    assert count_tiles(500, 500, 1024, 0.2) == 1


def test_shapefile_zip_includes_prj_for_wgs84_results(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.delenv("SYNTRA_API_KEY", raising=False)
    run = tmp_path / "results" / "job1"
    run.mkdir(parents=True)
    poly = [[55.0, 25.0], [55.001, 25.0], [55.001, 25.001], [55.0, 25.0]]
    (run / "ship_detection.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [poly]}, "properties": {"class": "ship", "score": 0.9}},
    ]}))
    (run / "ship_detection.manifest.json").write_text(json.dumps({"crs": "EPSG:4326"}))

    r = TestClient(app).get("/api/results/job1/ship_detection/shapefile")
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "ship_detection.prj" in names
