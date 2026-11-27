"""Add layer from a file (POST /api/vector/read) and the map's band / stretch choice for tiles."""
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SYNTRA_AUTH", "off")
    from app.main import app
    return TestClient(app)


def test_refuses_an_unknown_file_type(client):
    r = client.post("/api/vector/read", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and ".zip" in r.json()["detail"]


def test_refuses_a_zip_that_points_outside_itself(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.shp", b"x")
    r = client.post("/api/vector/read", files={"file": ("layer.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 400 and "outside" in r.json()["detail"]


def test_zip_without_a_vector_file_is_explained(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", b"x")
    r = client.post("/api/vector/read", files={"file": ("layer.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 400 and "No .shp" in r.json()["detail"]


def test_nothing_is_left_in_the_data_folder(client):
    from app.core.config import settings
    client.post("/api/vector/read", files={"file": ("notes.txt", b"x", "text/plain")})
    tmp = settings.DATA_DIR / "tmp"
    assert not tmp.exists() or not any(tmp.iterdir())


def test_reads_projected_geojson_into_wgs84(client):
    pytest.importorskip("geopandas")
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32639"}},
          "features": [{"type": "Feature", "properties": {"name": "pier", "len": 12.5},
                        "geometry": {"type": "Point", "coordinates": [520000, 2893000]}}]}
    r = client.post("/api/vector/read", files={"file": ("piers.geojson", json.dumps(fc).encode(), "application/geo+json")})
    assert r.status_code == 200, r.text
    body = r.json()
    lon, lat = body["geojson"]["features"][0]["geometry"]["coordinates"]
    assert 51 < lon < 52 and 26 < lat < 27 and "32639" in body["source_crs"]
    assert body["geojson"]["features"][0]["properties"] == {"name": "pier", "len": 12.5}


@pytest.mark.parametrize("bands,stretch,ok", [
    ("3,2,1", "2-98", True), ("1", None, True), (None, "0-100", True),
    ("3,2", None, False), ("0,1,2", None, False), (None, "98-2", False), (None, "abc", False),
])
def test_display_choice_is_checked(bands, stretch, ok):
    from app.engine.tiles import parse_display
    if ok:
        b, s = parse_display(bands, stretch)
        assert b is None or len(b) == 3
    else:
        with pytest.raises(ValueError):
            parse_display(bands, stretch)


def test_bad_display_choice_is_a_400_before_reading_anything(client):
    r = client.get("/api/images/nope/tiles/5/10/10.png?bands=9,x,1")
    assert r.status_code == 400
