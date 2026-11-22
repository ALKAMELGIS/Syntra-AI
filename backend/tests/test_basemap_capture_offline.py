"""A basemap area as a GeoTIFF: stitched and georeferenced right, and only from
sources whose terms allow keeping tiles, on public hosts."""
import io

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from PIL import Image  # noqa: E402


def _tile(_url):
    b = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), 120, np.uint8)).save(b, "PNG")
    return b.getvalue()


def test_capture_writes_a_web_mercator_geotiff_over_the_box(tmp_path):
    from rasterio.warp import transform_bounds

    from app.engine import basemap_capture as bc

    box = (55.27, 25.19, 55.28, 25.20)                    # about 1 km, Dubai
    p = bc.plan(box, 17)
    assert 4 <= p["tiles"] <= 40 and 0.9 < p["gsd_m"] < 1.2
    out = tmp_path / "a.tif"
    info = bc.capture("https://tiles.example/{z}/{x}/{y}.png", box, 17, out, fetch=_tile)
    assert info["missing"] == 0
    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 3857 and ds.count == 3
        w, s, e, n = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        assert abs(w - box[0]) < 2e-5 and abs(e - box[2]) < 2e-5 and abs(s - box[1]) < 2e-5 and abs(n - box[3]) < 2e-5
        assert (ds.width, ds.height) == (p["width"], p["height"])


def test_too_many_tiles_and_bad_boxes_are_refused(tmp_path):
    from app.engine import basemap_capture as bc

    with pytest.raises(bc.CaptureError, match="tiles"):
        bc.capture("https://t.example/{z}/{x}/{y}", (55.0, 25.0, 55.5, 25.5), 19, tmp_path / "x.tif", fetch=_tile)
    with pytest.raises(bc.CaptureError):
        bc.plan((55.3, 25.2, 55.2, 25.3), 17)


def test_providers_whose_terms_forbid_it_and_private_hosts_are_refused(monkeypatch):
    from app.engine import basemap_capture as bc

    for url in ("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "https://ecn.t0.tiles.virtualearth.net/tiles/a{q}.jpeg?{z}{x}{y}"):
        with pytest.raises(bc.CaptureError, match="terms"):
            bc.check_url(url)
    monkeypatch.setattr(bc.socket, "getaddrinfo", lambda h, p: [(0, 0, 0, "", ("127.0.0.1", 0))])
    with pytest.raises(bc.CaptureError, match="public"):
        bc.check_url("https://tiles.example/{z}/{x}/{y}.png")
    monkeypatch.setattr(bc.socket, "getaddrinfo", lambda h, p: [(0, 0, 0, "", ("93.184.216.34", 0))])
    assert bc.check_url("https://tiles.example/{z}/{x}/{y}.png")
    with pytest.raises(bc.CaptureError, match=r"\{z\}"):
        bc.check_url("https://tiles.example/tile.png")


def test_api_adds_the_area_to_the_library(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.engine import basemap_capture as bc

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "images")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.delenv("SYNTRA_AUTH", raising=False)
    real = bc.capture
    monkeypatch.setattr(bc, "capture", lambda t, b, z, out: real(t, b, z, out, fetch=_tile))
    from app.main import app

    c = TestClient(app)
    assert c.get("/api/images/basemap-sources").json()["sources"][0]["id"] == "usgs"
    r = c.post("/api/images/from-basemap", json={"source": "usgs", "bbox": [-77.04, 38.89, -77.03, 38.90], "zoom": 16, "name": "dc"})
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["filename"].startswith("dc-z16-") and e["source"] == "basemap:usgs" and e["bounds_wgs84"]
    assert c.post("/api/images/from-basemap", json={"source": "usgs", "bbox": [-77.04, 38.89, -77.03, 38.90], "zoom": 19}).status_code == 422
    assert c.post("/api/images/from-basemap", json={"source": "custom", "url": "https://mt0.google.com/vt?x={x}&y={y}&z={z}",
                                                    "bbox": [-77.04, 38.89, -77.03, 38.90], "zoom": 16}).status_code == 422
