"""Remote sensing: index evalscripts, rendering, per-feature values (Sentinel Hub mocked)."""
import io
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from app.engine import sentinel as s2  # noqa: E402

AOI = {"type": "Polygon", "coordinates": [[[55.70, 24.20], [55.71, 24.20], [55.71, 24.21], [55.70, 24.21], [55.70, 24.20]]]}


def _tiff(w, h, value=0.5, cloud_rows=0):
    arr = np.full((h, w), value, "float32")
    ok = np.ones((h, w), "float32")
    ok[:cloud_rows] = 0
    buf = io.BytesIO()
    with rasterio.MemoryFile() as m:
        with m.open(driver="GTiff", width=w, height=h, count=2, dtype="float32") as ds:
            ds.write(arr, 1)
            ds.write(ok, 2)
        buf.write(m.read())
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setenv("SENTINELHUB_INSTANCE_ID", "test-instance")
    calls = []

    def fake_get(url, params, cache=True):
        if not s2.configured():
            raise s2.NotConfigured("not configured")
        calls.append(params)
        if params.get("SERVICE") == "WFS":
            return json.dumps({"features": [{"properties": {"date": "2026-11-15", "cloudCoverPercentage": 0}},
                                             {"properties": {"date": "2026-11-10", "cloudCoverPercentage": 12.5}}]}).encode()
        return _tiff(int(params["WIDTH"]), int(params["HEIGHT"]), 0.5, cloud_rows=int(params["HEIGHT"]) // 4)
    monkeypatch.setattr(s2, "_get", fake_get)
    return TestClient(app), calls


def test_every_index_builds_both_scripts():
    for i in s2.INDICES.values():
        png, flt = s2.evalscript(i, "png"), s2.evalscript(i, "tiff")
        assert "ColorRampVisualizer" in png and "FLOAT32" in flt
        for b in i.bands:
            assert f'"{b}"' in flt
        assert "indexOf(s.SCL)" in flt
        assert [v for v, _ in i.ramp] == sorted(v for v, _ in i.ramp)
    ndmi = s2.INDICES["ndmi"]                               # the user's reference ramp
    assert ndmi.ramp[0] == (-0.8, 0x800000) and ndmi.ramp[-1] == (0.8, 0x000080)


def test_grid_is_ten_metres_and_bounded():
    b = s2.bbox_3857(s2.polygons(AOI))
    w, h = s2.grid(b)
    assert 95 <= w <= 110 and 105 <= h <= 115               # 0.01 degree is ~1 km
    big = s2.bbox_3857(s2.polygons({"type": "Polygon", "coordinates": [[[50, 20], [56, 20], [56, 26], [50, 20]]]}))
    with pytest.raises(ValueError):
        s2.grid(big)


def test_status_lists_indices(client):
    c, _ = client
    r = c.get("/api/rs").json()
    assert r["configured"] and {"ndvi", "ndmi", "ndwi", "ndsi", "savi"} <= {i["id"] for i in r["indices"]}


def test_not_configured_is_409(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("SENTINELHUB_INSTANCE_ID")
    r = c.post("/api/rs/render", json={"index": "ndvi", "aoi": AOI, "start": "2026-10-01", "end": "2026-11-20"})
    assert r.status_code == 409


def test_render_png_and_stats_from_the_same_numbers(client):
    c, calls = client
    r = c.post("/api/rs/render", json={"index": "ndmi", "aoi": AOI, "start": "2026-10-01", "end": "2026-11-20"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["image"].startswith("data:image/png;base64,")
    assert j["stats"]["mean"] == pytest.approx(0.5) and 0.7 <= j["stats"]["valid_fraction"] <= 0.8   # a quarter cloudy
    s, w, n, e = j["bounds"]
    assert s == pytest.approx(24.20, abs=1e-6) and e == pytest.approx(55.71, abs=1e-6)
    assert calls[-1]["MAXCC"] == 20 and calls[-1]["TIME"] == "2026-10-01/2026-11-20"


def test_bad_input_refused(client):
    c, _ = client
    assert c.post("/api/rs/render", json={"index": "nope", "aoi": AOI, "date": "2026-11-15"}).status_code == 400
    assert c.post("/api/rs/render", json={"index": "ndvi", "aoi": AOI, "date": "15/11/2026"}).status_code == 400
    assert c.post("/api/rs/render", json={"index": "ndvi", "aoi": {}, "date": "2026-11-15"}).status_code == 400
    r = c.post("/api/rs/attributes", json={"index": "ndvi", "job_id": "..", "model_id": "x", "date": "2026-11-15"})
    assert r.status_code == 404


def test_attributes_saved_with_honest_small_feature_note(client, tmp_path):
    c, _ = client
    from app.core.config import settings

    d = settings.RESULTS_DIR / "job1"
    d.mkdir(parents=True)
    field = {"type": "Feature", "properties": {"class": "field"},
             "geometry": {"type": "Polygon", "coordinates": [[[55.702, 24.205], [55.706, 24.205], [55.706, 24.209], [55.702, 24.209], [55.702, 24.205]]]}}
    tree = {"type": "Feature", "properties": {"class": "tree"},
            "geometry": {"type": "Polygon", "coordinates": [[[55.7081, 24.2051], [55.70813, 24.2051], [55.70813, 24.20513], [55.7081, 24.2051]]]}}
    (d / "m.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [field, tree]}))
    r = c.post("/api/rs/attributes", json={"index": "ndvi", "job_id": "job1", "model_id": "m", "date": "2026-11-15"}).json()
    assert r["saved"] and r["with_values"] == 1 and r["too_small"] == 1
    fc = json.loads((d / "m.geojson").read_text())
    p0, p1 = fc["features"][0]["properties"], fc["features"][1]["properties"]
    assert p0["ndvi_mean"] == pytest.approx(0.5) and p0["ndvi_px"] > 4 and p0["ndvi_date"] == "2026-11-15"
    assert "ndvi_mean" not in p1 and p1["ndvi_note"].startswith("too small")


def test_scenes_and_timeline(client):
    c, _ = client
    r = c.post("/api/rs/scenes", json={"aoi": AOI, "start": "2026-10-01", "end": "2026-11-20"}).json()
    assert [x["date"] for x in r["scenes"]] == ["2026-11-15", "2026-11-10"]
    t = c.post("/api/rs/timeline", json={"index": "ndvi", "aoi": AOI, "start": "2026-10-01", "end": "2026-11-20"}).json()
    assert [p["date"] for p in t["series"]] == ["2026-11-10", "2026-11-15"]
    assert t["series"][0]["mean"] == pytest.approx(0.5)


def test_export_geotiff(client):
    c, _ = client
    r = c.post("/api/rs/export", json={"index": "ndwi", "aoi": AOI, "date": "2026-11-15"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/tiff"
    with rasterio.open(io.BytesIO(r.content)) as ds:
        assert ds.crs.to_epsg() == 3857 and ds.dtypes[0] == "float32"
        a = ds.read(1)
        assert np.isnan(a).any() and np.nanmean(a) == pytest.approx(0.5)


def test_cache_and_rate_limit(tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("SENTINELHUB_INSTANCE_ID", "x")
    n = {"calls": 0}

    class R:
        def __init__(self, code, body=b"ok", headers=None):
            self.status_code, self.content, self.headers = code, body, headers or {}

    class C:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, params=None):
            n["calls"] += 1
            return R(200, b"png-bytes") if params.get("A") == "1" else R(429, b"", {"retry-after": "60"})
    monkeypatch.setattr(s2, "_client", lambda: C())
    assert s2._get("https://x", {"A": "1"}) == b"png-bytes"
    assert s2._get("https://x", {"A": "1"}) == b"png-bytes" and n["calls"] == 1      # second answer from disk
    with pytest.raises(s2.RateLimited):
        s2._get("https://x", {"A": "2"})
