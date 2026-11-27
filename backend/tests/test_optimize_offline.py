"""Large stripped GeoTIFFs become tiled with overviews, with identical pixels and georeferencing."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")


def _stripped(path, w=5000, h=1200, nodata=0, count=4, dtype="uint16"):
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    a = rng.integers(1, 4000, (count, h, w)).astype(dtype)
    a[:, :, :300] = nodata
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=count, dtype=dtype, crs="EPSG:32640",
                       transform=from_origin(330000, 2773000, 0.5, 0.5), nodata=nodata) as d:
        d.write(a)
        d.set_band_description(1, "blue")
    return a


def test_stripped_file_is_retiled_with_overviews(tmp_path):
    from app.engine.optimize import needs_optimizing, optimize

    p = tmp_path / "big.tif"
    a = _stripped(p)
    with rasterio.open(p) as ds:
        before = (ds.crs, ds.transform, ds.nodata)
    assert needs_optimizing(p) == {"retile": True, "overviews": True}
    seen = []
    res = optimize(p, progress=seen.append)
    assert res["changed"] and res["retiled"] and seen[-1] == 100.0
    with rasterio.open(p) as ds:
        assert ds.profile["tiled"] and ds.block_shapes[0] == (512, 512)
        assert ds.overviews(1)[:3] == [2, 4, 8]
        assert (ds.crs, ds.transform, ds.nodata) == before
        assert ds.descriptions[0] == "blue"
        assert np.array_equal(ds.read(), a)
    assert needs_optimizing(p) is None


def test_small_files_are_left_alone(tmp_path):
    from app.engine.optimize import needs_optimizing

    p = tmp_path / "small.tif"
    _stripped(p, w=800, h=600)
    assert needs_optimizing(p) is None


def test_upload_starts_the_optimise_job(tmp_path, monkeypatch):
    import time

    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "d")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "d" / "uploads")
    settings.ensure_dirs()
    src = tmp_path / "big.tif"
    _stripped(src, count=3, dtype="uint8")
    with TestClient(app) as c:
        up = c.post("/api/images", files={"file": ("big.tif", src.read_bytes(), "image/tiff")}).json()
        jid = up["optimize_job_id"]
        for _ in range(100):
            j = c.get(f"/api/jobs/{jid}").json()
            if j["status"] in ("completed", "error"):
                break
            time.sleep(0.1)
        assert j["status"] == "completed", j
        listed = next(e for e in c.get("/api/images").json() if e["image_id"] == up["image_id"])
        assert listed["health"]["tiled"] is True and listed["health"]["overviews"]
