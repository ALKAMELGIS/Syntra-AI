"""Run time: the remaining time from the run's own pace, estimates only from measured runs."""
import numpy as np
import pytest


def test_speed_record_and_median(tmp_path):
    from app.engine.speed import record, seconds_per_tile

    assert seconds_per_tile(tmp_path, "ship_detection", "cpu", 1024) is None     # never measured: no number
    for secs in (10, 12, 30, 11, 9, 13):
        record(tmp_path, "ship_detection", "cpu", 1024, model_tiles=10, seconds=secs)
    assert seconds_per_tile(tmp_path, "ship_detection", "cpu", 1024) == pytest.approx(1.2)   # last 5, median
    assert seconds_per_tile(tmp_path, "ship_detection", "cuda:0", 1024) is None   # GPU measured separately
    record(tmp_path, "ship_detection", "cpu", 512, model_tiles=2, seconds=5)      # too few tiles: ignored
    assert seconds_per_tile(tmp_path, "ship_detection", "cpu", 512) is None


def test_eta_follows_the_runs_own_pace():
    from app.engine.speed import Eta

    t = {"now": 0.0}
    eta = Eta(clock=lambda: t["now"])
    assert eta.update(0, 100) is None            # starts the clock
    t["now"] = 1.0
    assert eta.update(5, 100) is None            # under 2 s: pace not known yet
    t["now"] = 10.0
    assert eta.update(25, 100) == 30             # 10 s for 25 tiles -> 30 s for the other 75
    assert eta.update(100, 100) == 0


def test_estimate_endpoint_area_then_time_after_a_run(tmp_path, monkeypatch):
    rasterio = pytest.importorskip("rasterio")
    from fastapi.testclient import TestClient
    from rasterio.transform import from_origin

    from app.core.config import settings
    from app.engine.speed import record

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    settings.ensure_dirs()
    from app.main import app

    c = TestClient(app)
    p = tmp_path / "scene.tif"
    with rasterio.open(p, "w", driver="GTiff", width=2000, height=2000, count=3, dtype="uint8",
                       crs="EPSG:32639", transform=from_origin(518260, 2897944, 0.5, 0.5)) as f:
        f.write(np.random.default_rng(0).integers(1, 255, (3, 2000, 2000), dtype=np.uint8))
    with open(p, "rb") as fh:
        img = c.post("/api/images", files={"file": ("scene.tif", fh, "image/tiff")}).json()
    first = c.post("/api/predict/estimate", json={"image_id": img["image_id"], "model_id": "ship_detection"}).json()
    assert first["area_km2"] == pytest.approx(1.0, rel=0.01) and first["seconds"] is None and first["tiles"] > 0
    record(tmp_path, "ship_detection", first["device"], 1024, model_tiles=20, seconds=40)
    again = c.post("/api/predict/estimate", json={"image_id": img["image_id"], "model_id": "ship_detection"}).json()
    assert again["seconds"] == again["tiles"] * 2
