"""A scene's cloud mask uploaded instead of the image: named, flagged, and not run."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


def _write(path, data, crs="EPSG:4326", res=1.8e-5):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[-1], height=data.shape[-2],
                       count=data.shape[0], dtype=data.dtype, crs=crs,
                       transform=from_origin(56.0, 26.6, res, res)) as f:
        f.write(data)
    return path


def test_empty_cloud_mask_is_an_error_that_names_the_mask(tmp_path):
    from app.engine.raster_health import check_raster

    p = _write(tmp_path / "SVN3-20241127_L2A0000211147_03_1062500030010003_Cloud.tiff", np.zeros((1, 400, 400), np.uint8))
    h = check_raster(p)
    codes = {f.code: f for f in h.findings}
    assert h.status == "error" and "no_data" in codes
    assert "mask" in codes["no_data"].title and "PAN, MUX or PMS" in codes["no_data"].fix
    assert "sparse" not in codes                    # not also reported as "0% of the frame has data"
    assert "single_band" not in codes               # a mask is not a panchromatic image


def test_cloud_mask_with_clouds_is_a_warning(tmp_path):
    from app.engine.raster_health import check_raster

    d = np.zeros((1, 400, 400), np.uint8)
    d[0, 100:200, 100:200] = 255
    h = check_raster(_write(tmp_path / "scene_CLD.tif", d))
    assert h.status == "warning" and any(f.code == "mask" for f in h.findings)


def test_real_panchromatic_scene_is_not_called_a_mask(tmp_path):
    from app.engine.raster_health import check_raster

    rng = np.random.default_rng(0)
    h = check_raster(_write(tmp_path / "SVN3_PAN.tiff", rng.integers(1, 255, (1, 400, 400), dtype=np.uint8)))
    assert not any(f.code in ("mask", "no_data") for f in h.findings)


def test_prediction_refused_on_empty_mask(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    settings.ensure_dirs()
    from app.main import app

    c = TestClient(app)
    p = _write(tmp_path / "x_Cloud.tiff", np.zeros((1, 300, 300), np.uint8))
    with open(p, "rb") as fh:
        up = c.post("/api/images", files={"file": ("x_Cloud.tiff", fh, "image/tiff")})
    assert up.status_code == 200, up.text
    entry = up.json()
    assert entry["health"]["status"] == "error"
    r = c.post("/api/predict", json={"model_id": "ship_detection", "image_id": entry["image_id"]})
    assert r.status_code == 409 and "mask" in r.text
