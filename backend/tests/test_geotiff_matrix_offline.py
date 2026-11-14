"""Every GeoTIFF variant in geotiff_matrix/variants.py goes through upload,
raster check, preview and a vegetation run, end to end through the API.

Checks: nothing crashes; every placeable file gets WGS84 bounds; results
are lon/lat near the scene centre (no projected or pixel coordinates leak);
files that can't be analysed are refused with a clear message instead.
"""
import time

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from geotiff_matrix.variants import CENTER_LONLAT, build_all  # noqa: E402

REFUSED = {
    "pan_uint16_single.tif": "near-infrared",
    "palette_indexed.tif": "palette",
}
ELSEWHERE = ("stateplane", "3857")   # placed correctly, but not at the shared scene centre


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    tmp = tmp_path_factory.mktemp("matrix")
    old = {k: getattr(settings, k) for k in ("DATA_DIR", "UPLOAD_DIR", "RESULTS_DIR")}
    settings.DATA_DIR = tmp / "data"
    settings.UPLOAD_DIR = tmp / "data" / "uploads"
    settings.RESULTS_DIR = tmp / "data" / "results"
    settings.ensure_dirs()
    files = build_all(tmp, rasterio)
    from app.main import app

    with TestClient(app) as c:
        yield c, files
    for k, v in old.items():
        setattr(settings, k, v)


def _wait(c, job_id):
    for _ in range(600):
        j = c.get(f"/api/predict/{job_id}").json()
        if j["status"] in ("completed", "error", "interrupted"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_every_variant(matrix):
    c, files = matrix
    failures = []
    for name, info in files.items():
        try:
            up = c.post("/api/images", files={"file": (name, info["path"].read_bytes(), "image/tiff")})
            assert up.status_code == 200, up.text[:200]
            entry = up.json()
            health = entry.get("health") or {}
            assert c.get(f"/api/images/{entry['image_id']}/preview.png").status_code == 200

            start = c.post("/api/predict", json={"model_id": "vegetation_index", "image_id": entry["image_id"]})
            if not info["placed"]:
                assert health.get("status") == "error"
                assert start.status_code == 409 and "coordinate system" in start.text
                continue
            assert entry.get("bounds_wgs84"), "no WGS84 bounds"
            assert health.get("gsd_m") and 0.4 < health["gsd_m"] < 0.6, health.get("gsd_m")
            assert start.status_code == 200, start.text[:200]
            job = _wait(c, start.json()["job_id"])
            if name in REFUSED:
                assert job["status"] == "error" and REFUSED[name] in str(job.get("error")), job.get("error")
                continue
            assert job["status"] == "completed", job.get("error")
            gj = c.get(f"/api/results/{job['job_id']}/vegetation_index").json()
            pts = [p for f in gj["features"] for ring in [f["geometry"]["coordinates"][0]] for p in ring]
            assert all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in pts)
            if pts and not any(k in name for k in ELSEWHERE):
                lon, lat = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
                assert abs(lon - CENTER_LONLAT[0]) + abs(lat - CENTER_LONLAT[1]) < 0.01, (lon, lat)
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
    assert not failures, "\n".join(failures)


def test_gcp_and_rpc_files_get_georeferenced(matrix):
    c, files = matrix
    for name, code in (("gcps_only.tif", "georef_gcp"), ("rpc_only.tif", "georef_rpc")):
        up = c.post("/api/images", files={"file": (name, files[name]["path"].read_bytes(), "image/tiff")}).json()
        codes = [f["code"] for f in up["health"]["findings"]]
        assert code in codes, codes
        assert up["health"]["crs"], "georeferenced copy has a CRS"
