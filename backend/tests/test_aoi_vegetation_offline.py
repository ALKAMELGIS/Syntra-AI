"""AOI-restricted analysis and the NDVI/VARI vegetation analysis."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.engine.vegetation import VegetationError, choose_index_bands, compute_index, read_calibration_gains


# --- pure parts ----------------------------------------------------------------
def test_index_band_choice():
    assert choose_index_bands(8, "SVN3") == ("NDVI", {"red": 3, "nir": 4})
    assert choose_index_bands(8, "WV03") == ("NDVI", {"red": 5, "nir": 7})
    assert choose_index_bands(4, None) == ("NDVI", {"red": 3, "nir": 4})
    assert choose_index_bands(3, None) == ("VARI", {"red": 1, "green": 2, "blue": 3})
    assert choose_index_bands(4, None, override=[1, 4]) == ("NDVI", {"red": 1, "nir": 4})
    with pytest.raises(VegetationError):
        choose_index_bands(1, None)
    with pytest.raises(VegetationError):
        choose_index_bands(4, None, override=[1, 9])


def test_index_maths():
    ndvi = compute_index("NDVI", {"red": np.array([10.0, 50.0, 0.0]), "nir": np.array([30.0, 50.0, 0.0])})
    assert ndvi[0] == pytest.approx(0.5) and ndvi[1] == pytest.approx(0.0) and np.isnan(ndvi[2])
    vari = compute_index("VARI", {"red": np.array([50.0]), "green": np.array([100.0]), "blue": np.array([30.0])})
    assert vari[0] == pytest.approx(50 / 120)


def test_calibration_gains_from_sidecar(tmp_path):
    tif = tmp_path / "scene.tif"
    tif.write_bytes(b"")
    tif.with_suffix(".xml").write_text(
        "<x><AbsCalibrationGain>0.1,0.2,0.3</AbsCalibrationGain><AbsCalibrationBias>0,1,2</AbsCalibrationBias></x>")
    assert read_calibration_gains(tif) == ([0.1, 0.2, 0.3], [0.0, 1.0, 2.0])
    assert read_calibration_gains(tmp_path / "none.tif") is None


# --- with GeoTIFFs ---------------------------------------------------------------
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")
from pyproj import Transformer  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

ORIGIN = (323000.0, 2774000.0)   # UTM 40N, Dubai Marina
PX = 0.5


def _write(path, data, crs="EPSG:32640"):
    c, h, w = data.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=c, dtype=str(data.dtype),
                       crs=crs, transform=from_origin(*ORIGIN, PX, PX)) as d:
        d.write(data)


def _aoi_from_utm(x0, y0, x1, y1):
    tf = Transformer.from_crs("EPSG:32640", "EPSG:4326", always_xy=True)
    ring = [tf.transform(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {}}


@pytest.fixture
def scene(tmp_path):
    """400x400 px (200x200 m) 4-band B,G,R,NIR uint16. Two vegetated squares:
    A 40x40 px (20x20 m = 400 m2) at px (40..80), B 20x20 px (10x10 m = 100 m2)
    at px (300..320), and a 1-pixel speck that must be filtered out."""
    d = np.full((4, 400, 400), 1000, dtype=np.uint16)        # bare ground: NDVI 0
    d[3, :, :] = 1000
    for r0, c0, n in ((40, 40, 40), (300, 300, 20)):
        d[2, r0:r0 + n, c0:c0 + n] = 400                       # red low
        d[3, r0:r0 + n, c0:c0 + n] = 2400                      # NIR high -> NDVI ~0.71
        d[3, r0 + n // 2, c0 + n // 2] = 3000                  # a crown peak
    d[2, 200, 200], d[3, 200, 200] = 400, 2400                 # speck
    path = tmp_path / "farm.tif"
    _write(path, d)
    return path


def test_vegetation_polygons_areas_and_trees(scene, tmp_path):
    from app.engine.vegetation import run_vegetation

    res = run_vegetation(str(scene), tmp_path / "out", "vegetation_index")
    gj = json.loads((tmp_path / "out" / "vegetation_index.geojson").read_text())
    feats = sorted(gj["features"], key=lambda f: -f["properties"]["area_m2"])
    assert len(feats) == 2, [f["properties"] for f in feats]           # speck removed
    assert feats[0]["properties"]["area_m2"] == pytest.approx(400, rel=0.06)
    assert feats[1]["properties"]["area_m2"] == pytest.approx(100, rel=0.12)
    assert feats[0]["properties"]["density"] == "dense"
    assert feats[0]["properties"]["mean_ndvi"] == pytest.approx(0.71, abs=0.03)
    # flat squares with a one-pixel bump are not crowns (see the orchard test for real crowns)
    assert all(f["properties"]["tree_candidates"] <= 1 for f in feats)
    lon, lat = feats[0]["geometry"]["coordinates"][0][0]
    assert 55.2 < lon < 55.3 and 25.0 < lat < 25.1                      # WGS84, not UTM
    s = res["summary"]
    assert s["index"] == "NDVI" and s["vegetation_area_ha"] == pytest.approx(0.05, rel=0.08)
    assert s["analysed_area_ha"] == pytest.approx(4.0, rel=0.01)
    manifest = json.loads((tmp_path / "out" / "vegetation_index.manifest.json").read_text())
    assert manifest["crs"] == "EPSG:4326" and manifest["bands"] == {"red": 3, "nir": 4}


def test_vegetation_respects_aoi(scene, tmp_path):
    from app.engine.aoi import prepare_aoi
    from app.engine.vegetation import run_vegetation

    x0, y0 = ORIGIN
    aoi = prepare_aoi(str(scene), _aoi_from_utm(x0 + 10, y0 - 50, x0 + 50, y0 - 10))   # around square A only
    res = run_vegetation(str(scene), tmp_path / "out", aoi=aoi)
    assert res["final_detection_count"] == 1
    assert res["summary"]["analysed_area_ha"] == pytest.approx(0.16, rel=0.05)


def test_vegetation_on_rgb_uses_vari(tmp_path):
    from app.engine.vegetation import run_vegetation

    d = np.zeros((3, 200, 200), dtype=np.uint8)
    d[:] = np.array([120, 110, 100], dtype=np.uint8)[:, None, None]   # bare: VARI < 0
    d[0, 50:90, 50:90], d[1, 50:90, 50:90], d[2, 50:90, 50:90] = 60, 140, 50
    path = tmp_path / "drone.tif"
    _write(path, d)
    res = run_vegetation(str(path), tmp_path / "out")
    assert res["summary"]["index"] == "VARI" and res["final_detection_count"] == 1


def test_calibration_applied_from_sidecar(tmp_path):
    from app.engine.vegetation import run_vegetation

    d = np.full((8, 200, 200), 1000, dtype=np.uint16)
    d[2] = 1500                                                # sand: red radiance > NIR -> NDVI ~0.06
    d[2, 20:80, 20:80] = 1000                                  # vegetation: red low
    d[3, 20:80, 20:80] = 1800                                  # band 4 = NIR for SVN3
    path = tmp_path / "SVN3-01_TEST_MUX.tif"
    _write(path, d)
    path.with_suffix(".xml").write_text(
        "<p><AbsCalibrationGain>0.0684,0.0558,0.0398,0.0463,0.0932,0.0366,0.0336,0.0293</AbsCalibrationGain></p>")
    run_vegetation(str(path), tmp_path / "out", "veg")
    m = json.loads((tmp_path / "out" / "veg.manifest.json").read_text())
    assert m["calibration_applied"] is True and m["radiometry"] == "toa_reflectance_ratio"
    f = json.loads((tmp_path / "out" / "veg.geojson").read_text())["features"][0]["properties"]
    # reflectance ratio: nir 1800*0.0463/1069 = 0.0780, red 1000*0.0398/1555 = 0.0256 -> NDVI 0.506
    # (raw DN would give 0.286, radiance 0.353)
    assert f["mean_ndvi"] == pytest.approx(0.506, abs=0.01)


# --- AOI -------------------------------------------------------------------------
def test_prepare_aoi_window_and_area(scene):
    from app.engine.aoi import prepare_aoi

    x0, y0 = ORIGIN
    a = prepare_aoi(str(scene), _aoi_from_utm(x0 + 50, y0 - 150, x0 + 100, y0 - 50))
    c, r, w, h = a.window
    assert (c, r) == pytest.approx((100, 100), abs=2) and (w, h) == pytest.approx((100, 200), abs=3)
    assert a.area_m2 == pytest.approx(50 * 100, rel=0.02)


def test_prepare_aoi_errors(scene, tmp_path):
    from app.engine.aoi import AoiError, prepare_aoi

    x0, y0 = ORIGIN
    with pytest.raises(AoiError):
        prepare_aoi(str(scene), _aoi_from_utm(x0 + 5000, y0 + 5000, x0 + 5100, y0 + 5100))   # off the image
    with pytest.raises(AoiError):
        prepare_aoi(str(scene), {"type": "Point", "coordinates": [55.25, 25.07]})
    nocrs = tmp_path / "nocrs.tif"
    with rasterio.open(nocrs, "w", driver="GTiff", height=10, width=10, count=1, dtype="uint8") as d:
        d.write(np.zeros((1, 10, 10), dtype=np.uint8))
    with pytest.raises(AoiError):
        prepare_aoi(str(nocrs), _aoi_from_utm(x0, y0 - 5, x0 + 5, y0))


def test_tiles_limited_to_aoi_window_and_polygon(scene):
    from app.engine.aoi import prepare_aoi
    from app.engine.raster_io import iter_tiles

    x0, y0 = ORIGIN
    # a thin diagonal triangle: its bounding box spans 4 tiles, the triangle touches only 3
    tf = Transformer.from_crs("EPSG:32640", "EPSG:4326", always_xy=True)
    pts = [(x0 + 1, y0 - 1), (x0 + 199, y0 - 1), (x0 + 1, y0 - 199)]
    ring = [tf.transform(*p) for p in pts + [pts[0]]]
    aoi = prepare_aoi(str(scene), {"type": "Polygon", "coordinates": [ring]})
    skipped = []
    tiles = list(iter_tiles(scene, tile_size=200, overlap=0.0, window=aoi.window,
                            tile_filter=aoi.tile_intersects, on_skip=lambda: skipped.append(1)))
    assert len(tiles) + len(skipped) == 4
    assert len(skipped) == 1                                    # bottom-right tile lies outside the triangle
    assert aoi.contains_point(x0 + 20, y0 - 20) and not aoi.contains_point(x0 + 190, y0 - 190)


def test_predict_rejects_aoi_off_the_image(scene, tmp_path, monkeypatch):
    import shutil

    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "data" / "results")
    monkeypatch.delenv("SYNTRA_API_KEY", raising=False)
    settings.ensure_dirs()
    img = settings.UPLOAD_DIR / "farm.tif"
    shutil.copy(scene, img)
    x0, y0 = ORIGIN
    with TestClient(app) as client:
        r = client.post("/api/predict", json={"model_id": "vegetation_index", "image_path": str(img),
                                             "aoi": _aoi_from_utm(x0 + 9000, y0, x0 + 9100, y0 + 100)})
        assert r.status_code == 400 and "overlap" in r.json()["detail"]
        r = client.post("/api/predict", json={"model_id": "vegetation_index", "image_path": str(img),
                                             "aoi": _aoi_from_utm(x0 + 10, y0 - 50, x0 + 50, y0 - 10)})
        assert r.status_code == 200
        job = client.get(f"/api/predict/{r.json()['job_id']}").json()
        assert job["status"] == "completed", job.get("error")
        assert job["result"]["final_detection_count"] == 1 and job["has_aoi"] is True


def test_tree_count_needs_distinct_crowns_not_a_flat_patch(tmp_path):
    """Regression: a uniform vegetated patch used to yield thousands of
    'trees' (one per spacing cell). Only crowns that stand out count."""
    from app.engine.vegetation import run_vegetation

    d = np.full((4, 300, 300), 1000, dtype=np.uint16)
    d[2] = 1500
    d[2, 20:140, 20:140], d[3, 20:140, 20:140] = 500, 2000          # flat 60x60 m patch
    yy, xx = np.mgrid[0:300, 0:300]
    for cy, cx in ((200, 180), (200, 220), (240, 200), (260, 250)):  # 4 crowns, ~5 m wide, 20 m apart
        crown = (yy - cy) ** 2 + (xx - cx) ** 2 <= 5 ** 2
        d[2][crown], d[3][crown] = 500, 2600
    path = tmp_path / "orchard.tif"
    _write(path, d)
    run_vegetation(str(path), tmp_path / "out", "veg", min_area_m2=5)
    feats = json.loads((tmp_path / "out" / "veg.geojson").read_text())["features"]
    flat = max(feats, key=lambda f: f["properties"]["area_m2"])["properties"]
    crowns = [f["properties"] for f in feats if f["properties"]["area_m2"] < 200]
    assert flat["tree_candidates"] <= 2, flat
    assert len(crowns) == 4 and all(c["tree_candidates"] == 1 for c in crowns), crowns


def test_no_tree_count_at_coarse_resolution(tmp_path):
    from app.engine.vegetation import run_vegetation

    d = np.full((4, 100, 100), 1000, dtype=np.uint16)
    d[2] = 1500
    d[2, 20:60, 20:60], d[3, 20:60, 20:60] = 500, 2000
    path = tmp_path / "coarse.tif"
    with rasterio.open(path, "w", driver="GTiff", height=100, width=100, count=4, dtype="uint16",
                       crs="EPSG:32640", transform=from_origin(*ORIGIN, 2.0, 2.0)) as f:
        f.write(d)
    res = run_vegetation(str(path), tmp_path / "out", "veg")
    assert res["summary"]["tree_candidates"] is None and res["final_detection_count"] == 1


def test_upload_keeps_product_xml_for_calibration(scene, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "data" / "results")
    monkeypatch.delenv("SYNTRA_API_KEY", raising=False)
    settings.ensure_dirs()
    xml = b"<p><SatelliteID>SVN3-01</SatelliteID><AbsCalibrationGain>1,1,1,1</AbsCalibrationGain></p>"
    with TestClient(app) as client:
        r = client.post("/api/images", files={"file": ("farm.tif", scene.read_bytes(), "image/tiff"),
                                             "metadata": ("farm.xml", xml, "application/xml")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert any(f["code"] == "calibration" for f in body["health"]["findings"])
        stored = list(settings.UPLOAD_DIR.glob("*.xml"))
        assert len(stored) == 1 and stored[0].read_bytes() == xml
        r = client.post("/api/images", files={"file": ("farm.tif", scene.read_bytes(), "image/tiff"),
                                             "metadata": ("evil.sh", b"rm -rf /", "text/plain")})
        assert r.status_code == 400
        r = client.post("/api/images", files={"file": ("farm.tif", scene.read_bytes(), "image/tiff")})
        assert any(f["code"] == "no_calibration" for f in r.json()["health"]["findings"])
