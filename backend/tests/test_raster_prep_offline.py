"""Band choice, 16-bit stretch, empty-tile skipping and the raster health
check. Pure-numpy parts always run; GeoTIFF parts need rasterio."""
from __future__ import annotations

import numpy as np
import pytest

from app.engine.raster_prep import choose_rgb_bands, compute_stretch, detect_sensor, tile_has_data, to_uint8


# --- band choice -------------------------------------------------------------
def test_override_wins_and_is_validated():
    assert choose_rgb_bands(8, override=[5, 3, 2]) == ((5, 3, 2), "override")
    assert choose_rgb_bands(1, override=[1]) == ((1, 1, 1), "override")
    with pytest.raises(ValueError):
        choose_rgb_bands(4, override=[5, 3, 2])


def test_colorinterp_then_description_then_sensor_then_default():
    assert choose_rgb_bands(4, colorinterp=["blue", "green", "red", "undefined"]) == ((3, 2, 1), "colorinterp")
    assert choose_rgb_bands(3, descriptions=["Red", "Green", "Blue"]) == ((1, 2, 3), "description")
    assert choose_rgb_bands(8, sensor="SVN3-01") == ((3, 2, 1), "sensor")
    assert choose_rgb_bands(8) == ((5, 3, 2), "default")
    assert choose_rgb_bands(4) == ((3, 2, 1), "default")
    assert choose_rgb_bands(1) == ((1, 1, 1), "default")


def test_detect_sensor_from_name_and_xml(tmp_path):
    assert detect_sensor("SVN3-01_20241127_L2A_MUX.tiff") == "SVN3"
    tif = tmp_path / "scene.tif"
    tif.write_bytes(b"")
    tif.with_suffix(".xml").write_text("<ProductInfo><SatelliteID>WV03</SatelliteID></ProductInfo>")
    assert detect_sensor(tif) == "WV03"
    assert detect_sensor(tmp_path / "plain.tif") is None


# --- stretch -----------------------------------------------------------------
def test_stretch_ignores_nodata_and_maps_to_full_range():
    rng = np.random.default_rng(0)
    data = rng.integers(800, 2000, size=(3, 64, 64)).astype(np.uint16)
    data[:, :40, :] = 0                                 # empty footprint
    lo, hi = compute_stretch(data, nodata=None)
    assert all(799 < v < 900 for v in lo), lo           # zeros did not drag the low end to 0
    assert all(1900 < v < 2001 for v in hi), hi
    rgb = to_uint8(data, lo, hi)
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 64, 3)
    valid = rgb[40:]
    assert valid.min() == 0 and valid.max() == 255


def test_tile_has_data():
    empty = np.zeros((3, 32, 32), dtype=np.uint16)
    assert not tile_has_data(empty, None)
    empty[:, 0, 0] = 5
    assert not tile_has_data(empty, None)               # one pixel is noise, not data
    empty[:, :4, :] = 5
    assert tile_has_data(empty, None)


# --- with real GeoTIFFs ------------------------------------------------------
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


def _write(path, data, crs="EPSG:32639", transform=None, nodata=None, **kw):
    count, h, w = data.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=count, dtype=str(data.dtype),
                       crs=crs, transform=transform or from_origin(518260, 2897944, 0.5, 0.5),
                       nodata=nodata, **kw) as d:
        d.write(data)


@pytest.fixture
def svn3_like(tmp_path):
    """8-band uint16, UTM 39N, 0.5 m, 70% empty, named like a SuperView Neo-3 product."""
    rng = np.random.default_rng(1)
    data = rng.integers(300, 3000, size=(8, 600, 1000)).astype(np.uint16)
    data[:, :, :700] = 0
    path = tmp_path / "SVN3-01_20241127_L2A_MUX.tif"
    _write(path, data)
    return path


def test_plan_and_tiles_for_svn3_like_scene(svn3_like):
    from app.engine.raster_io import count_tiles, iter_tiles
    from app.engine.raster_prep import plan_rgb

    plan = plan_rgb(svn3_like)
    assert plan.bands == (3, 2, 1) and plan.source == "sensor" and plan.dtype == "uint16"
    assert all(lo > 200 for lo in plan.lo)

    skipped = []
    tiles = list(iter_tiles(svn3_like, tile_size=256, overlap=0.0, rgb_plan=plan, on_skip=lambda: skipped.append(1)))
    total = count_tiles(1000, 600, 256, 0.0)
    assert len(tiles) + len(skipped) == total
    assert len(skipped) > 0 and len(tiles) > 0
    t = tiles[0]
    assert t.data.dtype == np.uint8 and t.data.shape[2] == 3
    assert t.col_off >= 512                              # every kept tile touches the data area


def test_health_reports_crs_gsd_bitdepth_bands_and_sparse(svn3_like):
    from app.engine.raster_health import check_raster

    h = check_raster(svn3_like)
    codes = {f.code: f for f in h.findings}
    assert h.epsg == 32639 and h.projected and h.units in ("metre", "meter", "m")
    assert h.gsd_m == 0.5 and "gsd" in codes
    assert "bit_depth" in codes and "band_choice" in codes and "sparse" in codes
    assert h.rgb_bands == [3, 2, 1] and h.sensor == "SVN3"
    assert 0.25 < h.valid_fraction < 0.35
    assert h.status == "ok"


def test_health_geographic_crs_converts_gsd_to_metres(tmp_path):
    from app.engine.raster_health import check_raster

    data = np.full((3, 50, 50), 100, dtype=np.uint8)
    path = tmp_path / "geo.tif"
    # ~0.5 m at 26 N
    _write(path, data, crs="EPSG:4326", transform=from_origin(51.2, 26.2, 4.99e-6, 4.5e-6))
    h = check_raster(path)
    assert h.projected is False and h.units == "degree"
    assert 0.45 < h.gsd_m < 0.55
    assert any(f.code == "geographic_crs" for f in h.findings)


def test_health_missing_crs_is_an_error(tmp_path):
    from app.engine.raster_health import check_raster

    data = np.full((3, 40, 40), 100, dtype=np.uint8)
    path = tmp_path / "nocrs.tif"
    with rasterio.open(path, "w", driver="GTiff", height=40, width=40, count=3, dtype="uint8") as d:
        d.write(data)
    h = check_raster(path)
    assert h.status == "error" and any(f.code == "no_crs" for f in h.findings)


def test_health_flags_stripped_large_file_without_overviews(tmp_path, monkeypatch):
    from app.engine import raster_health

    data = np.full((1, 64, 64), 100, dtype=np.uint16)
    path = tmp_path / "pan.tif"
    _write(path, data)
    h = raster_health.check_raster(path)
    h.file_size_bytes = 3_600_000_000                   # pretend it's the 3.6 GB PAN band
    codes = {f.code for f in raster_health._findings(h)}
    assert {"striped", "no_overviews", "single_band"} <= codes


def test_preview_png_is_rgba_with_transparent_empty_area(svn3_like):
    import io

    from PIL import Image

    from app.engine.raster_io import read_preview_png

    img = Image.open(io.BytesIO(read_preview_png(svn3_like, max_dim=200)))
    assert img.mode == "RGBA"
    a = np.array(img)[..., 3]
    assert a[:, :100].max() == 0 and a[:, -20:].min() == 255
