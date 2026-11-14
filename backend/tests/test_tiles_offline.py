"""Full-resolution XYZ tiles land where the raster is, and agree with its pixels."""
import io
import math

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")


def _lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def test_tile_pixel_matches_raster_marker(tmp_path):
    from PIL import Image
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp

    from app.engine.tiles import render_tile

    p = tmp_path / "m.tif"
    a = np.full((3, 600, 800), 20, np.uint8)
    a[:, 300:304, 400:404] = 250                             # a 2 m marker at 0.5 m GSD, centre (402, 302)
    with rasterio.open(p, "w", driver="GTiff", width=800, height=600, count=3, dtype="uint8",
                       crs="EPSG:32640", transform=from_origin(330000, 2773000, 0.5, 0.5)) as d:
        d.write(a)
    x, y = from_origin(330000, 2773000, 0.5, 0.5) * (402, 302)
    lon, lat = warp("EPSG:32640", "EPSG:4326", [x], [y])
    z = 20                                                    # ~0.14 m per tile pixel here
    fx, fy = _lonlat_to_tile(lon[0], lat[0], z)
    png = render_tile(str(p), z, int(fx), int(fy))
    img = np.asarray(Image.open(io.BytesIO(png)))
    ys, xs = np.nonzero(img[..., 0] > 150)
    got = (xs.mean() + 0.5, ys.mean() + 0.5)
    want = ((fx % 1) * 256, (fy % 1) * 256)
    err_px = math.hypot(got[0] - want[0], got[1] - want[1])
    m_per_px = 2 * math.pi * 6378137 * math.cos(math.radians(lat[0])) / 256 / 2 ** z
    assert err_px * m_per_px < 0.5, f"marker {err_px * m_per_px:.2f} m off"
    assert render_tile(str(p), z, int(fx) + 50, int(fy)) is None     # far away: nothing


def test_stretch_plan_computed_once_under_parallel_tile_requests(monkeypatch, tmp_path):
    """A map view requests many tiles at once; the scene stretch must be computed once."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import app.engine.raster_prep as prep
    import app.engine.tiles as tiles

    calls = []
    lock = threading.Lock()

    def slow_plan(path):
        with lock:
            calls.append(path)
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(prep, "plan_rgb", slow_plan)
    tiles._plan.cache_clear()
    f = tmp_path / "x.tif"
    f.write_bytes(b"x")
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda _: tiles.plan_for(str(f)), range(16)))
    assert len(calls) == 1
