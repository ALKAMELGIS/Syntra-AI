"""The map preview must sit on the raster's real footprint.

A white square at a known pixel is drawn into GeoTIFFs in several layouts; its
centre in the preview PNG, placed with the preview bounds the way Leaflet's
ImageOverlay places it (linear in Web Mercator), must land within half a metre
of its true position. The old preview (raster grid stretched over a lat/lon
bounding box) was 8 m off north-up, 165 m rotated and 460 m south-up.
"""
import io
import math

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")


def _merc(lon, lat):
    return math.radians(lon) * 6378137, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * 6378137


def _make(path, transform, crs="EPSG:32640"):
    a = np.full((3, 600, 800), 20, np.uint8)
    a[:, 400:412, 150:162] = 250                                   # marker centred on pixel (156, 406)
    with rasterio.open(path, "w", driver="GTiff", width=800, height=600, count=3, dtype="uint8",
                       crs=crs, transform=transform) as d:
        d.write(a)
    return (156, 406)


@pytest.mark.parametrize("layout", ["north_up", "rotated", "south_up"])
def test_marker_lands_on_its_true_position(tmp_path, layout):
    from PIL import Image
    from rasterio.transform import Affine, from_origin
    from rasterio.warp import transform as warp

    from app.engine.raster_io import preview_bounds_wgs84, read_preview_png

    gsd = 0.5
    if layout == "north_up":
        tf = from_origin(330000, 2773000, gsd, gsd)
    elif layout == "south_up":
        tf = Affine(gsd, 0, 330000, 0, gsd, 2772700)
    else:
        a = math.radians(15)
        tf = Affine(gsd * math.cos(a), -gsd * math.sin(a), 330000, -gsd * math.sin(a), -gsd * math.cos(a), 2773000)
    p = tmp_path / f"{layout}.tif"
    col, row = _make(p, tf)

    x, y = tf * (col, row)
    lon, lat = warp("EPSG:32640", "EPSG:4326", [x], [y])
    tx, ty = _merc(lon[0], lat[0])

    img = np.asarray(Image.open(io.BytesIO(read_preview_png(p))))
    ys, xs = np.nonzero((img[..., 0] > 150) & (img[..., 3] > 0))
    ph, pw = img.shape[:2]
    b = preview_bounds_wgs84(p)
    x0, y0 = _merc(b["west"], b["north"])
    x1, y1 = _merc(b["east"], b["south"])
    gx = x0 + (xs.mean() + 0.5) / pw * (x1 - x0)
    gy = y0 + (ys.mean() + 0.5) / ph * (y1 - y0)
    err_m = math.hypot(gx - tx, gy - ty) * math.cos(math.radians(lat[0]))
    assert err_m < 0.6, f"{layout}: marker drawn {err_m:.2f} m from its true position"
