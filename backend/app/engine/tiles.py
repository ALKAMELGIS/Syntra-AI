"""
Full-resolution map tiles straight from the GeoTIFF.

The map preview is one picture of at most 2048 px, so on a 20 000 px
orthomosaic it shows a tenth of the detail. These XYZ tiles (Web Mercator,
256 px) are cut from the raster itself at any zoom, each warped to its exact
place, with the same bands, stretch and masking the models see.

Reads go through a WarpedVRT; rasterio's decimated reads use the file's
overviews when it has them, so build overviews (or use a COG) for fast tiles
on large files: the raster check says so.
"""
from __future__ import annotations

import io
import math
import os
import threading
from functools import lru_cache

import numpy as np

ORIGIN = 20037508.342789244
TILE = 256
_lock = threading.Lock()


def tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    size = 2 * ORIGIN / (2 ** z)
    left = -ORIGIN + x * size
    top = ORIGIN - y * size
    return left, top - size, left + size, top


@lru_cache(maxsize=32)
def _plan(path: str, mtime: float):
    from app.engine.raster_prep import plan_rgb

    return plan_rgb(path)


_PLAN_LOCK = threading.Lock()


def plan_for(path: str):
    """The scene-wide stretch, computed once per file version.

    A map view asks for ~30 tiles at once; lru_cache alone lets every one of them
    miss together and compute the same plan (30 x 0.35 s on a 5400 px scene, which
    held the first tiles after an optimise back by 4-6 s). The lock makes the
    first request compute it and the rest wait for that one result.
    """
    mtime = os.path.getmtime(path)
    with _PLAN_LOCK:
        return _plan(path, mtime)


def render_tile(path: str, z: int, x: int, y: int) -> bytes | None:
    """PNG bytes, or None when the tile does not touch the image."""
    import rasterio
    from PIL import Image
    from rasterio.windows import Window, from_bounds

    from app.engine.raster_io import _preview_vrt
    from app.engine.raster_prep import read_rgb

    plan = plan_for(path)
    left, bottom, right, top = tile_bounds_3857(z, x, y)
    with rasterio.open(path) as ds:
        if ds.crs is None:
            return None
        with _preview_vrt(ds) as vrt:
            vb = vrt.bounds
            il, ib, ir, it = max(left, vb.left), max(bottom, vb.bottom), min(right, vb.right), min(top, vb.top)
            if il >= ir or ib >= it:
                return None
            # where the image part sits inside the 256 px tile
            px = TILE / (right - left)
            c0, c1 = int(round((il - left) * px)), int(round((ir - left) * px))
            r0, r1 = int(round((top - it) * px)), int(round((top - ib) * px))
            if c1 <= c0 or r1 <= r0:
                return None
            win = from_bounds(il, ib, ir, it, vrt.transform)
            win = Window(win.col_off, win.row_off, max(win.width, 1e-6), max(win.height, 1e-6))
            data, valid = read_rgb(vrt, plan, window=win, out_shape=(r1 - r0, c1 - c0))
    out = np.zeros((TILE, TILE, 4), np.uint8)
    out[r0:r1, c0:c1, :3] = data
    out[r0:r1, c0:c1, 3] = np.where(valid, 255, 0)
    buf = io.BytesIO()
    Image.fromarray(out, mode="RGBA").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def native_zoom(gsd_m: float | None, lat: float) -> int | None:
    """Zoom whose tile pixels match the image's ground resolution."""
    if not gsd_m:
        return None
    return int(math.ceil(math.log2(2 * math.pi * 6378137 * math.cos(math.radians(lat)) / 256 / gsd_m)))
