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
def _plan(path: str, mtime: float, bands: tuple[int, ...] | None = None, stretch: tuple[float, float] | None = None):
    from app.engine.raster_prep import plan_rgb

    from app.engine.raster_prep import DISPLAY_STRETCH

    # the map's look: ArcGIS-like percent clip unless the person chose another
    return plan_rgb(path, override=list(bands) if bands else None, stretch=stretch or DISPLAY_STRETCH)


def parse_display(bands: str | None, stretch: str | None) -> tuple[tuple[int, ...] | None, tuple[float, float] | None]:
    """The map's display choice from query text: bands "3,2,1" (1-based, one
    or three) and stretch "2-98" (low-high percentiles). Raises ValueError."""
    b = None
    if bands:
        parts = [int(v) for v in bands.split(",")]
        if len(parts) not in (1, 3) or not all(1 <= v <= 64 for v in parts):
            raise ValueError("bands must be one or three band numbers from 1")
        b = tuple(parts * 3 if len(parts) == 1 else parts)
    st = None
    if stretch == "none":
        return b, "none"                          # as stored (8-bit), else min-max
    if stretch:
        lo, hi = (float(v) for v in stretch.split("-", 1))
        if not (0 <= lo < hi <= 100):
            raise ValueError("stretch must be low-high percentiles, 0 <= low < high <= 100")
        st = (lo, hi)
    return b, st


_PLAN_LOCK = threading.Lock()


def plan_for(path: str, bands: tuple[int, ...] | None = None, stretch: tuple[float, float] | None = None):
    """The scene-wide stretch, computed once per file version.

    A map view asks for ~30 tiles at once; lru_cache alone lets every one of them
    miss together and compute the same plan (30 x 0.35 s on a 5400 px scene, which
    held the first tiles after an optimise back by 4-6 s). The lock makes the
    first request compute it and the rest wait for that one result.
    """
    mtime = os.path.getmtime(path)
    with _PLAN_LOCK:
        return _plan(path, mtime, bands, stretch)


def render_tile(path: str, z: int, x: int, y: int, bands: tuple[int, ...] | None = None,
                stretch: tuple[float, float] | None = None) -> bytes | None:
    """PNG bytes, or None when the tile does not touch the image."""
    import rasterio
    from PIL import Image
    from rasterio.windows import Window, from_bounds

    from app.engine.raster_io import _preview_vrt
    from app.engine.raster_prep import read_rgb

    plan = plan_for(path, bands, stretch)
    left, bottom, right, top = tile_bounds_3857(z, x, y)
    with rasterio.open(path) as ds:
        if ds.crs is None:
            return None
        if bands and max(bands) > ds.count:
            raise ValueError(f"this image has {ds.count} band(s)")
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


# Rendered tiles, kept in memory so zooming back and forth or panning over
# the same place answers at once instead of reading the GeoTIFF again. Keyed
# by the file's size and change time, so an edited file is never served stale.
_TILE_CACHE: "OrderedDict[tuple, bytes | None]" = None  # type: ignore[assignment]
_TILE_CACHE_BYTES = 0
_TILE_CACHE_MAX = 96 * 1024 * 1024
_TILE_LOCK = threading.Lock()


def cached_tile(path: str, z: int, x: int, y: int, bands: tuple[int, ...] | None = None,
                stretch: tuple[float, float] | None = None) -> bytes | None:
    global _TILE_CACHE, _TILE_CACHE_BYTES
    from collections import OrderedDict

    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns, z, x, y, bands, stretch)
    with _TILE_LOCK:
        if _TILE_CACHE is None:
            _TILE_CACHE = OrderedDict()
        if key in _TILE_CACHE:
            _TILE_CACHE.move_to_end(key)
            return _TILE_CACHE[key]
    png = render_tile(path, z, x, y, bands, stretch)
    with _TILE_LOCK:
        _TILE_CACHE[key] = png
        _TILE_CACHE_BYTES += len(png or b"")
        while _TILE_CACHE_BYTES > _TILE_CACHE_MAX and _TILE_CACHE:
            _, old = _TILE_CACHE.popitem(last=False)
            _TILE_CACHE_BYTES -= len(old or b"")
    return png
