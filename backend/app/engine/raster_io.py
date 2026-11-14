"""
Raster I/O and tiling.

Large satellite/drone orthomosaics (multi-GB GeoTIFFs) must never be loaded
whole into RAM. This module reads windows on demand via rasterio and yields
configurable, overlapping tiles for batched inference.

`rasterio` is imported lazily (inside functions, not at module load) so this
file stays importable for unit tests in environments where GDAL/rasterio
aren't installed; `iter_tiles_from_array` provides the same tiling contract
for synthetic in-memory arrays (used by the offline smoke test).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from app.core.config import settings
from app.engine.geo import GeoTransform


@dataclass(frozen=True)
class Tile:
    data: np.ndarray        # (H, W, C) uint8/uint16
    col_off: int
    row_off: int
    width: int
    height: int
    transform: GeoTransform  # already offset for this tile's top-left


@dataclass
class RasterInfo:
    width: int
    height: int
    count: int               # band count
    dtype: str
    crs: str | None
    transform: GeoTransform
    gsd_x: float              # ground sample distance, meters/px (x)
    gsd_y: float
    path: str


def compute_wgs84_bounds(path: str | Path) -> dict[str, float] | None:
    """
    Real geographic footprint of a raster, reprojected to WGS84 (EPSG:4326)
    so the frontend can place it on a Leaflet map with <ImageOverlay>.

    Returns None rather than guessing when the raster has no CRS at all —
    an unreferenced image has no honest place to sit on a web map.
    """
    import rasterio  # lazy import
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as ds:
        if ds.crs is None:
            return None
        west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        return {"south": south, "west": west, "north": north, "east": east}


def _preview_vrt(ds):
    """The raster warped to Web Mercator, the projection of the web map.

    A Leaflet ImageOverlay stretches its picture linearly between two corners
    in Web Mercator. Only a picture that is itself in Web Mercator lands right:
    a UTM picture stretched over its lat/lon bounding box was off by ~8 m
    (north-up), 165 m (rotated 15 deg) and 460 m (south-up, drawn upside down).
    """
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    return WarpedVRT(ds, crs="EPSG:3857", resampling=Resampling.bilinear, add_alpha=ds.nodata is None)


def preview_bounds_wgs84(path: str | Path) -> dict[str, float] | None:
    """Exact lat/lon corners of the Web Mercator preview (what ImageOverlay needs)."""
    import rasterio
    from pyproj import Transformer

    with rasterio.open(path) as ds:
        if ds.crs is None:
            return None
        with _preview_vrt(ds) as vrt:
            b = vrt.bounds
    t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, south = t.transform(b.left, b.bottom)
    east, north = t.transform(b.right, b.top)
    return {"south": south, "west": west, "north": north, "east": east}


def read_preview_png(path: str | Path, max_dim: int = 2048) -> bytes:
    """
    A true-colour PNG of the raster for the map, in Web Mercator so it sits on
    its real footprint (see _preview_vrt). Decimated at read time via out_shape,
    so a multi-GB orthomosaic never loads at full resolution. Pixels outside the
    footprint are transparent. Same bands, stretch and masking the detector sees.
    """
    import io

    import rasterio  # lazy import
    from PIL import Image

    from app.engine.raster_prep import plan_rgb, read_rgb

    plan = plan_rgb(path)
    with rasterio.open(path) as ds:
        if ds.crs is None:
            src = ds
            ctx = None
        else:
            ctx = _preview_vrt(ds)
            src = ctx
        try:
            scale = min(1.0, max_dim / max(src.width, src.height))
            out_w = max(1, round(src.width * scale))
            out_h = max(1, round(src.height * scale))
            data, valid = read_rgb(src, plan, out_shape=(out_h, out_w))
        finally:
            if ctx is not None:
                ctx.close()

    buf = io.BytesIO()
    alpha = np.where(valid, 255, 0).astype(np.uint8)          # outside the footprint -> transparent
    Image.fromarray(np.dstack([data, alpha]), mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def read_raster_info(path: str | Path) -> RasterInfo:
    """Read metadata only — does not load pixel data."""
    import rasterio  # lazy import

    with rasterio.open(path) as ds:
        transform = GeoTransform.from_rasterio_transform(ds.transform)
        gsd_x = abs(transform.a)
        gsd_y = abs(transform.e)
        return RasterInfo(
            width=ds.width,
            height=ds.height,
            count=ds.count,
            dtype=str(ds.dtypes[0]),
            crs=str(ds.crs) if ds.crs else None,
            transform=transform,
            gsd_x=gsd_x,
            gsd_y=gsd_y,
            path=str(path),
        )


def _window_grid(width: int, height: int, tile_size: int, overlap_frac: float) -> Iterator[tuple[int, int, int, int]]:
    """Yield (col_off, row_off, w, h) windows covering the raster with the configured overlap."""
    stride = max(1, int(tile_size * (1 - overlap_frac)))
    for row_off in range(0, height, stride):
        h = min(tile_size, height - row_off)
        if h <= 0:
            continue
        for col_off in range(0, width, stride):
            w = min(tile_size, width - col_off)
            if w <= 0:
                continue
            yield col_off, row_off, w, h
            if col_off + w >= width:
                break
        if row_off + h >= height:
            break


def count_tiles(width: int, height: int, tile_size: int, overlap: float) -> int:
    """How many tiles iter_tiles() will yield, for an honest progress total."""
    return sum(1 for _ in _window_grid(width, height, tile_size, overlap))


def raster_crs_and_size(path: str | Path) -> tuple[str | None, int, int]:
    import rasterio  # lazy import

    with rasterio.open(path) as ds:
        return (ds.crs.to_string() if ds.crs else None), ds.width, ds.height


def iter_tiles(
    path: str | Path,
    tile_size: int = settings.DEFAULT_TILE_SIZE,
    overlap: float = settings.DEFAULT_TILE_OVERLAP,
    rgb_plan=None,
    on_skip=None,
    window: tuple[int, int, int, int] | None = None,
    tile_filter=None,
) -> Iterator[Tile]:
    """
    Stream tiles from a GeoTIFF/orthomosaic using windowed reads (rasterio),
    never materializing the full raster in memory.

    With an `rgb_plan` (app.engine.raster_prep.plan_rgb) each tile is read
    from the planned bands only, stretched to 8-bit RGB with the scene-wide
    stretch, and tiles with no valid pixels are skipped (`on_skip()` is
    called for each, so progress can still count them).

    `window` (col_off, row_off, width, height) limits tiling to part of the
    raster (an AOI's bounding box); `tile_filter(transform, col, row, w, h)`
    can skip tiles inside it (counted through `on_skip` too).
    """
    import rasterio  # lazy import
    from rasterio.windows import Window

    from app.engine.raster_prep import read_rgb

    with rasterio.open(path) as ds:
        base_transform = GeoTransform.from_rasterio_transform(ds.transform)
        wc, wr, ww, wh = window or (0, 0, ds.width, ds.height)
        for col_off, row_off, w, h in _window_grid(ww, wh, tile_size, overlap):
            col_off, row_off = col_off + wc, row_off + wr
            if tile_filter is not None and not tile_filter(ds.transform, col_off, row_off, w, h):
                if on_skip:
                    on_skip()
                continue
            window_ = Window(col_off, row_off, w, h)
            if rgb_plan is not None:
                data, valid = read_rgb(ds, rgb_plan, window=window_)
                if float(valid.mean()) < 0.005:
                    if on_skip:
                        on_skip()
                    continue
            else:
                data = ds.read(window=window_)                 # (bands, h, w)
                data = np.moveaxis(data, 0, -1)                # -> (h, w, bands)
                if data.shape[-1] >= 3:
                    data = data[..., :3]
                else:
                    data = np.repeat(data[..., :1], 3, axis=-1)
            yield Tile(
                data=data,
                col_off=col_off,
                row_off=row_off,
                width=w,
                height=h,
                transform=base_transform.offset(col_off, row_off),
            )


def iter_tiles_from_array(
    array: np.ndarray,
    transform: GeoTransform,
    tile_size: int = 512,
    overlap: float = 0.2,
) -> Iterator[Tile]:
    """
    Same tiling contract as `iter_tiles`, operating on an in-memory array.
    Used for unit tests / the offline smoke test, where no real GeoTIFF or
    rasterio installation is available.
    """
    height, width = array.shape[:2]
    for col_off, row_off, w, h in _window_grid(width, height, tile_size, overlap):
        data = array[row_off:row_off + h, col_off:col_off + w]
        yield Tile(
            data=data,
            col_off=col_off,
            row_off=row_off,
            width=w,
            height=h,
            transform=transform.offset(col_off, row_off),
        )
