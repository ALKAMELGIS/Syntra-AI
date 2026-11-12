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


def read_preview_png(path: str | Path, max_dim: int = 1024) -> bytes:
    """
    A small, real true-color (or single-band grayscale) PNG rendering of the
    raster for map display — decimated at read time via rasterio's out_shape
    so this never materializes the full-resolution array (orthomosaics here
    run into the tens of thousands of pixels per side).
    """
    import rasterio  # lazy import
    from rasterio.enums import Resampling
    from PIL import Image

    with rasterio.open(path) as ds:
        scale = min(1.0, max_dim / max(ds.width, ds.height))
        out_w = max(1, round(ds.width * scale))
        out_h = max(1, round(ds.height * scale))
        band_count = min(ds.count, 3)
        data = ds.read(
            indexes=list(range(1, band_count + 1)),
            out_shape=(band_count, out_h, out_w),
            resampling=Resampling.bilinear,
        )

    data = np.moveaxis(data, 0, -1)  # (h, w, bands)
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)

    # Stretch whatever bit depth the source is (8/16-bit) to displayable uint8
    # using real per-band min/max from the decimated sample, not a hardcoded
    # guess — some orthomosaics are 16-bit and would render solid black
    # without this.
    data = data.astype(np.float32)
    lo = np.percentile(data, 1, axis=(0, 1), keepdims=True)
    hi = np.percentile(data, 99, axis=(0, 1), keepdims=True)
    span = np.where(hi > lo, hi - lo, 1)
    data = np.clip((data - lo) / span, 0, 1)
    data = (data * 255).astype(np.uint8)

    import io
    buf = io.BytesIO()
    Image.fromarray(data, mode="RGB").save(buf, format="PNG")
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


def iter_tiles(
    path: str | Path,
    tile_size: int = settings.DEFAULT_TILE_SIZE,
    overlap: float = settings.DEFAULT_TILE_OVERLAP,
) -> Iterator[Tile]:
    """
    Stream tiles from a GeoTIFF/orthomosaic using windowed reads (rasterio),
    never materializing the full raster in memory.
    """
    import rasterio  # lazy import
    from rasterio.windows import Window

    with rasterio.open(path) as ds:
        base_transform = GeoTransform.from_rasterio_transform(ds.transform)
        for col_off, row_off, w, h in _window_grid(ds.width, ds.height, tile_size, overlap):
            window = Window(col_off, row_off, w, h)
            data = ds.read(window=window)                 # (bands, h, w)
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
