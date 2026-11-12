"""
Windowed raster reads + tile image writing for the dataset builder, backed by
rasterio (this project's existing raster dependency — see
app/engine/raster_io.py) instead of the QGIS plugin's direct GDAL/osgeo calls.

`rasterio`, `PIL` and `numpy` are imported lazily inside functions/methods so
this module stays importable (and `DatasetBuilder` stays testable with a
fake reader) in environments where those packages aren't installed.
"""
from __future__ import annotations

import os
from pathlib import Path


class RasterReader:
    """Read-only, windowed access to a georeferenced raster for dataset export.

    Mirrors the interface the GeoSyntra AI QGIS plugin's own RasterReader
    exposed to its (QGIS-free) DatasetBuilder: `.width`, `.height`, `.count`,
    `.dtype`, `.gt` (GDAL-style 6-tuple geotransform), `.wkt`, `.path`,
    `read_window(...)`, `black_ratio(...)`, `tile_geotransform(...)`.
    """

    def __init__(self, path: str | Path, bands: list[int] | None = None):
        import rasterio  # lazy import

        self.path = str(path)
        self._ds = rasterio.open(self.path)
        self.width = self._ds.width
        self.height = self._ds.height
        self.count = self._ds.count
        self.dtype = str(self._ds.dtypes[0])
        self.wkt = self._ds.crs.to_wkt() if self._ds.crs else ""
        t = self._ds.transform
        # GDAL-style geotransform (ox, px_w, rot_x, oy, rot_y, px_h) from an
        # affine.Affine(a, b, c, d, e, f) == (px_w, rot, ox, rot, px_h, oy)
        self.gt = (t.c, t.a, t.b, t.f, t.d, t.e)

        colour = [i for i in range(1, self.count + 1) if i != self._alpha_band()]
        if bands:
            self.bands = [int(b) for b in bands]
        elif len(colour) >= 3:
            self.bands = colour[:3]
        else:
            self.bands = [colour[0]] * 3 if colour else [1, 1, 1]

    def _alpha_band(self) -> int | None:
        try:
            for i, ci in enumerate(self._ds.colorinterp, start=1):
                if ci.name.lower() == "alpha":
                    return i
        except Exception:
            pass
        return None

    def read_window(self, xoff: int, yoff: int, w: int, h: int, out_size: tuple[int, int] | None = None):
        """Read a window (may extend beyond the raster; padding is zero and
        flagged invalid). Returns (rgb uint8 HxWx3, valid bool HxW)."""
        import numpy as np
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window

        ow, oh = out_size if out_size else (w, h)
        rgb = np.zeros((oh, ow, 3), dtype=np.uint8)
        valid = np.zeros((oh, ow), dtype=bool)
        x0, y0 = max(xoff, 0), max(yoff, 0)
        x1, y1 = min(xoff + w, self.width), min(yoff + h, self.height)
        if x1 <= x0 or y1 <= y0:
            return rgb, valid
        sx, sy = ow / float(w), oh / float(h)
        dx0, dy0 = int(round((x0 - xoff) * sx)), int(round((y0 - yoff) * sy))
        dx1, dy1 = int(round((x1 - xoff) * sx)), int(round((y1 - yoff) * sy))
        bw, bh = max(dx1 - dx0, 1), max(dy1 - dy0, 1)

        window = Window(x0, y0, x1 - x0, y1 - y0)
        resampling = Resampling.bilinear if (sx != 1 or sy != 1) else Resampling.nearest
        block = self._ds.read(
            indexes=self.bands, window=window, out_shape=(len(self.bands), bh, bw), resampling=resampling
        )
        block = np.moveaxis(block, 0, -1)  # (h, w, 3)
        if block.dtype != np.uint8:
            # non-8-bit imagery: simple linear stretch from the dataset's own
            # min/max (good enough for tile export; satellite percentile
            # stretching can be added later if needed)
            lo, hi = float(block.min()), float(block.max())
            span = hi - lo if hi > lo else 1.0
            block = np.clip((block.astype(np.float32) - lo) * (255.0 / span), 0, 255).astype(np.uint8)
        v = np.ones((bh, bw), dtype=bool)
        nodata = self._ds.nodata
        if nodata is not None:
            v &= ~np.all(block == nodata, axis=-1)
        else:
            v &= ~np.all(block == 0, axis=-1)
        rgb[dy0:dy0 + bh, dx0:dx0 + bw] = block
        valid[dy0:dy0 + bh, dx0:dx0 + bw] = v
        return rgb, valid

    @staticmethod
    def black_ratio(rgb, valid, black_level: int = 0) -> float:
        """Fraction of pixels that are no-data, outside the raster, or black."""
        black = (rgb <= black_level).all(axis=2) | ~valid
        return float(black.mean())

    def tile_geotransform(self, xoff: int, yoff: int, scale: float = 1.0) -> tuple:
        gt = self.gt
        ox = gt[0] + xoff * gt[1] + yoff * gt[2]
        oy = gt[3] + xoff * gt[4] + yoff * gt[5]
        return (ox, gt[1] * scale, gt[2] * scale, oy, gt[4] * scale, gt[5] * scale)

    def close(self):
        self._ds.close()


_WORLD_EXT = {".jpg": ".jgw", ".jpeg": ".jgw", ".png": ".pgw", ".tif": ".tfw", ".tiff": ".tfw"}


def write_world_file(img_path: str, gt: tuple) -> None:
    base, ext = os.path.splitext(img_path)
    wext = _WORLD_EXT.get(ext.lower(), ".wld")
    # world files reference the centre of the upper-left pixel
    cx = gt[0] + gt[1] * 0.5 + gt[2] * 0.5
    cy = gt[3] + gt[4] * 0.5 + gt[5] * 0.5
    with open(base + wext, "w", encoding="utf-8") as fh:
        fh.write("\n".join("%.12f" % v for v in (gt[1], gt[4], gt[2], gt[5], cx, cy)) + "\n")


def write_image(path: str, rgb, fmt: str = "jpg", quality: int = 95,
                 geotransform: tuple | None = None, wkt: str | None = None) -> None:
    """Write an RGB uint8 array. JPG/PNG use PIL (already a project
    dependency via app/engine/raster_io.py) plus a world file for
    georeferencing; GeoTIFF tiles are written with rasterio and keep a real
    CRS + transform."""
    if fmt in ("jpg", "jpeg", "png"):
        from PIL import Image

        img = Image.fromarray(rgb, mode="RGB")
        if fmt in ("jpg", "jpeg"):
            img.save(path, format="JPEG", quality=quality)
        else:
            img.save(path, format="PNG")
        if geotransform:
            write_world_file(path, geotransform)
    elif fmt == "tif":
        import numpy as np
        import rasterio
        from affine import Affine

        h, w = rgb.shape[:2]
        transform = None
        crs = None
        if geotransform:
            a, b, c, d, e, f = geotransform
            transform = Affine(b, c, a, e, f, d)
            crs = wkt
        with rasterio.open(
            path, "w", driver="GTiff", width=w, height=h, count=3, dtype=np.uint8,
            compress="DEFLATE", transform=transform, crs=crs,
        ) as ds:
            ds.write(np.moveaxis(rgb, -1, 0))
    else:
        raise ValueError("Unsupported tile image format: %r" % fmt)
