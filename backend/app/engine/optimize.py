"""
Make a large uploaded GeoTIFF fast to read: tiled layout + overviews.

A satellite product is often stored in strips (whole image rows) with no
overviews. Every 1024 px tile then reads full rows of a multi-GB file, and
every zoomed-out map view or preview reads the whole image. This is the
same fix as converting to a Cloud-Optimized GeoTIFF:

  1. rewrite stripped files as 512 x 512 tiles (deflate, BigTIFF when needed),
     row band by row band so memory stays bounded;
  2. build overviews (2, 4, 8 ... until under 512 px) with averaging.

Only the copy in the upload folder is changed; pixels, georeferencing,
no-data, band order, colour interpretation and masks are kept exactly.
"""
from __future__ import annotations

import os
from pathlib import Path

ROWS_PER_PASS = 512
MIN_SIDE = 4096          # smaller files read fast enough as they are


def needs_optimizing(path: str | Path) -> dict | None:
    """What would be done, or None when the file is already fast to read."""
    import rasterio

    with rasterio.open(path) as ds:
        big = max(ds.width, ds.height) >= MIN_SIDE
        tiled = bool(ds.profile.get("tiled", False)) and ds.block_shapes[0][0] > 1
        has_ov = bool(ds.overviews(1))
    if not big or (tiled and has_ov):
        return None
    return {"retile": not tiled, "overviews": not has_ov}


def _overview_factors(w: int, h: int) -> list[int]:
    f, out = 2, []
    while max(w, h) / f >= 256:
        out.append(f)
        f *= 2
    return out


def optimize(path: str | Path, progress=lambda pct: None) -> dict:
    import rasterio
    from rasterio.enums import MaskFlags, Resampling
    from rasterio.windows import Window

    path = Path(path)
    plan = needs_optimizing(path)
    if plan is None:
        return {"changed": False}

    if plan["retile"]:
        tmp = path.with_suffix(".tiled.tif")
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            profile.update(driver="GTiff", tiled=True, blockxsize=512, blockysize=512,
                           compress="deflate", predictor=2 if src.dtypes[0] in ("uint8", "uint16", "int16") else 1,
                           BIGTIFF="IF_SAFER", interleave="pixel")
            profile.pop("photometric", None)
            per_dataset_mask = MaskFlags.per_dataset in src.mask_flag_enums[0] and src.nodata is None \
                and MaskFlags.alpha not in src.mask_flag_enums[0]
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.colorinterp = src.colorinterp
                for i in range(1, src.count + 1):
                    if src.descriptions[i - 1]:
                        dst.set_band_description(i, src.descriptions[i - 1])
                dst.update_tags(**src.tags())
                if src.gcps[0]:
                    dst.gcps = src.gcps
                for row in range(0, src.height, ROWS_PER_PASS):
                    h = min(ROWS_PER_PASS, src.height - row)
                    win = Window(0, row, src.width, h)
                    dst.write(src.read(window=win), window=win)
                    if per_dataset_mask:
                        dst.write_mask(src.dataset_mask(window=win), window=win)
                    progress(70.0 * (row + h) / src.height)
        # On Windows a file another reader has open can't be replaced: retry briefly
        import time
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 19:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError("The image was in use by a running analysis; it will be optimised next time.")
                time.sleep(0.5)

    with rasterio.open(path, "r+") as ds:
        factors = _overview_factors(ds.width, ds.height)
        if factors and not ds.overviews(1):
            ds.build_overviews(factors, Resampling.average)
            ds.update_tags(ns="rio_overview", resampling="average")
    progress(100.0)
    return {"changed": True, "retiled": plan["retile"], "overviews": factors}
