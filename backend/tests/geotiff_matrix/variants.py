"""GeoTIFF variants that real users bring: every way a GeoTIFF can differ
in data type, no-data, bands, georeferencing and layout. Each builder writes
one file and returns what the pipeline should conclude about it."""
from __future__ import annotations

import numpy as np

ORIGIN_UTM40 = (323000.0, 2774000.0)       # Dubai Marina, EPSG:32640
CENTER_LONLAT = (55.2522, 25.0692)          # roughly the centre of the 0.5 m test scene below


BOATS = "/tmp/obbtest/boats.jpg"   # real aerial marina photo, used when present (ship detection check)


def scene(h=240, w=320, seed=0):
    """RGB + NIR uint8-ish content: sand, water, a vegetated block, bright 'boats'.
    With the real marina photo available, its pixels replace the synthetic RGB."""
    import os
    if os.path.exists(BOATS):
        from PIL import Image
        img = np.asarray(Image.open(BOATS).convert("RGB").crop((300, 0, 1500, 900)), dtype=np.float64)
        h, w = img.shape[:2]
        n = np.full((h, w), 30.0); n[650:900, 0:250] = 200
        rgb = img.transpose(2, 0, 1).copy()
        rgb[1, 650:900, 0:250] = np.maximum(rgb[1, 650:900, 0:250], 90); rgb[0, 650:900, 0:250] = 40
        return np.clip(np.concatenate([rgb, n[None]]), 1, 254)
    rng = np.random.default_rng(seed)
    r = np.full((h, w), 180.0); g = np.full((h, w), 160.0); b = np.full((h, w), 130.0); n = np.full((h, w), 170.0)
    r[:, w // 2:], g[:, w // 2:], b[:, w // 2:], n[:, w // 2:] = 30, 70, 80, 15          # water on the right
    r[40:100, 40:120], g[40:100, 40:120], b[40:100, 40:120], n[40:100, 40:120] = 40, 90, 40, 200   # vegetation
    for y in range(30, 200, 25):
        r[y:y + 8, 200:224] = g[y:y + 8, 200:224] = b[y:y + 8, 200:224] = 240                 # boats
    noise = rng.normal(0, 3, (4, h, w))
    return np.clip(np.stack([r, g, b, n]) + noise, 1, 254)


def build_all(tmp, rasterio):
    from rasterio.control import GroundControlPoint
    from rasterio.crs import CRS
    from rasterio.transform import Affine, from_origin

    tf = from_origin(*ORIGIN_UTM40, 0.5, 0.5)
    s = scene()
    h, w = s.shape[1:]
    out = {}

    def write(name, data, *, crs="EPSG:32640", transform=tf, nodata=None, expect=None, colorinterp=None,
              gcps=None, rpcs=None, **profile):
        path = tmp / name
        kw = dict(driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
                  dtype=str(data.dtype), nodata=nodata, **profile)
        if gcps is None and rpcs is None:
            kw.update(crs=crs, transform=transform)
        with rasterio.open(path, "w", **kw) as d:
            d.write(data)
            if colorinterp:
                d.colorinterp = colorinterp
            if gcps is not None:
                d.gcps = (gcps, CRS.from_string(crs))
            if rpcs is not None:
                d.rpcs = rpcs
        out[name] = dict(path=path, **(expect or {}))
        return path

    from rasterio.enums import ColorInterp as CI
    rgb = s[:3].astype(np.uint8)

    write("rgb_uint8_utm.tif", rgb, expect=dict(placed=True))
    write("rgbn_uint16_nodata0.tif", (np.concatenate([s[[2, 1, 0]], s[3:]]) * 8).astype(np.uint16), nodata=0,
          expect=dict(placed=True))
    d = (s * 8).astype(np.uint16); d[:, :, :40] = 0
    write("bgrn_uint16_empty_strip.tif", np.concatenate([d[[2, 1, 0]], d[3:]]), expect=dict(placed=True))
    f = (s / 255.0).astype(np.float32); f[:, :20, :] = np.nan
    write("float32_reflectance_nan.tif", f, nodata=float("nan"), expect=dict(placed=True))
    i16 = (s * 10).astype(np.int16); i16[:, -30:, :] = -9999
    write("int16_nodata_neg9999.tif", i16, nodata=-9999, expect=dict(placed=True))
    write("pan_uint16_single.tif", (s[:3].mean(0, keepdims=True) * 16).astype(np.uint16), expect=dict(placed=True))
    write("rgba_alpha.tif", np.concatenate([rgb, np.full((1, h, w), 255, np.uint8)]),
          colorinterp=[CI.red, CI.green, CI.blue, CI.alpha], expect=dict(placed=True))
    write("geographic_4326.tif", rgb, crs="EPSG:4326",
          transform=from_origin(55.2470, 25.0720, 0.5 / 100_900, 0.5 / 111_300), expect=dict(placed=True))
    write("webmercator_3857.tif", rgb, crs="EPSG:3857",
          transform=from_origin(6150700.0, 2885000.0, 0.55, 0.55), expect=dict(placed=True))
    write("feet_stateplane_2263.tif", rgb, crs="EPSG:2263",
          transform=from_origin(980000.0, 200000.0, 1.64, 1.64), expect=dict(placed=True, near=None))
    write("south_up.tif", rgb[:, ::-1, :].copy(), transform=Affine(0.5, 0, ORIGIN_UTM40[0], 0, 0.5, ORIGIN_UTM40[1] - h * 0.5),
          expect=dict(placed=True))
    rot = Affine.translation(*ORIGIN_UTM40) * Affine.rotation(15) * Affine.scale(0.5, -0.5)
    write("rotated_15deg.tif", rgb, transform=rot, expect=dict(placed=True))
    write("cog_tiled_deflate.tif", rgb, tiled=True, blockxsize=128, blockysize=128, compress="deflate",
          expect=dict(placed=True))
    write("jpeg_ycbcr.tif", rgb, tiled=True, blockxsize=128, blockysize=128, compress="jpeg", photometric="ycbcr",
          expect=dict(placed=True))
    write("tiny_40px.tif", rgb[:, :40, :40].copy(), expect=dict(placed=True))
    write("huge_tile_overlap_1px_wide.tif", rgb[:, :, :1].copy(), expect=dict(placed=True))
    write("no_crs.tif", rgb, crs=None, transform=Affine.identity(), expect=dict(placed=False))
    # GCP-only (typical L1 product): no affine geotransform at all
    gcps = [GroundControlPoint(row=0, col=0, x=ORIGIN_UTM40[0], y=ORIGIN_UTM40[1]),
            GroundControlPoint(row=0, col=w, x=ORIGIN_UTM40[0] + w * 0.5, y=ORIGIN_UTM40[1]),
            GroundControlPoint(row=h, col=0, x=ORIGIN_UTM40[0], y=ORIGIN_UTM40[1] - h * 0.5),
            GroundControlPoint(row=h, col=w, x=ORIGIN_UTM40[0] + w * 0.5, y=ORIGIN_UTM40[1] - h * 0.5)]
    write("gcps_only.tif", rgb, crs="EPSG:32640", gcps=gcps, expect=dict(placed=True))
    m = write("internal_mask_255fill.tif", rgb.copy(), expect=dict(placed=True))
    with rasterio.open(m, "r+") as dd:
        arr = dd.read(); arr[:, :, : w // 4] = 255; dd.write(arr)
        msk = np.full((h, w), 255, np.uint8); msk[:, : w // 4] = 0
        dd.write_mask(msk)
    # RPC-only: a simple linear camera model centred on the scene
    from rasterio.rpc import RPC
    lat_half, lon_half = h * 0.5 / 111_300 / 2, w * 0.5 / 100_900 / 2
    num_s = [0.0] * 20; num_s[1] = 1.0
    num_l = [0.0] * 20; num_l[2] = -1.0
    den = [0.0] * 20; den[0] = 1.0
    rpc = RPC(height_off=0, height_scale=100, lat_off=25.0692, lat_scale=lat_half, line_den_coeff=den,
              line_num_coeff=num_l, line_off=h / 2, line_scale=h / 2, long_off=55.2522, long_scale=lon_half,
              samp_den_coeff=den, samp_num_coeff=num_s, samp_off=w / 2, samp_scale=w / 2)
    write("rpc_only.tif", rgb, rpcs=rpc, expect=dict(placed=True))
    pal = np.zeros((1, h, w), np.uint8); pal[0, :, w // 2:] = 1; pal[0, 40:100, 40:120] = 2
    p = write("palette_indexed.tif", pal, expect=dict(placed=True))
    with rasterio.open(p, "r+") as dd:
        dd.write_colormap(1, {0: (200, 180, 140, 255), 1: (30, 70, 90, 255), 2: (40, 110, 40, 255)})
    return out
