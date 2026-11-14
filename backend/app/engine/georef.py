"""
Give every uploaded GeoTIFF a usable geotransform.

Many satellite products ship without an affine geotransform:

* **GCP-only** (ground control points, typical of L1 products): if an
  affine fit through the GCPs is good (RMS residual <= 1 pixel) the fit is
  written into the uploaded copy in place, which is instant. Otherwise the
  image is warped through the GCPs into a new, north-up raster.
* **RPC-only** (rational polynomial camera model): the image is
  orthorectified with the RPCs onto the local UTM zone. No terrain model is
  used (height 0), so positions shift where there is relief; the raster
  check says so.

Only the copy in the upload folder is changed; the user's original file is
never touched. Warping runs band to band through GDAL, so it does not load
the whole image into memory.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _gcp_affine_rms_px(gcps, fit) -> float:
    res2 = []
    px = math.sqrt(abs(fit.a * fit.e - fit.b * fit.d)) or 1.0
    for g in gcps:
        x = fit.a * g.col + fit.b * g.row + fit.c
        y = fit.d * g.col + fit.e * g.row + fit.f
        res2.append(((x - g.x) ** 2 + (y - g.y) ** 2) / px ** 2)
    return math.sqrt(sum(res2) / len(res2)) if res2 else float("inf")


def _warp(path: Path, *, dst_crs, gcps=None, gcp_crs=None, rpcs=None) -> None:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    tmp = path.with_suffix(".georef.tif")
    with rasterio.open(path) as src:
        kw = {"gcps": gcps} if gcps else {"rpcs": rpcs}
        tf, w, h = calculate_default_transform(gcp_crs or "EPSG:4326", dst_crs, src.width, src.height, **kw)
        profile = src.profile.copy()
        profile.update(driver="GTiff", crs=dst_crs, transform=tf, width=w, height=h, tiled=True,
                       blockxsize=256, blockysize=256, compress="deflate", BIGTIFF="IF_SAFER",
                       nodata=src.nodata if src.nodata is not None else 0)
        for k in ("photometric",):
            profile.pop(k, None)
        with rasterio.open(tmp, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),
                destination=rasterio.band(dst, list(range(1, src.count + 1))),
                src_crs=gcp_crs, dst_crs=dst_crs, dst_transform=tf,
                resampling=Resampling.bilinear, dst_nodata=profile["nodata"], **kw,
            )
            dst.colorinterp = src.colorinterp
    os.replace(tmp, path)


def ensure_georeferenced(path: str | Path) -> dict | None:
    """Returns what was done (for the raster check), or None if nothing was needed."""
    import rasterio
    from rasterio.transform import Affine, from_gcps

    path = Path(path)
    with rasterio.open(path) as ds:
        if ds.crs is not None and ds.transform != Affine.identity():
            return None
        gcps, gcp_crs = ds.gcps
        rpcs = ds.rpcs

    if gcps and gcp_crs:
        fit = from_gcps(gcps)
        rms = _gcp_affine_rms_px(gcps, fit)
        if rms <= 1.0:
            from app.core.image_library import detach_if_linked
            detach_if_linked(path)   # never write into a user's original through a hard link
            with rasterio.open(path, "r+") as ds:
                ds.crs = gcp_crs
                ds.transform = fit
            return {"method": "gcp_affine", "gcp_count": len(gcps), "rms_px": round(rms, 3)}
        dst = gcp_crs
        if not gcp_crs.is_projected:
            lon = sum(g.x for g in gcps) / len(gcps)
            lat = sum(g.y for g in gcps) / len(gcps)
            dst = f"EPSG:{_utm_epsg(lon, lat)}"
        _warp(path, dst_crs=dst, gcps=gcps, gcp_crs=gcp_crs)
        return {"method": "gcp_warp", "gcp_count": len(gcps), "affine_rms_px": round(rms, 3), "crs": str(dst)}

    if rpcs is not None:
        lon, lat = rpcs.long_off, rpcs.lat_off
        dst = f"EPSG:{_utm_epsg(lon, lat)}"
        _warp(path, dst_crs=dst, rpcs=rpcs)
        return {"method": "rpc_ortho_no_dem", "crs": dst}

    return None


def georef_finding(report: dict | None) -> dict | None:
    """The raster-check finding that explains what georeferencing was applied."""
    if not report:
        return None
    m = report["method"]
    if m == "gcp_affine":
        return {"level": "info", "code": "georef_gcp",
                "title": f"Placed from {report['gcp_count']} ground control points",
                "detail": f"The file had no geotransform; an affine fit through its GCPs (RMS {report['rms_px']} px) was used.",
                "fix": None}
    if m == "gcp_warp":
        return {"level": "info", "code": "georef_gcp_warp",
                "title": f"Warped through {report['gcp_count']} ground control points",
                "detail": f"The GCPs don't fit a flat affine transform (RMS {report['affine_rms_px']} px), so the image was "
                          f"warped onto {report['crs']}.",
                "fix": None}
    return {"level": "warning", "code": "georef_rpc",
            "title": "Orthorectified from RPC, without a terrain model",
            "detail": f"The file had only an RPC camera model. It was projected onto {report['crs']} at height 0, so "
                      "positions shift in hilly terrain; flat coasts are barely affected.",
            "fix": "For survey-grade positions: gdalwarp -rpc -to RPC_DEM=dem.tif input.tif output.tif"}
