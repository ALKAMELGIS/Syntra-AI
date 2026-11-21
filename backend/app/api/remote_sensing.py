"""
Remote sensing: Sentinel-2 L2A spectral indices over an AOI, and per-feature
index values written into a result layer's attributes.

Every answer is computed from the index values Sentinel Hub returns (FLOAT32,
clouds masked with SCL); the coloured picture is drawn here from the same
numbers, so the map, the legend statistics and the attributes always agree,
and one request serves all three. See app.engine.sentinel.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from app.engine import sentinel as s2

router = APIRouter(prefix="/api/rs", tags=["remote-sensing"])
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Span(BaseModel):
    index: str = "ndvi"
    date: str | None = None                 # one acquisition day, or
    start: str | None = None                # the least cloudy pixel between start and end
    end: str | None = None
    max_cloud: float = Field(20, ge=0, le=100)
    mask_clouds: bool = True


class AreaRequest(Span):
    aoi: dict


class AttributesRequest(Span):
    job_id: str | None = None               # a result layer: values are saved into it
    model_id: str | None = None
    features: dict | None = None            # or any FeatureCollection: values are only returned
    min_px: int = Field(4, ge=1, le=100)


class TimelineRequest(AreaRequest):
    max_points: int = Field(12, ge=2, le=24)


def _index(span: Span) -> s2.Index:
    i = s2.INDICES.get(span.index)
    if i is None:
        raise HTTPException(status_code=400, detail=f"Unknown index '{span.index}'")
    for d in (span.date, span.start, span.end):
        if d is not None and not _DATE.match(d):
            raise HTTPException(status_code=400, detail="Dates are YYYY-MM-DD")
    if not span.date and not (span.start and span.end):
        raise HTTPException(status_code=400, detail="Give a date, or a start and an end")
    return i


def _polys(geojson: dict) -> list:
    polys = s2.polygons(geojson or {})
    if not polys:
        raise HTTPException(status_code=400, detail="Draw an area (a polygon) first")
    return polys


def _call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except s2.NotConfigured as e:
        raise HTTPException(status_code=409, detail=str(e))
    except s2.RateLimited as e:
        raise HTTPException(status_code=429, detail=str(e), headers={"Retry-After": str(e.seconds)})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:  # noqa: BLE001 - network: say what failed
        if type(e).__module__.startswith("httpx"):
            raise HTTPException(status_code=502, detail=f"Sentinel Hub is not reachable from the engine: {e}")
        raise


def _span_kw(span: Span) -> dict:
    return {"date": span.date, "start": span.start, "end": span.end,
            "max_cloud": span.max_cloud, "mask_clouds": span.mask_clouds}


def _summary(v) -> dict:
    import numpy as np

    if not v.size:
        return {"valid": 0}
    q = np.percentile(v, [10, 50, 90])
    return {"valid": int(v.size), "mean": round(float(v.mean()), 4), "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4), "std": round(float(v.std()), 4),
            "p10": round(float(q[0]), 4), "median": round(float(q[1]), 4), "p90": round(float(q[2]), 4)}


def colorize(arr, show, ramp) -> bytes:
    """RGBA PNG of `arr` through the ramp's stops, transparent where not `show`."""
    import numpy as np
    from PIL import Image

    vs = np.array([v for v, _ in ramp], dtype="float32")
    cs = np.array([[(c >> 16) & 255, (c >> 8) & 255, c & 255] for _, c in ramp], dtype="float32")
    x = np.clip(arr, vs[0], vs[-1])
    rgb = np.stack([np.interp(x, vs, cs[:, k]) for k in range(3)], axis=-1)
    rgba = np.dstack([rgb, np.where(show, 255, 0)]).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _aoi_mask(polys, shape, transform):
    from rasterio.features import geometry_mask

    geom = {"type": "MultiPolygon",
            "coordinates": [[[list(s2.to_3857(lon, lat)) for lon, lat, *_ in ring] for ring in p] for p in polys]}
    return geometry_mask([geom], out_shape=shape, transform=transform, invert=True)


@router.get("")
def status():
    cfg = s2.config()
    return {"configured": s2.configured(), "provider": "Sentinel Hub", "collection": "Sentinel-2 L2A",
            "service": cfg["base"], "layer": cfg["layer"], "pixel_m": s2.GROUND_RES_M,
            "indices": [s2.public_index(i) for i in s2.INDICES.values()]}


@router.post("/scenes")
def scenes(req: AreaRequest):
    if not (req.start and req.end and _DATE.match(req.start) and _DATE.match(req.end)):
        raise HTTPException(status_code=400, detail="Give a start and an end, YYYY-MM-DD")
    return {"scenes": _call(s2.scenes, _polys(req.aoi), req.start, req.end, req.max_cloud)}


@router.post("/render")
def render(req: AreaRequest):
    idx = _index(req)
    polys = _polys(req.aoi)
    arr, ok, transform, _ = _call(s2.values, idx, polys, clip=False, **_span_kw(req))
    inside = _aoi_mask(polys, arr.shape, transform)
    show = ok & inside
    png = colorize(arr, show, idx.ramp)
    (w, s), (e, n) = s2.to_4326(transform.c, transform.f + transform.e * arr.shape[0]), \
        s2.to_4326(transform.c + transform.a * arr.shape[1], transform.f)
    total = int(inside.sum())
    stats = _summary(arr[show])
    stats["valid_fraction"] = round(stats["valid"] / total, 3) if total else 0.0
    return {"index": s2.public_index(idx), "bounds": [s, w, n, e], "size": [arr.shape[1], arr.shape[0]],
            "image": "data:image/png;base64," + base64.b64encode(png).decode(), "stats": stats,
            "time": req.date or f"{req.start}/{req.end}"}


@router.post("/attributes")
def attributes(req: AttributesRequest):
    """Index statistics per feature. With a job's result layer they are saved as
    `<index>_mean`, `_min`, `_max`, `_std`, `_px`, `_date`; a feature under
    `min_px` Sentinel-2 pixels gets `<index>_note` instead of numbers."""
    idx = _index(req)
    path = None
    if req.job_id and req.model_id:
        from app.api.results import _result_path

        path = _result_path(req.job_id, req.model_id)
        fc = json.loads(path.read_text())
    elif req.features:
        fc = req.features
    else:
        raise HTTPException(status_code=400, detail="Give a result layer (job_id, model_id) or features")
    feats = fc.get("features", [])
    per = [s2.polygons(f) for f in feats]
    allp = [p for ps in per for p in ps]
    if not allp:
        raise HTTPException(status_code=400, detail="The layer has no polygons")
    arr, ok, transform, _ = _call(s2.values, idx, allp, clip=False, **_span_kw(req))
    stats = s2.zonal_stats(arr, ok, transform, per)
    when = req.date or f"{req.start}/{req.end}"
    k = idx.id
    small = nodata = 0
    for f, st in zip(feats, stats):
        p = f.setdefault("properties", {})
        for suffix in ("mean", "min", "max", "std", "px", "date", "note"):
            p.pop(f"{k}_{suffix}", None)
        p[f"{k}_px"] = st["px"]
        p[f"{k}_date"] = when
        if st["px"] < req.min_px:
            p[f"{k}_note"] = f"too small ({st['px']} px of {s2.GROUND_RES_M:g} m)"
            small += 1
        elif not st["valid"]:
            p[f"{k}_note"] = "no clear pixel (cloud or no data)"
            nodata += 1
        else:
            for suffix in ("mean", "min", "max", "std"):
                p[f"{k}_{suffix}"] = st[suffix]
    if path is not None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(fc))
        os.replace(tmp, path)
    return {"index": k, "features": len(feats), "with_values": len(feats) - small - nodata,
            "too_small": small, "no_data": nodata, "time": when, "saved": path is not None,
            "fields": [f"{k}_{x}" for x in ("mean", "min", "max", "std", "px", "date")],
            "values": None if path is not None else stats}


@router.post("/timeline")
def timeline(req: TimelineRequest):
    """Mean index over the AOI on up to `max_points` clear acquisitions."""
    if not (req.start and req.end):
        raise HTTPException(status_code=400, detail="Give a start and an end")
    idx = _index(req)
    polys = _polys(req.aoi)
    found = _call(s2.scenes, polys, req.start, req.end, req.max_cloud)
    dates = sorted(x["date"] for x in found)
    if len(dates) > req.max_points:                         # spread evenly, keep the newest
        step = (len(dates) - 1) / (req.max_points - 1)
        dates = sorted({dates[round(i * step)] for i in range(req.max_points)})
    cloud = {x["date"]: x["cloud"] for x in found}
    b = s2.bbox_3857(polys)
    side = max(b[2] - b[0], b[3] - b[1]) * __import__("math").cos(__import__("math").radians(s2.to_4326(0, (b[1] + b[3]) / 2)[1]))
    res = max(s2.GROUND_RES_M, side / 200)                   # at most ~200 px a side: cheap requests
    series, stopped = [], None
    for d in dates:
        try:
            arr, ok, transform, _ = s2.values(idx, polys, date=d, max_cloud=100, mask_clouds=req.mask_clouds,
                                              res_m=res, clip=False)
        except s2.RateLimited as e:
            stopped = str(e)
            break
        except (RuntimeError, ValueError) as e:
            series.append({"date": d, "cloud": cloud.get(d), "error": str(e)})
            continue
        inside = _aoi_mask(polys, arr.shape, transform)
        v = arr[ok & inside]
        tot = int(inside.sum())
        series.append({"date": d, "cloud": cloud.get(d), "valid_fraction": round(v.size / tot, 3) if tot else 0.0,
                       **({"mean": round(float(v.mean()), 4), "median": round(float(__import__("numpy").median(v)), 4)} if v.size else {})})
    return {"index": s2.public_index(idx), "series": series, "scenes": len(found), "pixel_m": round(res, 1),
            "stopped": stopped}


@router.post("/export")
def export(req: AreaRequest):
    """The index as a FLOAT32 GeoTIFF (EPSG:3857, NaN outside the AOI and under clouds)."""
    import numpy as np
    import rasterio

    idx = _index(req)
    polys = _polys(req.aoi)
    arr, ok, transform, _ = _call(s2.values, idx, polys, clip=False, **_span_kw(req))
    out = np.where(ok & _aoi_mask(polys, arr.shape, transform), arr, np.nan).astype("float32")
    buf = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=out.shape[1], height=out.shape[0], count=1, dtype="float32",
                      crs="EPSG:3857", transform=transform, nodata=float("nan"), compress="deflate") as ds:
            ds.write(out, 1)
            ds.set_band_description(1, idx.name)
            ds.update_tags(index=idx.name, source="Sentinel-2 L2A via Sentinel Hub",
                           time=req.date or f"{req.start}/{req.end}", clouds_masked=str(req.mask_clouds))
        buf.write(mem.read())
    name = f"{idx.id}_{(req.date or req.end or '').replace('-', '')}.tif"
    return Response(buf.getvalue(), media_type="image/tiff",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
