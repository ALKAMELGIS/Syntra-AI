"""
Sentinel-2 L2A spectral indices through a Sentinel Hub OGC configuration.

One configuration (instance) id, set as SENTINELHUB_INSTANCE_ID in .env, is
all it needs: Sentinel Hub runs our evalscript server-side (EVALSCRIPT
parameter of WMS GetMap), so any index and colour ramp works without a
layer per index. Three kinds of request:

  render  PNG of the index over an area, coloured by the index's ramp,
          clipped to the area, clouds masked with the L2A scene
          classification (SCL)
  values  FLOAT32 GeoTIFF of the index (band 1) and a validity mask (band 2):
          the numbers behind statistics, the timeline and the export
  scenes  WFS list of acquisitions (date, cloud cover) over an area

Everything is requested in Web Mercator (EPSG:3857), so the PNG lies exactly
on the Leaflet map, at about 10 m on the ground (Sentinel-2's finest bands).

Sentinel-2 pixels are 10-20 m. Statistics for a feature smaller than a few
pixels say little about that feature; `zonal_stats` reports the pixel count
so the caller can flag it instead of presenting a number as if it measured a
single tree.
"""
from __future__ import annotations

import base64
import math
import os
from dataclasses import dataclass

R_EARTH = 6378137.0
GROUND_RES_M = 10.0
MAX_SIDE_PX = 2500
CLOUD_SCL = (3, 8, 9, 10)          # cloud shadow, cloud medium, cloud high, cirrus


@dataclass(frozen=True)
class Index:
    id: str
    name: str
    use: str
    bands: tuple[str, ...]
    expr: str                      # JS expression of the sample `s`
    ramp: tuple[tuple[float, int], ...]
    keep_snow: bool = False


RDYLGN = ((-0.2, 0xa50026), (0.0, 0xd73027), (0.2, 0xfee08b), (0.4, 0xd9ef8b), (0.6, 0x66bd63), (0.8, 0x1a9850), (1.0, 0x006837))

INDICES: dict[str, Index] = {i.id: i for i in (
    Index("ndvi", "NDVI", "Vegetation health", ("B04", "B08"), "(s.B08-s.B04)/(s.B08+s.B04)", RDYLGN),
    Index("ndre", "NDRE", "Early crop stress", ("B05", "B8A"), "(s.B8A-s.B05)/(s.B8A+s.B05)",
          ((-0.2, 0xa50026), (0.0, 0xf46d43), (0.15, 0xfee08b), (0.3, 0xa6d96a), (0.45, 0x1a9850), (0.6, 0x006837))),
    Index("savi", "SAVI", "Sparse vegetation on soil", ("B04", "B08"), "1.5*(s.B08-s.B04)/(s.B08+s.B04+0.5)", RDYLGN),
    Index("msavi", "MSAVI", "Sparse vegetation, no soil factor", ("B04", "B08"),
          "(2*s.B08+1-Math.sqrt((2*s.B08+1)*(2*s.B08+1)-8*(s.B08-s.B04)))/2", RDYLGN),
    Index("evi", "EVI", "Dense vegetation", ("B02", "B04", "B08"),
          "2.5*(s.B08-s.B04)/(s.B08+6*s.B04-7.5*s.B02+1)", RDYLGN),
    # NDMI ramp: Sentinel Hub custom scripts, sentinel-2/ndmi (CC BY-SA 4.0)
    Index("ndmi", "NDMI", "Vegetation moisture", ("B8A", "B11"), "(s.B8A-s.B11)/(s.B8A+s.B11)",
          ((-0.8, 0x800000), (-0.24, 0xff0000), (-0.032, 0xffff00), (0.032, 0x00ffff), (0.24, 0x0000ff), (0.8, 0x000080))),
    Index("ndwi", "NDWI", "Open water", ("B03", "B08"), "(s.B03-s.B08)/(s.B03+s.B08)",
          ((-0.8, 0x8c510a), (-0.3, 0xd8b365), (0.0, 0xf6e8c3), (0.2, 0x9ecae1), (0.5, 0x3182bd), (0.8, 0x08306b))),
    Index("ndsi", "NDSI", "Snow and ice", ("B03", "B11"), "(s.B03-s.B11)/(s.B03+s.B11)",
          ((-0.5, 0x3d3d3d), (0.0, 0x8c8c8c), (0.4, 0x9ecae1), (0.6, 0xdeebf7), (1.0, 0xffffff)), keep_snow=True),
    Index("ndbi", "NDBI", "Built-up areas", ("B08", "B11"), "(s.B11-s.B08)/(s.B11+s.B08)",
          ((-0.6, 0x2166ac), (-0.2, 0x92c5de), (0.0, 0xf7f7f7), (0.1, 0xf4a582), (0.3, 0xd6604d), (0.6, 0x67001f))),
    Index("bsi", "BSI", "Bare soil", ("B02", "B04", "B08", "B11"),
          "((s.B11+s.B04)-(s.B08+s.B02))/((s.B11+s.B04)+(s.B08+s.B02))",
          ((-0.6, 0x1a9850), (-0.2, 0x91cf60), (0.0, 0xf7f7f7), (0.1, 0xdfc27d), (0.3, 0xa6611a), (0.6, 0x543005))),
    Index("nbr", "NBR", "Burnt areas", ("B08", "B12"), "(s.B08-s.B12)/(s.B08+s.B12)",
          ((-0.6, 0x67001f), (-0.2, 0xd6604d), (0.0, 0xfddbc7), (0.2, 0xd9f0d3), (0.5, 0x5aae61), (0.8, 0x00441b))),
)}


class NotConfigured(RuntimeError):
    pass


def config() -> dict:
    return {
        "instance": os.getenv("SENTINELHUB_INSTANCE_ID", "").strip(),
        "base": os.getenv("SENTINELHUB_BASE_URL", "https://services.sentinel-hub.com").rstrip("/"),
        "layer": os.getenv("SENTINELHUB_LAYER", "3_NDVI").strip(),
        "typename": os.getenv("SENTINELHUB_WFS_TYPENAME", "DSS2").strip(),
    }


def configured() -> bool:
    return bool(config()["instance"])


def public_index(i: Index) -> dict:
    return {"id": i.id, "name": i.name, "use": i.use, "bands": list(i.bands),
            "min": i.ramp[0][0], "max": i.ramp[-1][0],
            "ramp": [[v, f"#{c:06x}"] for v, c in i.ramp]}


# --- geometry -------------------------------------------------------------
def to_3857(lon: float, lat: float) -> tuple[float, float]:
    lat = max(-85.0, min(85.0, lat))
    return R_EARTH * math.radians(lon), R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def to_4326(x: float, y: float) -> tuple[float, float]:
    return math.degrees(x / R_EARTH), math.degrees(2 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2)


def polygons(geojson: dict) -> list[list[list[list[float]]]]:
    """Polygon rings (lon/lat) from a Polygon, MultiPolygon, Feature or FeatureCollection."""
    t = geojson.get("type")
    if t == "FeatureCollection":
        return [p for f in geojson.get("features", []) for p in polygons(f)]
    if t == "Feature":
        return polygons(geojson.get("geometry") or {})
    if t == "Polygon":
        return [geojson["coordinates"]]
    if t == "MultiPolygon":
        return list(geojson["coordinates"])
    return []


def bbox_3857(polys) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for poly in polys:
        for lon, lat, *_ in poly[0]:
            x, y = to_3857(lon, lat)
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError("The area has no polygon")
    return min(xs), min(ys), max(xs), max(ys)


def grid(b, res_m: float = GROUND_RES_M, max_side: int = MAX_SIDE_PX) -> tuple[int, int]:
    """Pixel size of a 3857 box at about `res_m` on the ground."""
    lat = to_4326(0, (b[1] + b[3]) / 2)[1]
    scale = math.cos(math.radians(lat))           # 3857 metres per ground metre is 1/cos(lat)
    w = max(1, round((b[2] - b[0]) * scale / res_m))
    h = max(1, round((b[3] - b[1]) * scale / res_m))
    if max(w, h) > max_side:
        raise ValueError(f"Zoom in: the area is about {max(w, h) * res_m / 1000:,.0f} km across, "
                         f"the limit is {max_side * res_m / 1000:.0f} km")
    return max(w, 2), max(h, 2)


def _thin(ring, keep: int = 120):
    if len(ring) <= keep:
        return ring
    step = len(ring) / keep
    out = [ring[int(i * step)] for i in range(keep)]
    return out + [ring[0]]


def wkt_3857(polys) -> str:
    parts = []
    for poly in polys[:20]:
        rings = []
        for ring in poly[:1]:
            ring = _thin(ring)
            rings.append("(" + ",".join("%.2f %.2f" % to_3857(lon, lat) for lon, lat, *_ in ring) + ")")
        parts.append("(" + ",".join(rings) + ")")
    return "MULTIPOLYGON(" + ",".join(parts) + ")"


# --- evalscripts ----------------------------------------------------------
def evalscript(index: Index, kind: str, mask_clouds: bool = True) -> str:
    bands = list(dict.fromkeys([*index.bands, "SCL", "dataMask"]))
    scl = [c for c in CLOUD_SCL] if mask_clouds else []
    ok = "s.dataMask===1" + (f"&&[{','.join(map(str, scl))}].indexOf(s.SCL)<0" if scl else "")
    head = f"//VERSION=3\nfunction ok(s){{return {ok}}}\nfunction val(s){{return {index.expr}}}\n"
    inp = ",".join(f'"{b}"' for b in bands)
    if kind == "png":
        ramp = ",".join(f"[{v},0x{c:06x}]" for v, c in index.ramp)
        return head + (f"const viz=new ColorRampVisualizer([{ramp}]);\n"
                       f'function setup(){{return {{input:[{inp}],output:{{bands:4}}}}}}\n'
                       "function evaluatePixel(s){if(!ok(s))return [0,0,0,0];const v=val(s);"
                       "if(!isFinite(v))return [0,0,0,0];return viz.process(v).concat(1)}")
    return head + (f'function setup(){{return {{input:[{inp}],output:{{bands:2,sampleType:"FLOAT32"}}}}}}\n'
                   "function evaluatePixel(s){if(!ok(s))return [0,0];const v=val(s);"
                   "return isFinite(v)?[v,1]:[0,0]}")


# --- requests -------------------------------------------------------------
def _time(date: str | None, start: str | None, end: str | None) -> str:
    if date:
        return f"{date}/{date}"
    if not (start and end):
        raise ValueError("Give a date, or a start and an end")
    return f"{start}/{end}"


def getmap_params(index: Index, polys, kind: str, *, date=None, start=None, end=None, max_cloud=20,
                  mask_clouds=True, res_m: float = GROUND_RES_M, max_side: int = MAX_SIDE_PX,
                  clip: bool = True) -> tuple[dict, tuple]:
    cfg = config()
    b = bbox_3857(polys)
    w, h = grid(b, res_m, max_side)
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": cfg["layer"],
        "CRS": "EPSG:3857", "BBOX": "%.2f,%.2f,%.2f,%.2f" % b, "WIDTH": w, "HEIGHT": h,
        "FORMAT": "image/png" if kind == "png" else "image/tiff",
        "TIME": _time(date, start, end), "MAXCC": int(max(0, min(100, max_cloud))),
        "PRIORITY": "leastCC", "SHOWLOGO": "false", "TRANSPARENT": "true",
        "EVALSCRIPT": base64.b64encode(evalscript(index, kind, mask_clouds).encode()).decode(),
    }
    if clip:                                     # the URL carries it: kept short (holes and >20 parts dropped)
        params["GEOMETRY"] = wkt_3857(polys)
    return params, b


def _client():
    import httpx

    return httpx.Client(timeout=90.0, follow_redirects=True)


class RateLimited(RuntimeError):
    def __init__(self, seconds: int):
        super().__init__(f"Sentinel Hub rate limit reached: try again in {seconds} s")
        self.seconds = seconds


def _cache_dir():
    from app.core.config import settings

    d = settings.DATA_DIR / "cache" / "sentinel"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, params: dict, cache: bool = True) -> bytes:
    """GET (Sentinel Hub OGC refuses POST). Answers are cached on disk by their
    request: the same scene, area and index never costs a second request."""
    import hashlib
    import time

    if not configured():
        raise NotConfigured("Add SENTINELHUB_INSTANCE_ID to .env, then restart the engine")
    key = hashlib.sha256((url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))).encode()).hexdigest()
    f = _cache_dir() / key
    if cache and f.is_file():
        return f.read_bytes()
    with _client() as c:
        for attempt in range(2):
            r = c.get(url, params=params)
            if r.status_code != 429:
                break
            wait = int(r.headers.get("retry-after") or 30)
            if attempt or wait > 20:
                raise RateLimited(wait)
            time.sleep(wait + 1)
    if r.status_code != 200 or b"ServiceException" in r.content[:600]:
        txt = r.content[:2000].decode("utf-8", "ignore")
        import re

        m = re.search(r"CDATA\[\s*(.*?)\s*\]\]", txt, re.S)
        raise RuntimeError(f"Sentinel Hub: {(m.group(1) if m else txt[:300]).strip()}")
    if cache:
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        os.replace(tmp, f)
    return r.content


def wms_url() -> str:
    cfg = config()
    return f"{cfg['base']}/ogc/wms/{cfg['instance']}"


def render_png(index: Index, polys, **kw) -> tuple[bytes, list[float]]:
    params, b = getmap_params(index, polys, "png", **kw)
    data = _get(wms_url(), params)
    (w, s), (e, n) = to_4326(b[0], b[1]), to_4326(b[2], b[3])
    return data, [s, w, n, e]


def values(index: Index, polys, **kw):
    """(index array float32, valid mask bool, rasterio transform in 3857)."""
    import io

    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    params, b = getmap_params(index, polys, "tiff", **kw)
    data = _get(wms_url(), params)
    with rasterio.open(io.BytesIO(data)) as ds:
        arr = ds.read(1).astype("float32")
        ok = ds.read(2) > 0.5 if ds.count > 1 else np.isfinite(arr)
    return arr, ok & np.isfinite(arr), from_bounds(*b, arr.shape[1], arr.shape[0]), data


def scenes(polys, start: str, end: str, max_cloud: float = 100) -> list[dict]:
    """Acquisition dates over the area, newest first, with scene cloud cover."""
    import json

    cfg = config()
    b = bbox_3857(polys)
    (w, s), (e, n) = to_4326(b[0], b[1]), to_4326(b[2], b[3])
    params = {"SERVICE": "WFS", "REQUEST": "GetFeature", "VERSION": "2.0.0", "TYPENAMES": cfg["typename"],
              "SRSNAME": "EPSG:4326", "BBOX": f"{s},{w},{n},{e}", "TIME": f"{start}/{end}",
              "MAXCC": int(max(0, min(100, max_cloud))), "OUTPUTFORMAT": "application/json", "MAXFEATURES": 500}
    raw = _get(f"{cfg['base']}/ogc/wfs/{cfg['instance']}", params)
    out: dict[str, float] = {}
    for f in json.loads(raw).get("features", []):
        p = f.get("properties") or {}
        d, cc = p.get("date"), p.get("cloudCoverPercentage")
        if d:
            out[d] = min(out.get(d, 101.0), float(cc) if cc is not None else 101.0)
    return [{"date": d, "cloud": None if c > 100 else round(c, 1)} for d, c in sorted(out.items(), reverse=True)]


# --- numbers --------------------------------------------------------------
def zonal_stats(arr, ok, transform, polys_per_feature) -> list[dict]:
    """Per feature: mean, min, max, std and pixel counts of valid index pixels inside it."""
    import numpy as np
    from rasterio.features import geometry_mask

    out = []
    for polys in polys_per_feature:
        if not polys:
            out.append({"px": 0, "valid": 0})
            continue
        geom = {"type": "MultiPolygon",
                "coordinates": [[[list(to_3857(lon, lat)) for lon, lat, *_ in ring] for ring in poly] for poly in polys]}
        inside = geometry_mask([geom], out_shape=arr.shape, transform=transform, invert=True, all_touched=False)
        if not inside.any():                     # smaller than a pixel: the pixel under its centre
            inside = geometry_mask([geom], out_shape=arr.shape, transform=transform, invert=True, all_touched=True)
        v = arr[inside & ok]
        rec = {"px": int(inside.sum()), "valid": int(v.size)}
        if v.size:
            rec.update(mean=round(float(v.mean()), 4), min=round(float(v.min()), 4),
                       max=round(float(v.max()), 4), std=round(float(v.std()), 4))
        out.append(rec)
    return out
