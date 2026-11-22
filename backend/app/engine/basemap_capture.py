"""
A basemap area as a GeoTIFF to label and train on.

The tiles of an XYZ (slippy-map) service covering a lon/lat box at one zoom are
fetched, stitched and cropped to the box, and written as an RGB GeoTIFF in Web
Mercator (EPSG:3857), so it enters the image library like an upload.

Only sources whose terms allow keeping their tiles may be used: Google's and
Esri's imagery terms forbid storing tiles or using them to train models, and
Bing's forbid saving its imagery, so their hosts are refused. A custom URL is the
user's own service; it must be http(s) to a public host (never this machine or
its network: the engine would otherwise fetch from places a visitor can't).
"""
from __future__ import annotations

import io
import ipaddress
import math
import socket
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

TILE = 256
MAX_TILES = 400                       # 20 x 20 tiles: 5120 px a side at most
R = 6378137.0                         # Web Mercator sphere

SOURCES = {
    "usgs": {
        "label": "USGS imagery (United States, public domain)",
        "url": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
        "max_zoom": 16,
        "licence": "Public domain (U.S. Geological Survey)",
    },
}

# hosts whose terms forbid keeping their tiles or training on them
BLOCKED = ("google.", "googleapis.com", "gstatic.com", "ggpht.com", "arcgisonline.com", "arcgis.com",
           "esri.com", "virtualearth.net", "bing.com")


class CaptureError(ValueError):
    pass


def check_url(template: str) -> str:
    """A custom tile URL: http(s), {z} {x} {y}, a public host, not one whose terms forbid this."""
    t = (template or "").strip()
    parts = urlsplit(t.replace("{", "").replace("}", ""))
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise CaptureError("Give a tile URL starting with https://")
    if not all(k in t for k in ("{z}", "{x}")) or not ("{y}" in t or "{-y}" in t):
        raise CaptureError("The URL needs {z}, {x} and {y} where the tile numbers go")
    host = parts.hostname.lower()
    if any(b in host for b in BLOCKED):
        raise CaptureError("This provider's terms don't allow saving its tiles or training on them")
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(host, None)}
    except OSError:
        raise CaptureError(f"Can't find the host {host}")
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise CaptureError("The tile host must be on the public internet")
    return t


def _tile_xy(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    lat = max(min(lat, 85.0511), -85.0511)
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def plan(bbox: tuple[float, float, float, float], zoom: int) -> dict:
    """Tiles and size for a box [west, south, east, north] at a zoom."""
    w, s, e, n = bbox
    if not (-180 <= w < e <= 180 and -85 <= s < n <= 85):
        raise CaptureError("Draw the area first (west < east, south < north)")
    if not 1 <= zoom <= 22:
        raise CaptureError("Zoom is 1 to 22")
    x0, y0 = _tile_xy(w, n, zoom)
    x1, y1 = _tile_xy(e, s, zoom)
    tx0, ty0, tx1, ty1 = int(x0), int(y0), int(math.ceil(x1)) - 1, int(math.ceil(y1)) - 1
    tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    gsd = 2 * math.pi * R * math.cos(math.radians((s + n) / 2)) / (TILE * 2 ** zoom)
    return {"tx": (tx0, tx1), "ty": (ty0, ty1), "px": (x0, y0, x1, y1), "tiles": tiles,
            "width": round((x1 - x0) * TILE), "height": round((y1 - y0) * TILE), "gsd_m": gsd}


def _url(template: str, z: int, x: int, y: int) -> str:
    return template.replace("{z}", str(z)).replace("{x}", str(x)).replace("{-y}", str(2 ** z - 1 - y)).replace("{y}", str(y))


def _fetch_http(url: str) -> bytes:
    import httpx

    r = httpx.get(url, timeout=20.0, follow_redirects=False, headers={"User-Agent": "Syntra-AI/1.0 (training data from a licensed basemap)"})
    r.raise_for_status()
    return r.content


def capture(template: str, bbox, zoom: int, out: Path, fetch=_fetch_http) -> dict:
    """Fetch, stitch and crop; write an RGB GeoTIFF in EPSG:3857 at `out`."""
    import rasterio
    from PIL import Image
    from rasterio.transform import from_bounds

    p = plan(tuple(bbox), int(zoom))
    if p["tiles"] > MAX_TILES:
        raise CaptureError(f"That's {p['tiles']} tiles: zoom out or draw a smaller area (up to {MAX_TILES})")
    (tx0, tx1), (ty0, ty1) = p["tx"], p["ty"]
    mosaic = np.zeros(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE, 3), np.uint8)
    missing = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            try:
                img = Image.open(io.BytesIO(fetch(_url(template, zoom, tx, ty)))).convert("RGB")
            except Exception:  # noqa: BLE001 - a missing tile stays black and is counted
                missing += 1
                continue
            if img.size != (TILE, TILE):
                img = img.resize((TILE, TILE))
            mosaic[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = np.asarray(img)
    if missing == p["tiles"]:
        raise CaptureError("The service sent no tiles for this area and zoom")
    x0, y0, x1, y1 = p["px"]
    c0, r0 = round((x0 - tx0) * TILE), round((y0 - ty0) * TILE)
    crop = mosaic[r0:r0 + p["height"], c0:c0 + p["width"]]
    n = 2 ** zoom
    to_m = lambda t: (t / n) * 2 * math.pi * R - math.pi * R  # noqa: E731 - tile coordinate to metres
    left, right = to_m(x0), to_m(x1)
    top, bottom = math.pi * R - (y0 / n) * 2 * math.pi * R, math.pi * R - (y1 / n) * 2 * math.pi * R
    h, w = crop.shape[:2]
    with rasterio.open(out, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs="EPSG:3857",
                       transform=from_bounds(left, bottom, right, top, w, h), compress="deflate",
                       tiled=True, blockxsize=256, blockysize=256, photometric="RGB") as dst:
        dst.write(crop.transpose(2, 0, 1))
    return {"width": w, "height": h, "tiles": p["tiles"], "missing": missing, "gsd_m": round(p["gsd_m"], 3)}
