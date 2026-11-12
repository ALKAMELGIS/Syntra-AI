"""
Pure-NumPy geometry helpers for the dataset builder: tile grids and polygon
clipping/measurement. No GIS/GDAL/rasterio dependency, so this stays
importable and unit-testable anywhere.

Ported from the GeoSyntra AI QGIS plugin's core/geometry.py (same math,
trimmed to what the dataset exporter needs; the detector's IoU/NMS helpers
are a later step).
"""
from __future__ import annotations

import math

import numpy as np


# ------------------------------------------------------------------ grid
def tile_starts(length: int, tile: int, overlap: int) -> np.ndarray:
    """Tile start offsets along one axis. The last tile is shifted back so it
    ends exactly at the raster edge (no padding) when length >= tile."""
    if overlap >= tile:
        raise ValueError("Overlap must be smaller than tile size")
    step = tile - overlap
    if length <= tile:
        return np.array([0], dtype=np.int64)
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return np.array(starts, dtype=np.int64)


def tiles_for_bbox(starts: np.ndarray, tile: int, vmin: float, vmax: float) -> range:
    """Indices of tiles along an axis whose span [s, s+tile) intersects [vmin, vmax]."""
    i0 = int(np.searchsorted(starts, vmin - tile, side="right"))
    i1 = int(np.searchsorted(starts, vmax, side="left"))
    return range(max(i0, 0), min(i1, len(starts)))


# ---------------------------------------------------------------- polygons
def polygon_area(pts) -> float:
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon_rect(pts, xmin: float, ymin: float, xmax: float, ymax: float) -> np.ndarray:
    """Sutherland-Hodgman clipping of a polygon by an axis-aligned rect."""

    def clip(poly, inside, intersect):
        out = []
        n = len(poly)
        for k in range(n):
            cur, prev = poly[k], poly[k - 1]
            cin, pin = inside(cur), inside(prev)
            if cin:
                if not pin:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif pin:
                out.append(intersect(prev, cur))
        return out

    def ix(x0):
        return lambda p, q: (x0, p[1] + (q[1] - p[1]) * (x0 - p[0]) / (q[0] - p[0]))

    def iy(y0):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (y0 - p[1]) / (q[1] - p[1]), y0)

    poly = [tuple(p) for p in np.asarray(pts, dtype=float)]
    for inside, inter in (
        (lambda p: p[0] >= xmin, ix(xmin)),
        (lambda p: p[0] <= xmax, ix(xmax)),
        (lambda p: p[1] >= ymin, iy(ymin)),
        (lambda p: p[1] <= ymax, iy(ymax)),
    ):
        if not poly:
            break
        poly = clip(poly, inside, inter)
    return np.array(poly, dtype=float).reshape(-1, 2)


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def convex_hull(pts) -> np.ndarray:
    pts = sorted(set(map(tuple, np.asarray(pts, dtype=float))))
    if len(pts) <= 2:
        return np.array(pts, dtype=float).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def order_corners(c) -> np.ndarray:
    """Order 4 corners clockwise (image coords) starting top-left-most."""
    c = np.asarray(c, dtype=float)
    ctr = c.mean(0)
    ang = np.arctan2(c[:, 1] - ctr[1], c[:, 0] - ctr[0])
    c = c[np.argsort(ang)]          # increasing angle == clockwise when y down
    start = np.argmin(c[:, 0] + c[:, 1])
    return np.roll(c, -start, axis=0)


def min_area_rect(pts) -> np.ndarray:
    """Minimum-area enclosing rectangle -> (4,2) corners, clockwise in image
    coordinates (y down), starting with the top-most/left-most corner."""
    hull = convex_hull(pts)
    if len(hull) < 3:
        if len(hull) == 0:
            return np.zeros((4, 2))
        mn, mx = hull.min(0), hull.max(0)
        return np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])
    best = None
    edges = np.roll(hull, -1, axis=0) - hull
    for ex, ey in edges:
        ang = math.atan2(ey, ex)
        c, s = math.cos(ang), math.sin(ang)
        rot_m = np.array([[c, s], [-s, c]])
        rot = hull @ rot_m.T
        mn, mx = rot.min(0), rot.max(0)
        area = (mx[0] - mn[0]) * (mx[1] - mn[1])
        if best is None or area < best[0]:
            best = (area, rot_m, mn, mx)
    _, rot_m, mn, mx = best
    corners = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])
    return order_corners(corners @ rot_m)


def simplify_rdp(pts, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of a closed ring (Nx2)."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n <= 4 or eps <= 0:
        return pts
    i0 = 0
    i1 = int(np.argmax(((pts - pts[0]) ** 2).sum(1)))
    if i1 == 0:
        return pts

    def rdp(seg):
        if len(seg) < 3:
            return seg
        a, b = seg[0], seg[-1]
        ab = b - a
        length = np.hypot(*ab)
        if length == 0:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / length
        k = int(np.argmax(d))
        if d[k] > eps:
            left = rdp(seg[:k + 1])
            right = rdp(seg[k:])
            return np.vstack([left[:-1], right])
        return np.vstack([a, b])

    first = rdp(pts[i0:i1 + 1])
    second = rdp(np.vstack([pts[i1:], pts[:1]]))
    out = np.vstack([first[:-1], second[:-1]])
    return out if len(out) >= 3 else pts
