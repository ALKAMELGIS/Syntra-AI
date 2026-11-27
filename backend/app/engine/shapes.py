"""
Outline geometry: a mask or a rough ring -> the shape the object really has.

Everything works in pixel space (the raster grid), so it is exact for any
CRS; sizes given in metres are turned into pixels with the image's GSD.

    mask_to_polygon   mask from click segmentation -> traced ring (largest
                      part under the click, small holes filled)
    refine            traced ring -> polygon | right angles | rectangle |
                      circle | smooth, then grow / shrink, then a quality
                      gate: if the fitted shape strays too far from what
                      was traced, the simplified trace is kept and a
                      warning says why

The right-angle fitter: orientation from a length-weighted histogram of edge
angles (on a lightly simplified ring, because pixel stairs only have 0/90
degree steps), Douglas-Peucker, every edge snapped to one of the two axes,
runs on the same axis merged, short jogs dropped, then corners rebuilt as the
intersections of the snapped lines. Tested on 30 cm roofs: 479 -> 6 vertices
at IoU 0.97 with the trace. Written for Syntra (no GPL code); the method is
the one `buildingregulariser` (MIT) and iD's orthogonalize (ISC) describe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SHAPES = ("polygon", "right_angles", "rectangle", "box", "square", "circle", "smooth")

# minimum IoU between the fitted shape and the trace before the fit is kept
IOU_FLOOR = {"box": 0.0, "square": 0.0, "right_angles": 0.88, "rectangle": 0.85, "circle": 0.90, "smooth": 0.90, "polygon": 0.0}
AREA_RATIO = (0.7, 1 / 0.7)

_CLASS_SHAPES = (
    (("building", "roof", "house", "warehouse", "hangar", "shed"), "right_angles"),
    (("solar", "panel", "car", "vehicle", "truck", "bus", "container", "court", "pool"), "rectangle"),
    (("tank", "silo", "roundabout", "fountain"), "circle"),
    (("tree", "palm", "vegetation", "bush", "water", "lake", "pond", "field", "grass"), "smooth"),
)


def shape_for_class(class_name: str) -> str:
    """Default outline style for a class name (the user can change it)."""
    n = (class_name or "").lower()
    for words, shape in _CLASS_SHAPES:
        if any(w in n for w in words):
            return shape
    return "polygon"


@dataclass
class RefineOptions:
    shape: str = "polygon"
    px_m: float | None = None          # metres per pixel (GSD); None = sizes are in pixels
    simplify_m: float = 0.6
    grow_m: float = 0.0
    fill_holes_m2: float = 6.0
    min_area_m2: float = 0.0

    def px(self, metres: float) -> float:
        return metres / self.px_m if self.px_m else metres

    def px2(self, square_metres: float) -> float:
        return square_metres / (self.px_m ** 2) if self.px_m else square_metres


def _poly(geom):
    """Largest Polygon of whatever shapely returned."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return max(polys, key=lambda g: g.area) if polys else None


def mask_to_polygon(mask: np.ndarray, seed_xy: tuple[float, float] | None = None,
                    fill_holes_px2: float = 0.0, offset: tuple[float, float] = (0.0, 0.0), scale: float = 1.0):
    """Mask (H, W) -> shapely Polygon in pixel space, or None.

    Keeps the connected part under `seed_xy` (else the largest), closes 1 px
    gaps, fills holes smaller than `fill_holes_px2`. `scale` and `offset` map
    mask pixels back to raster pixels (the mask may be a resampled window).
    """
    import cv2
    from shapely import affinity
    from shapely.geometry import Polygon

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if not m.any():
        return None
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    pick = 0
    if seed_xy is not None:
        sx, sy = int(round(seed_xy[0])), int(round(seed_xy[1]))
        if 0 <= sy < lab.shape[0] and 0 <= sx < lab.shape[1]:
            pick = int(lab[sy, sx])
    if pick == 0:
        pick = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    part = (lab == pick).astype(np.uint8)
    contours, hier = cv2.findContours(part, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outer_i = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]) if hier[0][i][3] < 0 else -1)
    shell = contours[outer_i][:, 0, :].astype(float)
    if len(shell) < 3:
        return None
    holes = []
    for i, c in enumerate(contours):
        if hier[0][i][3] == outer_i and len(c) >= 3 and cv2.contourArea(c) * scale * scale >= fill_holes_px2:
            holes.append(c[:, 0, :].astype(float))
    # pixel centres -> pixel corners: the contour runs through pixel centres
    shell += 0.5
    holes = [h + 0.5 for h in holes]
    poly = Polygon(shell, holes).buffer(0)
    poly = _poly(poly)
    if poly is None:
        return None
    if scale != 1.0 or offset != (0.0, 0.0):
        poly = affinity.affine_transform(poly, [scale, 0, 0, scale, offset[0], offset[1]])
    return poly


def dominant_angle(poly) -> float:
    """Main direction of the outline in degrees, folded into [0, 90).

    Edges of a simplified ring vote with weight length**2, on angles taken x4
    so 0 and 90 degrees count as the same direction. The simplify step has
    to be a few pixels: pixel stairs only have 0/90 degree steps and pull the
    estimate towards the image axes (1.5 px gave 20.7 deg for a 23 deg roof,
    4 px gives 22.8).
    """
    tol = float(np.clip(0.03 * math.sqrt(max(poly.area, 1.0)), 2.0, 6.0))
    c = np.asarray(poly.simplify(tol).exterior.coords)
    d = np.diff(c, axis=0)
    w = np.hypot(d[:, 0], d[:, 1])
    if not len(w) or w.sum() == 0:
        return 0.0
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    z = np.sum(w ** 2 * np.exp(1j * np.radians(ang * 4)))
    return float((np.degrees(np.angle(z)) / 4) % 90)


def _merge_runs(edges):
    out = []
    for e in edges:
        if out and out[-1][0] == e[0]:
            out[-1][1] += e[1]
            out[-1][2] += e[2]
        else:
            out.append(list(e))
    if len(out) > 1 and out[0][0] == out[-1][0]:
        out[0][1] += out[-1][1]
        out[0][2] += out[-1][2]
        out.pop()
    return out


def right_angles(poly, simplify_px: float = 2.0):
    """Every corner 90 degrees, along the outline's own direction."""
    from shapely import affinity
    from shapely.geometry import Polygon

    theta = dominant_angle(poly)
    ctr = poly.centroid
    p = affinity.rotate(poly, -theta, origin=ctr).simplify(simplify_px, preserve_topology=True)
    pts = np.asarray(p.exterior.coords)[:-1]
    edges = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        dx, dy = b - a
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        axis = 0 if abs(dx) >= abs(dy) else 1          # 0 horizontal (fixed y), 1 vertical (fixed x)
        pos = (a[1] + b[1]) / 2 if axis == 0 else (a[0] + b[0]) / 2
        edges.append([axis, pos * length, length])
    runs = _merge_runs(edges)
    for _ in range(3):                                   # drop jogs shorter than the detail size
        kept = [e for e in runs if e[2] >= 3 * simplify_px]
        runs = _merge_runs(kept if len(kept) >= 4 else runs)
    if len(runs) < 4 or len(runs) % 2:
        return poly.minimum_rotated_rectangle
    lines = [(axis, s / length) for axis, s, length in runs]
    verts = []
    for i, (a1, v1) in enumerate(lines):
        a2, v2 = lines[(i + 1) % len(lines)]
        verts.append((v2, v1) if a1 == 0 else (v1, v2))
    q = Polygon(verts)
    if not q.is_valid or q.area <= 0:
        return poly.minimum_rotated_rectangle
    return affinity.rotate(q, theta, origin=ctr)


def fit_circle(poly, n: int = 64):
    """Least-squares circle through the outline (not the enclosing circle)."""
    from shapely.geometry import Polygon

    pts = np.asarray(poly.exterior.coords)[:-1]
    x, y = pts[:, 0], pts[:, 1]
    a = np.c_[2 * x, 2 * y, np.ones(len(x))]
    cx, cy, c = np.linalg.lstsq(a, x * x + y * y, rcond=None)[0]
    r = math.sqrt(max(c + cx * cx + cy * cy, 1e-9))
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return Polygon(np.c_[cx + r * np.cos(t), cy + r * np.sin(t)])


def chaikin(poly, iterations: int = 3):
    """Corner cutting for organic outlines (trees, water)."""
    from shapely.geometry import Polygon

    pts = np.asarray(poly.exterior.coords)[:-1]
    for _ in range(iterations):
        nxt = np.roll(pts, -1, axis=0)
        out = np.empty((2 * len(pts), 2))
        out[0::2] = 0.75 * pts + 0.25 * nxt
        out[1::2] = 0.25 * pts + 0.75 * nxt
        pts = out
    return Polygon(pts)


def iou(a, b) -> float:
    u = a.union(b).area
    return float(a.intersection(b).area / u) if u else 0.0


def refine(poly, opts: RefineOptions) -> dict:
    """Fit the chosen shape and report how it compares with the trace.

    -> {polygon, shape, iou, vertices, area_m2|area_px, warning}
    """
    from shapely.geometry import JOIN_STYLE, Polygon

    if poly is None or poly.is_empty or poly.area <= 0:
        raise ValueError("Nothing to outline here")
    shape = opts.shape if opts.shape in SHAPES else "polygon"
    trace = Polygon(poly.exterior)                         # holes are not stored with labels yet
    if opts.fill_holes_m2 <= 0 and poly.interiors:
        trace = poly
    simplify_px = max(opts.px(opts.simplify_m), 0.5)
    simplified = trace.simplify(simplify_px, preserve_topology=True)

    if shape == "right_angles":
        fitted = right_angles(trace, simplify_px)
    elif shape == "rectangle":
        fitted = trace.minimum_rotated_rectangle
    elif shape == "box":                                   # axis-aligned: what a detection box is
        fitted = trace.envelope
    elif shape == "square":                                # square box (SBB): trees, poles, the YOLO square anchor
        x0, y0, x1, y1 = trace.bounds
        half = max(x1 - x0, y1 - y0) / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        from shapely.geometry import box as _box
        fitted = _box(cx - half, cy - half, cx + half, cy + half)
    elif shape == "circle":
        fitted = fit_circle(trace)
    elif shape == "smooth":
        fitted = chaikin(simplified).simplify(simplify_px / 4, preserve_topology=True)
    else:
        fitted = simplified
    fitted = _poly(fitted.buffer(0)) or simplified

    warning = None
    score = iou(fitted, trace)
    ratio = fitted.area / trace.area if trace.area else 0
    if shape not in ("polygon", "box", "square") and (score < IOU_FLOOR[shape] or not (AREA_RATIO[0] <= ratio <= AREA_RATIO[1])):
        warning = (f"Kept the traced outline: {shape.replace('_', ' ')} changed it too much "
                   f"(overlap {score:.2f}, needs {IOU_FLOOR[shape]:.2f})")
        fitted, shape, score = simplified, "polygon", iou(simplified, trace)

    if opts.grow_m:
        join = JOIN_STYLE.mitre if shape in ("right_angles", "rectangle", "box", "square", "polygon") else JOIN_STYLE.round
        grown = _poly(fitted.buffer(opts.px(opts.grow_m), join_style=join, mitre_limit=2.0))
        if grown is None:
            raise ValueError("Shrinking by that much leaves nothing")
        fitted = grown

    area_px = float(fitted.area)
    if opts.min_area_m2 and area_px < opts.px2(opts.min_area_m2):
        raise ValueError(f"Smaller than the minimum size ({opts.min_area_m2:g} m²)")
    out = {
        "polygon": fitted,
        "shape": shape,
        "iou": round(score, 3),
        "vertices": len(fitted.exterior.coords) - 1,
        "warning": warning,
    }
    if opts.px_m:
        out["area_m2"] = round(area_px * opts.px_m ** 2, 1)
    else:
        out["area_px"] = round(area_px, 1)
    return out
