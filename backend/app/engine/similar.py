"""Find more objects like the ones already labelled, in the visible view.

Each example label is cut out of the view as a small picture and matched over the
view at four turns (0, 90, 180, 270 degrees) and three sizes (0.8, 1, 1.25) with
normalised cross-correlation on the colour image. The best places above the
threshold, not overlapping each other, an example or another label, are the
candidates. Each candidate is then outlined:

- with SAM 2.1 when the click tool is installed (a box prompt round the match),
- otherwise with the example's own outline moved (and turned) onto the match.

Both are fitted to the class's shape like any other outline. They come back as
suggestions to keep or drop; nothing is saved here.
"""
from __future__ import annotations

import time

import numpy as np

MAX_SIDE = 1400           # the view is matched at most this many pixels across
TURNS = (0, 1, 2, 3)      # quarter turns
SIZES = (0.8, 1.0, 1.25)
MAX_FOUND = 150
RELATIVE = 0.75          # keep matches scoring at least this share of the best one
PAD = 0.25                # margin round each example, as a share of its size


def _ring_box(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _centre_in(a, b) -> bool:
    """The middle of either box lies inside the other."""
    ca, cb = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return (b[0] <= ca[0] <= b[2] and b[1] <= ca[1] <= b[3]) or (a[0] <= cb[0] <= a[2] and a[1] <= cb[1] <= a[3])


def _covers(a, b, frac=0.5) -> bool:
    """Box b lies more than `frac` inside box a, or a inside b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return small > 0 and inter / small > frac


def match_boxes(rgb: np.ndarray, examples: list[tuple[float, float, float, float]],
                avoid: list[tuple[float, float, float, float]], threshold: float = 0.6,
                limit: int = MAX_FOUND) -> list[dict]:
    """Template matching on an (H, W, 3) uint8 view. examples and avoid are boxes in
    the view's pixels. -> [{box, score, turn, example}] best first."""
    import cv2

    img = rgb.astype(np.float32)
    H, W = img.shape[:2]
    cands = []
    for ei, (bx0, by0, bx1, by1) in enumerate(examples):
        # the example with a margin round it: its edge against the ground is what
        # tells it apart (a plain roof alone matches any plain patch)
        mx, my = PAD * (bx1 - bx0), PAD * (by1 - by0)
        x0, y0 = max(0, int(round(bx0 - mx))), max(0, int(round(by0 - my)))
        x1, y1 = min(W, int(round(bx1 + mx))), min(H, int(round(by1 + my)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        inner = ((bx0 - x0) / (x1 - x0), (by0 - y0) / (y1 - y0), (bx1 - x0) / (x1 - x0), (by1 - y0) / (y1 - y0))
        tpl0 = img[y0:y1, x0:x1]
        for k in TURNS:
            tpl1 = np.ascontiguousarray(np.rot90(tpl0, k))
            for s in SIZES:
                th, tw = max(4, round(tpl1.shape[0] * s)), max(4, round(tpl1.shape[1] * s))
                if th >= H or tw >= W:
                    continue
                tpl = cv2.resize(tpl1, (tw, th), interpolation=cv2.INTER_AREA) if s != 1.0 else tpl1
                if float(tpl.std()) < 1.0:               # a flat patch matches anything
                    continue
                res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
                res = np.nan_to_num(res, nan=-1.0)
                # local maxima above the threshold
                k_sz = max(3, (min(th, tw) // 2) | 1)
                peaks = (res >= threshold) & (res == cv2.dilate(res, np.ones((k_sz, k_sz), np.uint8)))
                ys, xs = np.nonzero(peaks)
                # the object's part of the padded template, turned like it
                fx0, fy0, fx1, fy1 = inner
                for _ in range(k):                        # np.rot90 (counter-clockwise): (u, v) -> (v, 1 - u)
                    fx0, fy0, fx1, fy1 = fy0, 1 - fx1, fy1, 1 - fx0
                for y, x in zip(ys, xs):
                    cands.append({"box": (float(x + fx0 * tw), float(y + fy0 * th), float(x + fx1 * tw), float(y + fy1 * th)),
                                  "score": float(res[y, x]), "turn": k, "example": ei})
    cands.sort(key=lambda c: -c["score"])
    # a match well below the best one is a part of an object or a look-alike patch
    if cands:
        floor = RELATIVE * cands[0]["score"]
        cands = [c for c in cands if c["score"] >= floor]
    kept: list[dict] = []
    blocked = [tuple(b) for b in avoid] + [tuple(e) for e in examples]
    for c in cands:
        b = c["box"]
        if any(_iou(b, k["box"]) > 0.1 or _centre_in(b, k["box"]) for k in kept):
            continue
        if any(_iou(b, a) > 0.1 or _centre_in(b, a) or _covers(b, a) for a in blocked):
            continue
        kept.append(c)
        if len(kept) >= limit:
            break
    return kept


def _moved_outline(ring_px, ex_box, box, turn):
    """The example's ring (view pixels) carried onto a match box, turned like the match."""
    ex0, ey0, ex1, ey1 = ex_box
    pts = [(x - ex0, y - ey0) for x, y in ring_px]       # in the example box
    w, h = ex1 - ex0, ey1 - ey0
    for _ in range(turn):                                 # np.rot90: (u, v) -> (v, w - u), w and h swap
        pts = [(v, w - u) for u, v in pts]
        w, h = h, w
    sx, sy = (box[2] - box[0]) / w, (box[3] - box[1]) / h
    return [(box[0] + u * sx, box[1] + v * sy) for u, v in pts]


def find_similar(path: str, window: tuple[int, int, int, int], example_rings: list[list[tuple[float, float]]],
                 other_rings: list[list[tuple[float, float]]], opts, device: str = "cpu",
                 threshold: float = 0.6, use_sam: bool | None = None) -> dict:
    """Rings in raster pixels; window (col, row, w, h). -> {objects, timings, outlined_by}."""
    import rasterio
    from rasterio.windows import Window
    from shapely.geometry import Polygon, box as sbox

    from app.engine import click_segment
    from app.engine.raster_prep import read_rgb
    from app.engine.shapes import mask_to_polygon, refine

    col, row, w, h = window
    scale = max(1.0, max(w, h) / MAX_SIDE)
    out_h, out_w = max(1, round(h / scale)), max(1, round(w / scale))
    t0 = time.perf_counter()
    with rasterio.open(path) as ds:
        rgb, _valid = read_rgb(ds, click_segment._plan(path), window=Window(col, row, w, h), out_shape=(out_h, out_w))

    def to_view(ring):
        return [((x - col) / scale, (y - row) / scale) for x, y in ring]

    ex_view = [to_view(r) for r in example_rings]
    ex_boxes = [_ring_box(r) for r in ex_view]
    avoid = [_ring_box(to_view(r)) for r in other_rings]
    matches = match_boxes(rgb, ex_boxes, avoid, threshold)
    match_ms = (time.perf_counter() - t0) * 1000

    if use_sam is None:
        use_sam = bool(click_segment.status()["ready"])
    objects = []
    t1 = time.perf_counter()
    view_poly = sbox(col, row, col + w, row + h)
    if use_sam and matches:
        with click_segment._lock:
            eng, emb, sam_scale, _enc = click_segment._embedded(path, window, device)
            for m in matches:
                bx = m["box"]
                # the match box in the SAM view's pixels
                f = scale / sam_scale
                pts = np.array([[bx[0] * f, bx[1] * f], [bx[2] * f, bx[3] * f]], np.float32)
                try:
                    mask, score = eng.decode(emb, pts, np.array([2, 3], np.float32))
                except Exception:  # noqa: BLE001 - one bad prompt doesn't stop the rest
                    continue
                poly = mask_to_polygon(mask, seed_xy=tuple(pts.mean(axis=0)), fill_holes_px2=opts.px2(opts.fill_holes_m2),
                                       offset=(col, row), scale=sam_scale)
                if poly is not None:
                    objects.append((m, poly))
    else:
        for m in matches:
            ring = _moved_outline(ex_view[m["example"]], ex_boxes[m["example"]], m["box"], m["turn"])
            poly = Polygon([(col + x * scale, row + y * scale) for x, y in ring]).buffer(0)
            if poly.is_empty:
                continue
            if poly.geom_type != "Polygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            objects.append((m, poly.intersection(view_poly) if not view_poly.contains(poly) else poly))
    out = []
    for m, poly in objects:
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        try:
            res = refine(poly, opts)
        except ValueError:
            continue
        res.update(score=round(m["score"], 3))
        out.append(res)
    timings = {"match_ms": round(match_ms, 1), "outline_ms": round((time.perf_counter() - t1) * 1000, 1),
               "candidates": len(matches)}
    return {"objects": out, "timings": timings, "outlined_by": "sam" if use_sam else "example"}
