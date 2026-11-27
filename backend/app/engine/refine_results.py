"""
Refine a detection layer's outlines with SAM 2.1, then measure a layer
against the labels in the accuracy areas.

Refine: every object the detector found becomes a box prompt for SAM 2.1
(the same ONNX model the click tool uses). SAM traces the object inside the
box, the trace is fitted to the class's shape (right angles for buildings,
a rotated rectangle for ships and vehicles, a circle for tanks; see
shapes.py), and the outline goes back to WGS84 through the image's own
transform. An object keeps its detector outline, marked so, when SAM is not
sure (low score), when the trace does not overlap the detection, or when
it is far bigger or smaller than the box: a refinement never invents a
shape.

Score: labels and results inside the accuracy areas are matched one to one
by IoU (greedy, best first). precision = matched / results, recall =
matched / labels, F1, and the mean IoU of the matches: how well the
outlines themselves fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CELL = 1024              # one SAM encode covers a cell of this many native pixels
MIN_SCORE = 0.70         # SAM's own confidence below this: keep the detector outline
MIN_OVERLAP = 0.30       # the trace must overlap the detection this much (IoU)
AREA_RANGE = (0.15, 1.6)  # trace area / detection area outside this: keep the detector outline
_MEASURED = {"area_m2", "perimeter_m", "length_m", "width_m", "orientation_deg", "lon", "lat", "area_sqm", "area_ha"}


@dataclass
class Geo:
    """Exact lon/lat <-> raster pixel through the image's CRS and transform."""
    to_px: object
    to_ll: object
    inv: object
    fwd: object

    @classmethod
    def open(cls, ds) -> "Geo":
        from pyproj import Transformer

        if ds.crs is None:
            raise ValueError("This image has no coordinate system, so outlines cannot be placed")
        t_in = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        t_out = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
        return cls(t_in, t_out, ~ds.transform, ds.transform)

    def px(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = self.to_px.transform(lon, lat)
        c, r = self.inv * (x, y)
        return float(c), float(r)

    def ll(self, col: float, row: float) -> tuple[float, float]:
        x, y = self.fwd * (col, row)
        lon, lat = self.to_ll.transform(x, y)
        return float(lon), float(lat)


def _outer_ring(geom: dict) -> list | None:
    t = geom.get("type") if geom else None
    if t == "Polygon":
        return geom["coordinates"][0]
    if t == "MultiPolygon":
        rings = [p[0] for p in geom["coordinates"] if p and p[0]]
        return max(rings, key=len) if rings else None
    return None


def _cell_of(box, width, height):
    """The CELL window that holds a box with room around it, or None if it is too big for one."""
    x0, y0, x1, y1 = box
    if x1 - x0 > CELL * 0.7 or y1 - y0 > CELL * 0.7:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    c0 = int(min(max(0, cx - CELL / 2), max(0, width - CELL)))
    r0 = int(min(max(0, cy - CELL / 2), max(0, height - CELL)))
    # snap to a grid so neighbours share one encode
    c0 = min(int(c0 // (CELL / 2) * (CELL / 2)), max(0, width - CELL))
    r0 = min(int(r0 // (CELL / 2) * (CELL / 2)), max(0, height - CELL))
    if not (c0 <= x0 and x1 <= c0 + CELL and r0 <= y0 and y1 <= r0 + CELL):
        c0 = int(min(max(0, cx - CELL / 2), max(0, width - CELL)))
        r0 = int(min(max(0, cy - CELL / 2), max(0, height - CELL)))
    return (c0, r0, min(CELL, width - c0), min(CELL, height - r0))


def refine_layer(path: str, fc: dict, device: str = "cpu", engine=None, progress=None) -> tuple[dict, dict]:
    """A FeatureCollection in WGS84 -> (refined FeatureCollection, stats)."""
    import rasterio
    from rasterio.windows import Window
    from shapely.geometry import Polygon

    from app.engine import click_segment
    from app.engine.measure import measure_features
    from app.engine.raster_prep import read_rgb
    from app.engine.shapes import RefineOptions, mask_to_polygon, refine, shape_for_class

    eng = engine or click_segment._get_engine(device)
    plan = click_segment._plan(path)
    feats = fc.get("features") or []
    out, stats = [], {"objects": len(feats), "refined": 0, "kept": 0, "skipped": 0, "reasons": {}}

    def keep(f, why):
        g = dict(f)
        g["properties"] = {**(f.get("properties") or {}), "outline": "detector", "refine_note": why}
        out.append(g)
        stats["kept"] += 1
        stats["reasons"][why] = stats["reasons"].get(why, 0) + 1

    with rasterio.open(path) as ds:
        geo = Geo.open(ds)
        gsd = abs(ds.transform.a) if ds.crs and ds.crs.is_projected else None
        width, height = ds.width, ds.height
        # every object's pixel ring and box, grouped by the window that encodes it
        jobs: dict = {}
        for i, f in enumerate(feats):
            ring = _outer_ring(f.get("geometry"))
            if not ring or len(ring) < 3:
                out.append(f)
                stats["skipped"] += 1
                continue
            px = [geo.px(lon, lat) for lon, lat in ring[:-1]]
            xs, ys = [p[0] for p in px], [p[1] for p in px]
            box = (max(0.0, min(xs)), max(0.0, min(ys)), min(width, max(xs)), min(height, max(ys)))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                keep(f, "too small")
                continue
            win = _cell_of(box, width, height)
            if win is None:                              # a big object: its own window, read smaller
                pad = 0.1 * max(box[2] - box[0], box[3] - box[1])
                c0, r0 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad))
                win = (c0, r0, int(min(width, box[2] + pad)) - c0, int(min(height, box[3] + pad)) - r0)
            jobs.setdefault(win, []).append((i, f, px, box))

        done = 0
        for win, items in jobs.items():
            col, row, w, h = win
            scale = max(1.0, max(w, h) / click_segment.SIZE)
            out_h, out_w = max(1, round(h / scale)), max(1, round(w / scale))
            rgb, _valid = read_rgb(ds, plan, window=Window(col, row, w, h), out_shape=(out_h, out_w))
            emb = eng.encode(rgb)
            for i, f, px, box in items:
                done += 1
                if progress:
                    progress(done, len(feats))
                props = f.get("properties") or {}
                det = Polygon(px)
                if not det.is_valid:
                    det = det.buffer(0)
                pts = np.array([[(box[0] - col) / scale, (box[1] - row) / scale],
                                [(box[2] - col) / scale, (box[3] - row) / scale]], dtype=np.float32)
                mask, score = eng.decode(emb, pts, np.array([2, 3], dtype=np.float32))   # SAM box prompt
                if score < MIN_SCORE:
                    keep(f, "SAM not sure")
                    continue
                seed = ((box[0] + box[2]) / 2 - col) / scale, ((box[1] + box[3]) / 2 - row) / scale
                poly = mask_to_polygon(mask, seed_xy=seed, fill_holes_px2=(6.0 / (gsd * gsd)) if gsd else 0.0,
                                       offset=(col, row), scale=scale)
                if poly is None or poly.is_empty:
                    keep(f, "nothing traced")
                    continue
                ratio = poly.area / max(det.area, 1e-9)
                inter = poly.intersection(det).area
                iou = inter / max(poly.union(det).area, 1e-9)
                if not (AREA_RANGE[0] <= ratio <= AREA_RANGE[1]) or iou < MIN_OVERLAP:
                    keep(f, "trace does not match the detection")
                    continue
                cls = str(props.get("class") or "")
                res = refine(poly, RefineOptions(shape=shape_for_class(cls), px_m=gsd))
                ring = list(res["polygon"].exterior.coords)
                ll = [list(geo.ll(x, y)) for x, y in ring]
                if ll[0] != ll[-1]:
                    ll.append(ll[0])
                kept = {k: v for k, v in props.items() if k not in _MEASURED}   # re-measured on the new outline
                out.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ll]},
                            "properties": {**kept, "outline": "sam", "sam_score": round(score, 3),
                                           "shape": res.get("shape"), "fit_iou": res.get("iou")}})
                stats["refined"] += 1
    result = measure_features({"type": "FeatureCollection", "features": out})
    return result, stats


def score_layer(results: list[dict], labels: list[dict], areas: list[dict], class_name: str | None = None,
                iou_min: float = 0.5) -> dict:
    """Results and labels (GeoJSON features, WGS84) matched inside the accuracy areas."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if not areas:
        raise ValueError("Draw an accuracy area and label every object inside it first")
    zone = unary_union([shape(a["geometry"]) for a in areas])

    def pick(feats, key):
        out = []
        for f in feats:
            if class_name and str((f.get("properties") or {}).get(key, "")).lower() != class_name.lower():
                continue
            try:
                g = shape(f["geometry"])
            except Exception:  # noqa: BLE001 - a broken geometry is not counted
                continue
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty and g.representative_point().within(zone):
                out.append(g)
        return out

    pred, truth = pick(results, "class"), pick(labels, "class_name")
    pairs = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            if p.intersects(t):
                inter = p.intersection(t).area
                iou = inter / max(p.union(t).area, 1e-18)
                if iou >= iou_min:
                    pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_p, used_t, ious = set(), set(), []
    for iou, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j); ious.append(iou)
    tp = len(ious)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"results": len(pred), "labels": len(truth), "matched": tp,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "iou_min": iou_min, "areas": len(areas), "class": class_name}
