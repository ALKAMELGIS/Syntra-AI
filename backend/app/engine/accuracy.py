"""
Accuracy of a model on the user's own imagery.

An accuracy area is a polygon the analyst labelled completely and kept out
of training. The model is run inside it, and each prediction is matched to
a label of the same class with IoU >= the threshold (greedy, highest score
first). Per class: true positives, false positives, missed objects,
precision, recall, F1. This is the check ArcGIS Pro's "Compute Accuracy For
Object Detection" and Picterra's accuracy areas provide.

Geometry is compared in lon/lat: over an accuracy area IoU is a ratio of
areas, which a local scale change leaves unchanged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def _poly(coords):
    from shapely.geometry import Polygon

    p = Polygon(coords)
    return p if p.is_valid else p.buffer(0)


def match(preds: list[dict], truth: list[dict], iou_min: float = 0.5) -> dict:
    """preds: [{class, score, geom}], truth: [{class, geom}] -> per-class counts and scores."""
    classes = sorted({t["class"] for t in truth} | {p["class"] for p in preds})
    out = {}
    for c in classes:
        P = sorted([p for p in preds if p["class"] == c], key=lambda p: -p.get("score", 0))
        T = [t for t in truth if t["class"] == c]
        used = [False] * len(T)
        tp = 0
        for p in P:
            best, bi = 0.0, -1
            for i, t in enumerate(T):
                if used[i] or not p["geom"].intersects(t["geom"]):
                    continue
                inter = p["geom"].intersection(t["geom"]).area
                iou = inter / (p["geom"].area + t["geom"].area - inter or 1e-12)
                if iou > best:
                    best, bi = iou, i
            if bi >= 0 and best >= iou_min:
                used[bi] = True
                tp += 1
        fp, fn = len(P) - tp, len(T) - tp
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if (tp + fp + fn) else None)
        out[c] = {"tp": tp, "fp": fp, "fn": fn, "labels": len(T), "predictions": len(P),
                  "precision": _r(prec), "recall": _r(rec), "f1": _r(f1)}
    tp = sum(v["tp"] for v in out.values()); fp = sum(v["fp"] for v in out.values()); fn = sum(v["fn"] for v in out.values())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    return {"per_class": out, "overall": {"tp": tp, "fp": fp, "fn": fn, "precision": _r(prec), "recall": _r(rec), "f1": _r(f1)}}


def _r(v):
    return None if v is None else round(v, 3)


def class_aliases(model_classes: list[str], label_classes: list[str]) -> dict[str, str]:
    """model class -> label class: same name (case-insensitive); a one-class model
    against one-class labels is matched whatever the names (\"ship\" vs \"boat\")."""
    lab = {c.lower(): c for c in label_classes}
    m = {c: lab[c.lower()] for c in model_classes if c.lower() in lab}
    if not m and len(model_classes) == 1 and len(label_classes) == 1:
        m = {model_classes[0]: label_classes[0]}
    return m


def evaluate(geojson: dict, annotations: list[dict], check_areas: list[dict], model_classes: list[str],
             iou_min: float = 0.5) -> dict:
    from shapely.ops import unary_union

    zone = unary_union([_poly(a["points_lonlat"]) for a in check_areas])
    truth = [{"class": a["class_name"], "geom": _poly(a["points_lonlat"])} for a in annotations
             if not a["class_name"].startswith("__") and zone.contains(_poly(a["points_lonlat"]).centroid)]
    aliases = class_aliases(model_classes, sorted({t["class"] for t in truth}))
    preds = []
    for f in geojson.get("features", []):
        g = f.get("geometry") or {}
        cls = (f.get("properties") or {}).get("class")
        if g.get("type") != "Polygon" or cls not in aliases:
            continue
        geom = _poly(g["coordinates"][0])
        if zone.contains(geom.centroid):
            preds.append({"class": aliases[cls], "score": (f.get("properties") or {}).get("score", 0), "geom": geom})
    res = match(preds, truth, iou_min)
    res["matched_classes"] = aliases
    res["unmatched_label_classes"] = sorted({t["class"] for t in truth} - set(aliases.values()))
    res["area_km2"] = None
    return res


def _store(data_dir: Path) -> Path:
    d = Path(data_dir) / "accuracy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_check(data_dir: Path, model_id: str, record: dict) -> None:
    p = _store(data_dir) / f"{model_id}.json"
    hist = json.loads(p.read_text()) if p.exists() else []
    hist.insert(0, record)
    p.write_text(json.dumps(hist[:20], indent=1))


def checks(data_dir: Path, model_id: str) -> list[dict]:
    p = _store(data_dir) / f"{model_id}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return []


def latest_f1(data_dir: Path, model_id: str) -> float | None:
    c = checks(data_dir, model_id)
    return c[0]["overall"]["f1"] if c else None


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
