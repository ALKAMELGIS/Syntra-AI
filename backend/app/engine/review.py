"""
Accept / reject a run's results, and turn the decisions into training samples.

The decisions are kept beside the result (`<model>.review.json`), so a page
refresh or another session sees the same review. Applying them writes to the
image's annotations, tagged with the job and the result index, so applying
again replaces what an earlier apply wrote instead of adding duplicates.

  accepted -> a label of the result's class + a small training area around it
  rejected -> a small training area around it with no label (a hard negative:
              "this is not a ship")

Why a small area per object: training tiles are masked to the training areas
(app/api/training.py area_filter), so a tight area says "inside here every
object is labelled" without claiming anything about the rest of the tile,
where unreviewed objects would otherwise be taught as background.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from app.annotations import store

PAD_FRACTION = 0.25          # area = the object's box + 25% of its size on each side
PAD_MIN_PX = 8
DECISIONS = ("accept", "reject")


def review_path(result_geojson: Path) -> Path:
    return result_geojson.with_name(result_geojson.stem + ".review.json")


def load_review(result_geojson: Path) -> dict[str, str]:
    try:
        d = json.loads(review_path(result_geojson).read_text()).get("decisions", {})
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in d.items() if v in DECISIONS}


def save_review(result_geojson: Path, changes: dict[str, str | None]) -> dict[str, str]:
    """Merge `changes` ({"12": "accept", "40": None}) into the saved review."""
    d = load_review(result_geojson)
    for k, v in changes.items():
        if v is None:
            d.pop(str(k), None)
        elif v in DECISIONS:
            d[str(k)] = v
        else:
            raise ValueError(f"Unknown decision '{v}': use accept, reject or null")
    p = review_path(result_geojson)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"decisions": d}, indent=1))
    os.replace(tmp, p)
    return d


def _ring(feature: dict) -> list[list[float]] | None:
    g = feature.get("geometry") or {}
    if g.get("type") == "Polygon" and g.get("coordinates"):
        return [list(p[:2]) for p in g["coordinates"][0]]
    if g.get("type") == "MultiPolygon" and g.get("coordinates"):
        return [list(p[:2]) for p in g["coordinates"][0][0]]
    return None


def apply_review(result_geojson: Path, job_id: str, model_id: str, entry: dict) -> dict:
    """Write the review into the image's training annotations. Returns counts."""
    fc = json.loads(Path(result_geojson).read_text())
    feats = fc.get("features", [])
    decisions = load_review(result_geojson)
    bounds, width, height = entry["bounds_wgs84"], entry["width"], entry["height"]
    source = {"job_id": job_id, "model_id": model_id}

    anns = [a for a in store.load_annotations(entry["path"])
            if not (a.get("source", {}).get("job_id") == job_id and a.get("source", {}).get("model_id") == model_id)]
    import time
    import uuid

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    accepted = rejected = skipped = 0
    for key, decision in sorted(decisions.items(), key=lambda kv: int(kv[0])):
        i = int(key)
        ring = _ring(feats[i]) if 0 <= i < len(feats) else None
        if not ring or len(ring) < 3:
            skipped += 1
            continue
        px = [store.lonlat_to_pixel(lon, lat, bounds, width, height) for lon, lat in ring]
        xs, ys = [p[0] for p in px], [p[1] for p in px]
        pad = max(PAD_MIN_PX, PAD_FRACTION * max(max(xs) - min(xs), max(ys) - min(ys)))
        x0, y0 = max(0.0, min(xs) - pad), max(0.0, min(ys) - pad)
        x1, y1 = min(float(width), max(xs) + pad), min(float(height), max(ys) + pad)
        area_px = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        area_ll = [list(store.pixel_to_lonlat(x, y, bounds, width, height)) for x, y in area_px]
        tag = {**source, "index": i, "decision": decision}
        anns.append({"id": str(uuid.uuid4()), "class_name": store.TRAIN_AREA, "points_lonlat": area_ll,
                     "points_px": area_px, "created_at": now, "source": tag})
        if decision == "accept":
            cls = str((feats[i].get("properties") or {}).get("class") or "object")
            anns.append({"id": str(uuid.uuid4()), "class_name": cls, "points_lonlat": ring,
                         "points_px": [list(p) for p in px], "created_at": now, "source": tag})
            accepted += 1
        else:
            rejected += 1
    store.save_annotations(entry["path"], anns)
    return {"accepted": accepted, "rejected": rejected, "skipped": skipped, "annotations": len(anns)}
