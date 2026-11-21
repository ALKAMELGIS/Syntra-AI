"""
Bring labels drawn elsewhere (ArcGIS, QGIS, a previous project) into an
image's labels, so they train the same way as labels drawn in Syntra.

The label file standard (Syntra labels v1)
-------------------------------------------
Geometry  Polygon or MultiPolygon, one object per feature (a MultiPolygon's
          parts become separate labels). Points and lines are refused:
          a model learns outlines.
CRS       Any, declared in the file (.prj for a Shapefile). Without one the
          coordinates must be longitude / latitude.
class     text, required: the object's class, e.g. ship, building, tank.
          Read from the first of: class, class_name, classname, label,
          category, type, name (any letter case), or a field the person
          picks, or one class for the whole file. Values are trimmed,
          lower-cased and spaces become one; at most 32 characters.
role      text, optional: object (default, empty is the same), complete
          (a training area: every object inside it is labelled) or
          accuracy (an accuracy area: kept out of training and used to
          score models).
Anything else in the file is ignored.

On import a label is clipped to the image, holes are dropped (labels are
outer rings), a part smaller than 2 px across is skipped, and a label that
already exists on the image (same class, IoU >= 0.9) is skipped, so
importing the same file twice adds nothing.
"""
from __future__ import annotations

import re
from collections import Counter

from app.annotations import store

CLASS_FIELDS = ("class", "class_name", "classname", "label", "category", "type", "name")
ROLE_FIELDS = ("role",)
ROLES = {
    "": None, "object": None, "label": None,
    "complete": store.TRAIN_AREA, "training": store.TRAIN_AREA, "train": store.TRAIN_AREA, "training_area": store.TRAIN_AREA,
    "accuracy": store.CHECK_AREA, "check": store.CHECK_AREA, "test": store.CHECK_AREA, "accuracy_area": store.CHECK_AREA,
}
MIN_PX = 2.0
DUP_IOU = 0.9


def clean_class(v) -> str | None:
    if v is None:
        return None
    s = re.sub(r"\s+", " ", str(v)).strip().lower()
    return s[:32] or None


def _field(props_keys, wanted) -> str | None:
    low = {k.lower(): k for k in props_keys}
    return next((low[w] for w in wanted if w in low), None)


def inspect(fc: dict) -> dict:
    """What a file holds, for the person to map before anything is added."""
    feats = fc.get("features") or []
    keys: list[str] = []
    for f in feats[:2000]:
        for k in (f.get("properties") or {}):
            if k not in keys:
                keys.append(k)
    fields = []
    for k in keys:
        vals = Counter(str(v) for f in feats if (v := (f.get("properties") or {}).get(k)) not in (None, ""))
        fields.append({"name": k, "distinct": len(vals), "values": vals.most_common(12)})
    kinds = Counter((f.get("geometry") or {}).get("type") for f in feats)
    return {
        "features": len(feats),
        "polygons": kinds.get("Polygon", 0) + kinds.get("MultiPolygon", 0),
        "other_geometry": sum(n for t, n in kinds.items() if t not in ("Polygon", "MultiPolygon")),
        "fields": fields,
        "class_field": _field(keys, CLASS_FIELDS),
        "role_field": _field(keys, ROLE_FIELDS),
    }


def to_labels(fc: dict, bounds: dict, width: int, height: int, *, class_field: str | None = None,
              fixed_class: str | None = None, role_field: str | None = None, class_map: dict | None = None,
              existing: list[dict] | None = None) -> tuple[list[dict], dict]:
    """GeoJSON (WGS84) -> label entries ready for store.add_many, and a report."""
    from shapely.geometry import Polygon, box, shape

    frame = box(bounds["west"], bounds["south"], bounds["east"], bounds["north"])
    fixed = clean_class(fixed_class)
    cmap = {str(k): clean_class(v) for k, v in (class_map or {}).items()}
    if not fixed and not class_field:
        raise ValueError("Pick the field that holds the class, or one class for the whole file")
    if role_field and role_field == class_field:
        role_field = None

    def to_px(poly):
        return Polygon([store.lonlat_to_pixel(x, y, bounds, width, height) for x, y in poly.exterior.coords])

    seen = [(a["class_name"], Polygon(a["points_px"]).buffer(0)) for a in existing or [] if len(a.get("points_px") or []) >= 3]
    rep = Counter()
    added_by_class = Counter()
    bad_roles = Counter()
    out = []
    for f in fc.get("features") or []:
        g = f.get("geometry") or {}
        if g.get("type") not in ("Polygon", "MultiPolygon"):
            rep["not_polygon"] += 1
            continue
        props = f.get("properties") or {}
        role_raw = str(props.get(role_field) or "").strip().lower() if role_field else ""
        if role_raw not in ROLES:
            bad_roles[role_raw] += 1
            rep["unknown_role"] += 1
            continue
        area_cls = ROLES[role_raw]
        if area_cls:
            cls = area_cls
        else:
            raw = props.get(class_field) if class_field else None
            cls = cmap.get(str(raw), clean_class(raw)) if class_field else fixed
            if class_field and str(raw) in cmap and cmap[str(raw)] is None:
                rep["left_out"] += 1                       # the person chose to leave this value out
                continue
            cls = cls or fixed
            if not cls:
                rep["no_class"] += 1
                continue
            if cls in store.AREA_CLASSES:
                rep["no_class"] += 1
                continue
        try:
            geom = shape(g).buffer(0)
        except Exception:  # noqa: BLE001 - a broken geometry is skipped and counted
            rep["broken"] += 1
            continue
        if not geom.intersects(frame):
            rep["outside"] += 1
            continue
        clipped = geom.intersection(frame)
        if not geom.difference(frame).is_empty:
            rep["clipped"] += 1
        parts = [p for p in getattr(clipped, "geoms", [clipped]) if p.geom_type == "Polygon" and not p.is_empty]
        for poly in parts:
            if poly.interiors:
                rep["holes_dropped"] += 1
                poly = Polygon(poly.exterior)
            px = to_px(poly)
            x0, y0, x1, y1 = px.bounds
            if x1 - x0 < MIN_PX or y1 - y0 < MIN_PX:
                rep["too_small"] += 1
                continue
            pxv = px.buffer(0)
            dup = False
            for c2, p2 in seen:
                if c2 == cls and p2.intersects(pxv):
                    u = p2.union(pxv).area
                    if u and p2.intersection(pxv).area / u >= DUP_IOU:
                        dup = True
                        break
            if dup:
                rep["duplicates"] += 1
                continue
            seen.append((cls, pxv))
            ring = [list(c) for c in poly.exterior.coords][:-1]
            out.append({"class_name": cls, "points_lonlat": ring})
            added_by_class[cls] += 1
    report = {"added": len(out), "by_class": dict(added_by_class.most_common()), "skipped": dict(rep)}
    if bad_roles:
        report["unknown_roles"] = [r for r, _ in bad_roles.most_common(5)]
    return out, report
