"""
Per-image object annotations — the real "Annotate" step. Boxes/polygons
drawn on the map (lon/lat) are converted to the raster's pixel grid and
persisted next to the image, then can be handed straight to the Dataset
export step (see `/export` below, and the `image_id` shortcut on
POST /api/training/export-dataset) instead of pasting pixel JSON by hand.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.annotations import store
from app.core.config import settings
from app.core.image_library import find_entry_by_image_id

router = APIRouter(prefix="/api/images/{image_id}/annotations", tags=["annotations"])


class AddAnnotationRequest(BaseModel):
    class_name: str
    points_lonlat: list[list[float]]   # closed/open ring of [lon, lat]
    shape: str | None = None           # how it was drawn or fitted: polygon, rectangle, circle, right_angles…
    tool: str | None = None            # which tool made it: polygon, rect, square, circle, freehand, click


def _entry_or_404(image_id: str) -> dict:
    settings.ensure_dirs()
    entry = find_entry_by_image_id(settings.UPLOAD_DIR, image_id)
    from app.core.auth import can_see_any, current_request
    req = current_request()
    if entry is None or (req is not None and not can_see_any(req, entry.get("owners"))):
        raise HTTPException(status_code=404, detail=f"No library image with id '{image_id}'.")
    if not entry.get("bounds_wgs84") or not entry.get("width") or not entry.get("height"):
        raise HTTPException(
            status_code=409,
            detail="This image has no geographic bounds or dimensions yet — re-upload or wait for metadata.",
        )
    return entry


@router.get("")
def list_annotations(image_id: str):
    entry = _entry_or_404(image_id)
    anns = store.load_annotations(entry["path"])
    return {"image_id": image_id, "count": len(anns), "class_names": store.class_names_used(anns),
            "annotations": anns}


@router.post("")
def add_annotation(image_id: str, req: AddAnnotationRequest):
    entry = _entry_or_404(image_id)
    try:
        entry_ann = store.add_annotation(
            entry["path"], req.class_name, req.points_lonlat,
            entry["bounds_wgs84"], entry["width"], entry["height"],
            extra={k: v for k, v in (("shape", req.shape), ("tool", req.tool)) if v},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return entry_ann


class UpdateAnnotationRequest(BaseModel):
    points_lonlat: list[list[float]] | None = None
    class_name: str | None = None


@router.patch("/{annotation_id}")
def update_annotation(image_id: str, annotation_id: str, req: UpdateAnnotationRequest):
    entry = _entry_or_404(image_id)
    if req.points_lonlat is None and req.class_name is None:
        raise HTTPException(status_code=400, detail="Send points_lonlat, class_name or both.")
    try:
        a = store.update_annotation(entry["path"], annotation_id, entry["bounds_wgs84"], entry["width"], entry["height"],
                                    points_lonlat=req.points_lonlat, class_name=req.class_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if a is None:
        raise HTTPException(status_code=404, detail=f"No annotation '{annotation_id}' on this image.")
    return a


@router.delete("/{annotation_id}")
def delete_annotation(image_id: str, annotation_id: str):
    entry = _entry_or_404(image_id)
    if not store.delete_annotation(entry["path"], annotation_id):
        raise HTTPException(status_code=404, detail=f"No annotation '{annotation_id}' on this image.")
    return {"deleted": annotation_id}


@router.delete("")
def clear_annotations(image_id: str):
    entry = _entry_or_404(image_id)
    store.clear_annotations(entry["path"])
    return {"cleared": True}


@router.get("/export")
def export_annotations(image_id: str):
    """Pixel-space payload ready for POST /api/training/export-dataset's
    `class_names` + `annotations` fields."""
    entry = _entry_or_404(image_id)
    anns = store.load_annotations(entry["path"])
    if not anns:
        raise HTTPException(status_code=409, detail="No annotations saved for this image yet.")
    class_names, annotations = store.to_export_payload(anns)
    return {"image_id": image_id, "image_path": entry["path"], "class_names": class_names,
            "annotations": annotations}


# --- labels from a file (Shapefile zip, GeoPackage, GeoJSON, KML, FlatGeobuf) ---
@router.post("/import/preview")
async def import_preview(image_id: str, file: UploadFile = File(...)):
    """What the file holds and which fields look like class and role; nothing is added."""
    from app.annotations import importer
    from app.api.vector import receive_vector

    entry = _entry_or_404(image_id)
    got = await receive_vector(file)
    info = importer.inspect(got["geojson"])
    b = entry["bounds_wgs84"]
    from shapely.geometry import box, shape

    frame = box(b["west"], b["south"], b["east"], b["north"])
    on_image = 0
    for f in got["geojson"]["features"]:
        try:
            on_image += bool(f.get("geometry")) and shape(f["geometry"]).intersects(frame)
        except Exception:  # noqa: BLE001
            pass
    return {"name": got["name"], "source_crs": got["source_crs"], "note": got["note"], "on_image": on_image, **info}


@router.post("/import")
async def import_labels(image_id: str, file: UploadFile = File(...), class_field: str | None = Form(None),
                        fixed_class: str | None = Form(None), role_field: str | None = Form(None),
                        class_map: str | None = Form(None)):
    """Add the file's polygons as labels (see app/annotations/importer.py for the standard)."""
    from app.annotations import importer
    from app.api.vector import receive_vector

    entry = _entry_or_404(image_id)
    try:
        cmap = json.loads(class_map) if class_map else None
        if cmap is not None and not isinstance(cmap, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="class_map must be a JSON object of file value -> class.") from None
    got = await receive_vector(file)

    def work():
        items, report = importer.to_labels(
            got["geojson"], entry["bounds_wgs84"], entry["width"], entry["height"], class_field=class_field or None,
            fixed_class=fixed_class or None, role_field=role_field or None, class_map=cmap,
            existing=store.load_annotations(entry["path"]))
        store.add_many(entry["path"], items, entry["bounds_wgs84"], entry["width"], entry["height"],
                       extra={"tool": "import", "source": got["name"]})
        return report

    try:
        report = await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"name": got["name"], **report, "annotations": store.load_annotations(entry["path"])}
