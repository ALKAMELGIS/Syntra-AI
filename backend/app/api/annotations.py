"""
Per-image object annotations — the real "Annotate" step. Boxes/polygons
drawn on the map (lon/lat) are converted to the raster's pixel grid and
persisted next to the image, then can be handed straight to the Dataset
export step (see `/export` below, and the `image_id` shortcut on
POST /api/training/export-dataset) instead of pasting pixel JSON by hand.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.annotations import store
from app.core.config import settings
from app.core.image_library import find_entry_by_image_id

router = APIRouter(prefix="/api/images/{image_id}/annotations", tags=["annotations"])


class AddAnnotationRequest(BaseModel):
    class_name: str
    points_lonlat: list[list[float]]   # closed/open ring of [lon, lat]


def _entry_or_404(image_id: str) -> dict:
    settings.ensure_dirs()
    entry = find_entry_by_image_id(settings.UPLOAD_DIR, image_id)
    if entry is None:
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
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return entry_ann


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
