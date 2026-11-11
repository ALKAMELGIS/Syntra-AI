"""
Dataset management API — spec Section 9.

Validates an already-uploaded set of annotations (YOLO/COCO/GeoJSON),
reports class balance, computes a geographically separated split, and
registers a versioned snapshot of the result.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.datasets.annotations import (
    AnnotationError,
    parse_coco_json,
    parse_geojson_annotations,
    validate_dataset,
)
from app.datasets.schema import AnnotationFormat
from app.datasets.splitter import class_balance, geographically_separated_split
from app.datasets.versioning import list_versions, load_version, register_version

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class PrepareDatasetRequest(BaseModel):
    model_id: str
    annotation_format: AnnotationFormat
    annotation_path: str
    images_dir: str | None = None   # required for COCO
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    cell_size_m: float = 1000.0


@router.post("/prepare")
def prepare_dataset(req: PrepareDatasetRequest):
    try:
        if req.annotation_format == AnnotationFormat.COCO:
            if not req.images_dir:
                raise HTTPException(status_code=400, detail="images_dir is required for COCO annotations.")
            samples = parse_coco_json(req.annotation_path, req.images_dir)
        elif req.annotation_format == AnnotationFormat.GEOJSON:
            samples = parse_geojson_annotations(req.annotation_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="YOLO parsing needs one .txt per image; use the dataset's directory layout "
                       "via a future /datasets/prepare-yolo-dir endpoint, or submit per-image via annotations.parse_yolo_txt directly.",
            )
    except AnnotationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    problems = validate_dataset(samples)
    if problems:
        raise HTTPException(status_code=422, detail={"message": "Dataset failed validation.", "problems": problems})

    balance = class_balance(samples)
    split = geographically_separated_split(
        samples, train_frac=req.train_frac, val_frac=req.val_frac, test_frac=req.test_frac, cell_size=req.cell_size_m,
    )

    settings.ensure_dirs()
    version = register_version(
        settings.DATA_DIR, req.model_id, [s.sample_id for s in samples],
        balance, split, req.annotation_format, [req.annotation_path],
    )

    return {
        "version_id": version.version_id,
        "sample_count": version.sample_count,
        "class_balance": balance.summary(),
        "underrepresented_classes": balance.underrepresented_classes,
        "split": {
            "train": len(split.train), "val": len(split.val), "test": len(split.test),
            "geographically_separated": split.geographically_separated,
            "notes": split.notes,
        },
    }


@router.get("/{model_id}/versions")
def versions(model_id: str):
    return list_versions(settings.DATA_DIR, model_id)


@router.get("/{model_id}/versions/{version_id}")
def version_detail(model_id: str, version_id: str):
    try:
        return load_version(settings.DATA_DIR, model_id, version_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
