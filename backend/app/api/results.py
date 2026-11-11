"""Serve exported GeoJSON + manifest files for a completed run."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import settings

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/{job_id}/{model_id}")
def get_geojson(job_id: str, model_id: str):
    path = settings.RESULTS_DIR / job_id / f"{model_id}.geojson"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No results found for this job/model.")
    return json.loads(path.read_text())


@router.get("/{job_id}/{model_id}/manifest")
def get_manifest(job_id: str, model_id: str):
    path = settings.RESULTS_DIR / job_id / f"{model_id}.manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No manifest found for this job/model.")
    return json.loads(path.read_text())
