"""Serve exported GeoJSON + manifest files for a completed run."""
from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response

from app.core.config import settings
from app.engine.export_formats import geojson_to_kml
from app.engine.shapefile_export import write_shapefile

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/{job_id}/{model_id}")
def get_geojson(job_id: str, model_id: str):
    path = settings.RESULTS_DIR / job_id / f"{model_id}.geojson"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No results found for this job/model.")
    return json.loads(path.read_text())


@router.get("/{job_id}/{model_id}/kml")
def get_kml(job_id: str, model_id: str):
    path = settings.RESULTS_DIR / job_id / f"{model_id}.geojson"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No results found for this job/model.")
    geojson = json.loads(path.read_text())
    kml = geojson_to_kml(geojson, document_name=f"Syntra AI — {model_id}")
    return Response(content=kml, media_type="application/vnd.google-earth.kml+xml")


@router.get("/{job_id}/{model_id}/shapefile")
def get_shapefile(job_id: str, model_id: str):
    path = settings.RESULTS_DIR / job_id / f"{model_id}.geojson"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No results found for this job/model.")
    geojson = json.loads(path.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        try:
            paths = write_shapefile(Path(tmp) / model_id, geojson)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for key in ("shp", "shx", "dbf"):
                zf.write(paths[key], arcname=f"{model_id}.{key}")
        buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{model_id}_shapefile.zip"'},
    )


@router.get("/{job_id}/{model_id}/manifest")
def get_manifest(job_id: str, model_id: str):
    path = settings.RESULTS_DIR / job_id / f"{model_id}.manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No manifest found for this job/model.")
    return json.loads(path.read_text())
