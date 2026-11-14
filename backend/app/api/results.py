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


WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)


def _result_crs(job_id: str, model_id: str) -> str | None:
    manifest = _result_path(job_id, model_id).with_name(f"{model_id}.manifest.json")
    try:
        return json.loads(manifest.read_text()).get("crs")
    except (OSError, ValueError):
        return None


def _result_path(job_id: str, model_id: str) -> Path:
    """Map (job_id, model_id) to a GeoJSON file, refusing anything that
    would escape RESULTS_DIR (e.g. a job_id of '..')."""
    root = settings.RESULTS_DIR.resolve()
    path = (settings.RESULTS_DIR / job_id / f"{model_id}.geojson").resolve()
    if root not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="No results found for this job/model.")
    return path


@router.get("/{job_id}/{model_id}")
def get_geojson(job_id: str, model_id: str):
    path = _result_path(job_id, model_id)
    return json.loads(path.read_text())


@router.get("/{job_id}/{model_id}/kml")
def get_kml(job_id: str, model_id: str):
    path = _result_path(job_id, model_id)
    geojson = json.loads(path.read_text())
    kml = geojson_to_kml(geojson, document_name=f"Syntra AI — {model_id}")
    return Response(content=kml, media_type="application/vnd.google-earth.kml+xml")


@router.get("/{job_id}/{model_id}/shapefile")
def get_shapefile(job_id: str, model_id: str):
    path = _result_path(job_id, model_id)
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
            # Results are always WGS84 lon/lat (pipeline reprojects); without a .prj,
            # QGIS/ArcGIS ask the user to guess the CRS.
            zf.writestr(f"{model_id}.prj", WGS84_PRJ)
        buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{model_id}_shapefile.zip"'},
    )


@router.get("/{job_id}/{model_id}/gpkg")
def get_gpkg(job_id: str, model_id: str):
    from app.engine.gpkg_export import write_gpkg

    geojson = json.loads(_result_path(job_id, model_id).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        out = write_gpkg(Path(tmp) / f"{model_id}.gpkg", geojson, table=model_id)
        data = out.read_bytes()
    return Response(content=data, media_type="application/geopackage+sqlite3",
                    headers={"Content-Disposition": f'attachment; filename="{model_id}.gpkg"'})


@router.get("/{job_id}/{model_id}/csv")
def get_csv(job_id: str, model_id: str):
    from app.engine.gpkg_export import to_csv

    geojson = json.loads(_result_path(job_id, model_id).read_text())
    return Response(content=to_csv(geojson), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{model_id}.csv"'})


@router.get("/{job_id}/{model_id}/manifest")
def get_manifest(job_id: str, model_id: str):
    path = settings.RESULTS_DIR / job_id / f"{model_id}.manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No manifest found for this job/model.")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Review: accept / reject each result, then turn the review into samples
# ---------------------------------------------------------------------------
from pydantic import BaseModel  # noqa: E402


class ReviewChanges(BaseModel):
    changes: dict[str, str | None]      # {"12": "accept", "40": "reject", "7": null}


def _review_counts(decisions: dict[str, str], total: int) -> dict:
    acc = sum(1 for v in decisions.values() if v == "accept")
    rej = sum(1 for v in decisions.values() if v == "reject")
    return {"total": total, "accepted": acc, "rejected": rej, "open": total - acc - rej}


@router.get("/{job_id}/{model_id}/review")
def get_review(job_id: str, model_id: str):
    from app.engine.review import load_review

    path = _result_path(job_id, model_id)
    d = load_review(path)
    total = len(json.loads(path.read_text()).get("features", []))
    return {"decisions": d, **_review_counts(d, total)}


@router.put("/{job_id}/{model_id}/review")
def put_review(job_id: str, model_id: str, body: ReviewChanges):
    from app.engine.review import save_review

    path = _result_path(job_id, model_id)
    total = len(json.loads(path.read_text()).get("features", []))
    for k in body.changes:
        if not k.isdigit() or int(k) >= total:
            raise HTTPException(status_code=400, detail=f"No result number {k} in this run")
    try:
        d = save_review(path, body.changes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"decisions": d, **_review_counts(d, total)}


@router.post("/{job_id}/{model_id}/review/apply")
def apply_review_to_training(job_id: str, model_id: str):
    """Accepted results become labels, rejected ones hard negatives, on the run's image."""
    from app.core.image_library import find_entry_by_image_id
    from app.core.jobs import get_store
    from app.engine.review import apply_review

    path = _result_path(job_id, model_id)
    job = get_store().get(job_id) or {}
    entry = find_entry_by_image_id(settings.UPLOAD_DIR, job.get("image_id") or "")
    if entry is None:
        raise HTTPException(status_code=409, detail="The image this run used is no longer in the library")
    if not entry.get("bounds_wgs84") or not entry.get("width"):
        raise HTTPException(status_code=409, detail="The image has no geographic bounds yet")
    return apply_review(path, job_id, model_id, entry)
