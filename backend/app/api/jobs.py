"""Job history across every kind of background job (app.core.jobs).

GET /api/jobs?kind=predict&limit=20  -> newest first
GET /api/jobs/{job_id}               -> one job of any kind
POST /api/jobs/{job_id}/hide         -> not restored any more (files kept)

Lets the dashboard restore what was running or finished after a page
reload or a server restart, instead of forgetting every job id it held in
memory.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core.jobs import get_store

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

_KINDS = {"predict", "dataset_export", "train", "environment", "model_install", "accuracy", "optimize", "dataset_push", "segment_install", "refine", "model_push"}


@router.get("")
def list_jobs(
    kind: str | None = Query(default=None, description="predict | dataset_export | train | environment"),
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    image_id: str | None = Query(default=None, description="only jobs run on this image"),
    include_hidden: bool = False,
):
    if kind is not None and kind not in _KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown kind '{kind}'. Use one of: {', '.join(sorted(_KINDS))}.")
    jobs = get_store().list(kind, limit=500 if (image_id or not include_hidden) else limit, status=status)
    jobs = [j for j in jobs if (include_hidden or not j.get("hidden")) and (not image_id or j.get("image_id") == image_id)][:limit]
    for job in jobs:
        # Training/export logs can be long; the list view only needs a summary.
        job.pop("log", None)
        job.pop("history", None)
        job.pop("log_tail", None)
    return jobs


@router.post("/{job_id}/hide")
def hide_job(job_id: str):
    """Take a finished run off the map for good (its files stay on disk):
    the dashboard no longer restores it after a reload."""
    store = get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    store.update(job_id, hidden=True)
    return {"job_id": job_id, "hidden": True}


@router.get("/{job_id}")
def get_job(job_id: str):
    job = get_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job
