"""Training samples on Hugging Face: the token, the repo, and pushing an export."""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core.config import settings
from app.core.jobs import get_store
from app.core.secrets import get_secret, set_secret

router = APIRouter(prefix="/api/samples", tags=["samples"])
PUSH_KIND = "dataset_push"
_TOKEN, _REPO, _USER, _AUTO = "hf_token", "hf_repo", "hf_user", "hf_auto_backup"


class HfSettings(BaseModel):
    token: str | None = None      # empty string = forget the token
    repo: str | None = None       # "<user>/<name>"; default <user>/syntra-samples
    auto_backup: bool | None = None  # push every new library dataset version by itself


class PushRequest(BaseModel):
    dataset_job_id: str
    allow_public: bool = False


def _library_ok() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


@router.get("/hf")
def hf_status():
    return {"configured": bool(get_secret(_TOKEN)), "user": get_secret(_USER), "repo": get_secret(_REPO),
            "library": _library_ok(), "auto_backup": get_secret(_AUTO) == "1"}


@router.put("/hf")
async def hf_save(body: HfSettings):
    from app.engine.hf_push import PushError, whoami

    if body.auto_backup is not None:
        set_secret(_AUTO, "1" if body.auto_backup else None)
    if body.token is not None:
        token = body.token.strip()
        if not token:
            for k in (_TOKEN, _USER):
                set_secret(k, None)
        else:
            try:
                user = await run_in_threadpool(whoami, token)
            except PushError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            set_secret(_TOKEN, token)
            set_secret(_USER, user)
            if not (body.repo or get_secret(_REPO)):
                set_secret(_REPO, f"{user}/syntra-samples")
    if body.repo is not None:
        repo = body.repo.strip()
        if repo and (repo.count("/") != 1 or not all(repo.split("/"))):
            raise HTTPException(status_code=400, detail="Use the form name/repo, for example ali/syntra-samples")
        set_secret(_REPO, repo or None)
    return hf_status()


def _export_dir(job_id: str) -> tuple[Path, dict]:
    job = get_store().get(job_id) or {}
    if job.get("kind") not in (None, "dataset_export") or job.get("status") != "completed" or not job.get("data_yaml"):
        raise HTTPException(status_code=409, detail="Export the samples first; this export is not finished")
    d = Path(job["data_yaml"]).parent.resolve()
    root = settings.DATA_DIR.resolve()
    if root not in d.parents:
        raise HTTPException(status_code=403, detail="That export is outside the data folder")
    return d, job


def _source_meta(job: dict) -> dict:
    from app.core.image_library import find_entry_by_image_id

    entry = find_entry_by_image_id(settings.UPLOAD_DIR, job.get("image_id") or "") or {}
    health = entry.get("health") or {}
    info = job.get("info") or {}
    from app.core import layout

    manifest = layout.read_json(Path(job["data_yaml"]).parent / "manifest.json", {}) if job.get("data_yaml") else {}
    return {"dataset": manifest.get("dataset"), "version": manifest.get("version"),
            "label_files": [lf["file"] for lf in manifest.get("labels", [])], "classes_dir": str(layout.classes_dir()),
            "image_name": entry.get("filename") or job.get("image_name"), "image_id": job.get("image_id"),
            "crs": health.get("crs") or entry.get("crs"), "bounds_wgs84": entry.get("bounds_wgs84"),
            "gsd_m": health.get("gsd_m") or entry.get("gsd_x_m"), "task": job.get("task") or "detect",
            "classes": job.get("class_names"), "export": info, "made_with": "Syntra AI"}


def _run_push(job_id: str, dataset_dir: Path, repo: str, token: str, source: dict, allow_public: bool) -> None:
    from app.engine.hf_push import PushError, push_dataset

    jobs = get_store()
    jobs.update(job_id, status="processing")

    def progress(pct, msg=None):
        jobs.update(job_id, progress=pct, **({"message": msg} if msg else {}))

    try:
        result = push_dataset(dataset_dir, repo, token, source, allow_public=allow_public, progress=progress)
        from app.core.layout import write_json
        write_json(dataset_dir / "pushed.json", {**result, "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        jobs.update(job_id, status="completed", result=result)
    except PushError as exc:
        jobs.update(job_id, status="error", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - network errors are shown as they are
        jobs.update(job_id, status="error", error=f"Upload failed: {type(exc).__name__}: {exc}")


def auto_backup(dataset_job_id: str) -> str | None:
    """After a library build: push the new version when the owner turned automatic backup on."""
    token, repo = get_secret(_TOKEN), get_secret(_REPO)
    if get_secret(_AUTO) != "1" or not token or not repo:
        return None
    dataset_dir, job = _export_dir(dataset_job_id)
    job_id = get_store().create(PUSH_KIND, progress=0.0, dataset_job_id=dataset_job_id, repo=repo, automatic=True)
    _run_push(job_id, dataset_dir, repo, token, _source_meta(job), False)
    return job_id


@router.post("/push")
def push(body: PushRequest, background_tasks: BackgroundTasks):
    token, repo = get_secret(_TOKEN), get_secret(_REPO)
    if not token or not repo:
        raise HTTPException(status_code=409, detail="Add your Hugging Face token in Settings first")
    dataset_dir, job = _export_dir(body.dataset_job_id)
    job_id = get_store().create(PUSH_KIND, progress=0.0, dataset_job_id=body.dataset_job_id, repo=repo)
    background_tasks.add_task(_run_push, job_id, dataset_dir, repo, token, _source_meta(job), body.allow_public)
    return {"job_id": job_id, "status": "queued", "repo": repo}


@router.get("/push/{job_id}")
def push_status(job_id: str):
    job = get_store().get(job_id)
    if not job or job.get("kind") not in (None, PUSH_KIND):
        raise HTTPException(status_code=404, detail="No such upload")
    return job
