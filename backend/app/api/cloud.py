"""Train on a RunPod GPU: the API key, start, watch, stop."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.jobs import get_store
from app.core.secrets import get_secret, set_secret

router = APIRouter(prefix="/api/cloud", tags=["cloud"])
_KEY = "runpod_key"
TRAIN_KIND = "train"
CLOUDS = ("COMMUNITY", "SECURE")


def runpod_client():
    from app.engine.runpod_client import RunPod

    key = get_secret(_KEY)
    return RunPod(key) if key else None


class KeyBody(BaseModel):
    key: str


class CloudTrain(BaseModel):
    push_job_id: str | None = None      # a finished "Save to Hugging Face"; none: the version's pushed.json
    dataset_job_id: str                 # the export it came from (classes, task, local data.yaml)
    model_size: str = "nano"
    epochs: int | None = Field(default=None, ge=1, le=1000)
    gpu: str = "rtx4090"
    cloud: str = "COMMUNITY"
    max_hours: float = Field(default=2.0, gt=0, le=24)


@router.get("/runpod")
def runpod_status():
    from app.engine.cloud_train import GPUS

    running = [j for j in get_store().list(TRAIN_KIND, limit=200)
               if j.get("where") == "runpod" and j.get("status") == "processing"]
    return {"configured": bool(get_secret(_KEY)), "gpus": list(GPUS), "running": len(running)}


@router.put("/runpod")
async def runpod_save(body: KeyBody):
    from app.engine.runpod_client import RunPod, RunPodError

    key = body.key.strip()
    if not key:
        set_secret(_KEY, None)
        return runpod_status()
    try:
        await run_in_threadpool(RunPod(key).check_key)
    except RunPodError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    set_secret(_KEY, key)
    return runpod_status()


@router.post("/train")
async def cloud_train(body: CloudTrain):
    from app.engine import cloud_train
    from app.engine.runpod_client import RunPodError
    from app.training.trainer import build_job_config

    rp, hf_token = runpod_client(), get_secret("hf_token")
    if rp is None:
        raise HTTPException(status_code=409, detail="Add your RunPod API key in Settings first")
    if not hf_token:
        raise HTTPException(status_code=409, detail="Connect Hugging Face in Settings first: the GPU reads the samples from there")
    if body.cloud not in CLOUDS or body.gpu not in cloud_train.GPUS:
        raise HTTPException(status_code=400, detail="Unknown GPU or cloud")
    store = get_store()
    export = store.get(body.dataset_job_id) or {}
    push = store.get(body.push_job_id) or {} if body.push_job_id else {}
    if not body.push_job_id and export.get("data_yaml"):
        from app.core.layout import read_json
        pushed = read_json(Path(export["data_yaml"]).parent / "pushed.json", None)
        if pushed:
            push = {"status": "completed", "result": pushed}
    if push.get("status") != "completed" or not (push.get("result") or {}).get("folder"):
        raise HTTPException(status_code=409, detail="Save the samples to Hugging Face first")
    data_yaml = export.get("data_yaml")
    if not data_yaml or not Path(data_yaml).exists():
        raise HTTPException(status_code=409, detail="The dataset export is gone: build the dataset again")
    n_train = sum(1 for _ in (Path(data_yaml).parent / "images" / "train").glob("*"))
    job_id = store.create(TRAIN_KIND, where="runpod", data_yaml=data_yaml, gpu=body.gpu, cloud=body.cloud,
                          max_hours=body.max_hours, dataset_job_id=body.dataset_job_id, push_job_id=body.push_job_id)
    from app.core import layout

    group, version, mdir = layout.new_model_dir(data_yaml)
    run_dir = mdir or settings.DATA_DIR / "training_runs" / job_id
    if group:
        store.update(job_id, model_group=group, model_version=version)
    cfg = build_job_config(data_yaml, str(run_dir), num_train_images=n_train, device="cuda:0",
                           model_size=body.model_size, free_vram_mb=24000,
                           overrides={"epochs": body.epochs, "workers": 8})
    store.update(job_id, config=cfg)
    try:
        await run_in_threadpool(lambda: cloud_train.start(
            job_id, runpod=rp, hf_token=hf_token, hf_repo=push["result"]["repo_id"],
            hf_folder=push["result"]["folder"], job_cfg=cfg, gpu=body.gpu, cloud=body.cloud,
            max_hours=body.max_hours, store=store, run_dir=run_dir))
    except (RunPodError, RuntimeError) as exc:
        store.update(job_id, status="error", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc))
    return store.get(job_id)


@router.post("/train/{job_id}/stop")
def cloud_stop(job_id: str):
    """Stop now: the watcher terminates the pod on its next pass; this also does it at once."""
    store = get_store()
    job = store.get(job_id, TRAIN_KIND)
    if not job or job.get("where") != "runpod":
        raise HTTPException(status_code=404, detail="No GPU run with this id")
    store.update(job_id, cancel_requested=True)
    rp = runpod_client()
    if rp and job.get("pod_id"):
        try:
            rp.terminate(job["pod_id"])
            store.update(job_id, status="canceled", pod_status="TERMINATED", message="Stopped")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Could not stop pod {job['pod_id']}: {exc}. Stop it in the RunPod console")
    return store.get(job_id)


def resume_watchers() -> int:
    from app.engine import cloud_train

    return cloud_train.resume_all(get_store(), runpod_client, lambda: get_secret("hf_token"))
