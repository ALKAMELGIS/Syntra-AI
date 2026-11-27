"""
Train on a RunPod GPU, watched from this computer.

The job is an ordinary "train" job (so progress, model registration and the
performance charts work as for a local run) with `where: "runpod"`.

  start -> pod created with the pod script (train_worker.py + pod_main.py)
  watch -> every 20 s: pod status and cost from RunPod, epoch lines from the
           Hub (runs/<run>/events.jsonl, uploaded by the pod once a minute)
  done  -> runs/<run>/done.json appears: best.pt, best.onnx, results.csv are
           downloaded next to a local run's files; the pod is terminated
  stop  -> on cancel, on an error, or at the time limit, the pod is terminated

Three guards against a pod left running: the pod's own `timeout`, then its
attempt to terminate itself, then this watcher with the user's key. A watch
that was running when Syntra stopped is resumed when the engine starts.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path

POD_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"     # from RunPod's create-pod example
POD_DISK_GB = 40
ULTRALYTICS = "ultralytics==8.3.49"                                  # same as backend/requirements.txt
POLL_S = 20
GPUS = {                                                             # id as RunPod names it
    "rtx4090": "NVIDIA GeForce RTX 4090",
    "rtx5090": "NVIDIA GeForce RTX 5090",
    "l40s": "NVIDIA L40S",
    "a40": "NVIDIA A40",
}

_watchers: dict[str, threading.Thread] = {}
_lock = threading.Lock()


def pod_script() -> str:
    """train_worker.py (without its CLI block) + pod_main.py, as one file."""
    here = Path(__file__).resolve().parent.parent / "training"
    worker = (here / "train_worker.py").read_text(encoding="utf-8")
    worker = worker.split('\nif __name__ == "__main__":')[0]
    return worker + "\n\n" + (here / "pod_main.py").read_text(encoding="utf-8")


def pod_args() -> str:
    """Shell the pod runs. Everything after the training line runs whatever happened."""
    return (
        "bash -c '"
        'echo "$SYNTRA_SCRIPT" | base64 -d > /syntra_train.py; '
        f'pip install -q "{ULTRALYTICS}" "huggingface_hub>=0.26" onnx onnxslim; '
        'timeout "$MAX_SECONDS" python /syntra_train.py; '
        'curl -s -X POST "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action" '
        '-H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" '
        "-d '\"'\"'{\"action\":\"terminate\"}'\"'\"' "
        '|| runpodctl remove pod "$RUNPOD_POD_ID"; '
        "sleep 600'"
    )


def start(job_id: str, *, runpod, hf_token: str, hf_repo: str, hf_folder: str, job_cfg: dict,
          gpu: str, cloud: str, max_hours: float, store, run_dir: Path, hf_api_factory=None) -> dict:
    gpu_id = GPUS.get(gpu, gpu)
    max_s = int(max_hours * 3600)
    env = {
        "HF_TOKEN": hf_token, "HF_REPO": hf_repo, "HF_FOLDER": hf_folder, "RUN_ID": job_id,
        "JOB_JSON": json.dumps({k: v for k, v in job_cfg.items() if k not in ("cancel_flag_path",)}),
        "MAX_SECONDS": str(max_s),
        "SYNTRA_SCRIPT": base64.b64encode(pod_script().encode()).decode(),
    }
    pod = runpod.create_pod(name=f"syntra-train-{job_id[:8]}", image=POD_IMAGE, gpu_id=gpu_id,
                            cloud=cloud, disk_gb=POD_DISK_GB, env=env, args=pod_args())
    pod_id = pod.get("id")
    if not pod_id:
        raise RuntimeError("RunPod accepted the request but has no free GPU of this type right now. Try another GPU or later")
    store.update(job_id, status="processing", pod_id=pod_id, pod_status=pod.get("status"),
                 cost_per_hour=pod.get("cost"), started_at=time.time(), deadline=time.time() + max_s + 900,
                 hf_repo=hf_repo, run_dir=str(run_dir), message="Starting the GPU")
    watch(job_id, runpod=runpod, hf_token=hf_token, store=store, hf_api_factory=hf_api_factory)
    return pod


def watch(job_id, *, runpod, hf_token, store, hf_api_factory=None, poll_s=POLL_S, background=True):
    def loop():
        try:
            _watch(job_id, runpod, hf_token, store, hf_api_factory, poll_s)
        finally:
            with _lock:
                _watchers.pop(job_id, None)

    with _lock:
        if job_id in _watchers:
            return
        if not background:
            _watchers[job_id] = threading.current_thread()
        else:
            t = threading.Thread(target=loop, name=f"syntra-gpu-{job_id[:8]}", daemon=True)
            _watchers[job_id] = t
            t.start()
            return
    loop()


def _hf(token, factory):
    if factory:
        return factory(token)
    from huggingface_hub import HfApi
    return HfApi(token=token)


def _read(api, repo, path) -> bytes | None:
    try:
        if not api.file_exists(repo, path, repo_type="dataset"):
            return None
        local = api.hf_hub_download(repo, path, repo_type="dataset", force_download=True)
        return Path(local).read_bytes()
    except Exception:  # noqa: BLE001 - not there yet / transient
        return None


def _stop(runpod, pod_id, store, job_id, **fields):
    try:
        runpod.terminate(pod_id)
        fields.setdefault("pod_status", "TERMINATED")
    except Exception as exc:  # noqa: BLE001 - say it plainly: the user must stop it by hand
        fields["warning"] = f"Could not stop the GPU ({exc}). Stop pod {pod_id} in the RunPod console now"
    job = store.get(job_id) or {}
    if job.get("started_at") and job.get("cost_per_hour"):
        fields["cost_usd"] = round((time.time() - job["started_at"]) / 3600 * float(job["cost_per_hour"]), 3)
    store.update(job_id, **fields)


def _watch(job_id, runpod, hf_token, store, factory, poll_s):
    api = _hf(hf_token, factory)
    seen = 0
    while True:
        job = store.get(job_id) or {}
        pod_id, repo, run_dir = job.get("pod_id"), job.get("hf_repo"), Path(job.get("run_dir") or ".")
        if not pod_id or job.get("status") not in ("processing",):
            return
        if job.get("cancel_requested"):
            _stop(runpod, pod_id, store, job_id, status="canceled", message="Stopped")
            return
        if time.time() > float(job.get("deadline") or 0):
            _stop(runpod, pod_id, store, job_id, status="error", error="Stopped at the time limit you set")
            return
        try:
            pod = runpod.get_pod(pod_id)
            fields = {"pod_status": pod.get("status"), "cost_per_hour": pod.get("cost", job.get("cost_per_hour"))}
            if job.get("started_at") and fields["cost_per_hour"]:
                fields["cost_usd"] = round((time.time() - job["started_at"]) / 3600 * float(fields["cost_per_hour"]), 3)
            store.update(job_id, **fields)
            if pod.get("status") in ("TERMINATED", "ERROR"):
                pass                                  # done.json decides; a pod gone without one is an error below
        except Exception as exc:  # noqa: BLE001 - keep watching; the time limit still applies
            store.update(job_id, message=f"RunPod not reachable: {exc}")
            pod = {}

        raw = _read(api, repo, f"runs/{job_id}/events.jsonl")
        if raw:
            lines = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
            for ev in lines[seen:]:
                if ev.get("type") == "epoch":
                    store.update(job_id, progress=ev)
                    store.append(job_id, "history", ev, max_len=500)
                elif ev.get("type") == "status":
                    store.update(job_id, message=ev.get("message"))
            seen = len(lines)

        done = _read(api, repo, f"runs/{job_id}/done.json")
        if done:
            d = json.loads(done)
            if d.get("status") == "done":
                weights = run_dir / "run" / "weights"
                weights.mkdir(parents=True, exist_ok=True)
                for name, dest in (("best.pt", weights / "best.pt"), ("best.onnx", weights / "best.onnx"),
                                   ("results.csv", run_dir / "run" / "results.csv"),
                                   ("test_metrics.json", run_dir / "run" / "test_metrics.json")):
                    data = _read(api, repo, f"runs/{job_id}/{name}")
                    if data:
                        dest.write_bytes(data)
                ok = (weights / "best.pt").exists()
                _stop(runpod, pod_id, store, job_id, status="done" if ok else "error",
                      result={"type": "done", "best": str(weights / "best.pt") if ok else None,
                              "onnx": str(weights / "best.onnx") if (weights / "best.onnx").exists() else None,
                              "results_csv": str(run_dir / "run" / "results.csv"), "save_dir": str(run_dir / "run"),
                              "elapsed_s": d.get("elapsed_s")},
                      **({} if ok else {"error": "The GPU finished but its weights could not be downloaded"}))
            else:
                _stop(runpod, pod_id, store, job_id, status="error", error=d.get("message") or "Training failed on the GPU")
            return
        if pod and pod.get("status") in ("TERMINATED",) and not done:
            store.update(job_id, status="error", error="The GPU stopped before training finished")
            return
        time.sleep(poll_s)


def resume_all(store, runpod_factory, hf_token_getter) -> int:
    """Called when the engine starts: watch again any GPU run that was in progress."""
    n = 0
    for job in store.list("train") if hasattr(store, "list") else []:
        if job.get("where") == "runpod" and job.get("status") == "processing" and job.get("pod_id"):
            key, tok = runpod_factory(), hf_token_getter()
            if key and tok:
                watch(job["job_id"], runpod=key, hf_token=tok, store=store)
                n += 1
    return n
