"""
Runs inside a RunPod GPU pod (appended to train_worker.py and sent as one file).

1. If this run already finished (runs/<run>/done.json on the Hub), stop at
   once: a pod whose container restarts must not train and bill twice.
2. Download the dataset folder from the private Hugging Face repo.
3. Train with the same train_worker.run() Syntra uses on a local machine;
   every epoch line also goes to events.jsonl, uploaded once a minute.
4. Export best.pt to ONNX, upload best.pt, best.onnx, results.csv,
   events.jsonl, then done.json (last, so its presence means "complete").
The shell around this script then terminates the pod.
"""
import json as _json
import os as _os
import pathlib as _pl
import threading as _th
import time as _time


def _pod_main() -> int:
    from huggingface_hub import HfApi, snapshot_download

    tok, repo, folder, run_id = (_os.environ[k] for k in ("HF_TOKEN", "HF_REPO", "HF_FOLDER", "RUN_ID"))
    job = _json.loads(_os.environ["JOB_JSON"])
    api = HfApi(token=tok)
    prefix = f"runs/{run_id}"
    if api.file_exists(repo, f"{prefix}/done.json", repo_type="dataset"):
        print("This run already finished; nothing to do.")
        return 0

    root = _pl.Path("/workspace/syntra")
    root.mkdir(parents=True, exist_ok=True)
    events = root / "events.jsonl"
    lock = _th.Lock()

    def emit(obj):
        line = _json.dumps(obj)
        print(line, flush=True)
        with lock:
            with events.open("a") as fh:
                fh.write(line + "\n")

    globals()["_emit"] = emit            # train_worker.run() reports through _emit

    def put(path, name):
        api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{prefix}/{name}", repo_id=repo,
                        repo_type="dataset", commit_message=f"Syntra GPU run {run_id}: {name}")

    stop = _th.Event()

    def pusher():
        while not stop.wait(60):
            try:
                with lock:
                    data = events.read_bytes() if events.exists() else b""
                if data:
                    api.upload_file(path_or_fileobj=data, path_in_repo=f"{prefix}/events.jsonl", repo_id=repo,
                                    repo_type="dataset", commit_message=f"Syntra GPU run {run_id}: progress")
            except Exception as exc:  # noqa: BLE001 - progress is best effort
                print("progress upload failed:", exc, flush=True)

    done = {"run_id": run_id, "status": "error"}
    try:
        emit({"type": "status", "message": "Downloading samples"})
        snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=[f"{folder}/**"],
                          local_dir=str(root / "data"), token=tok)
        ds = root / "data" / folder
        yaml_path = ds / "data.yaml"
        lines = [ln for ln in yaml_path.read_text().splitlines() if not ln.startswith("path:")]
        yaml_path.write_text(f"path: {ds}\n" + "\n".join(lines) + "\n")
        _th.Thread(target=pusher, daemon=True).start()
        emit({"type": "status", "message": "Training"})
        job.update(data_yaml=str(yaml_path), project=str(root / "run"), name="run", device=0)

        final = {}
        inner = emit

        def capture(obj):
            final.update(obj) if obj.get("type") in ("done", "canceled", "error") else None
            inner(obj)

        globals()["_emit"] = capture
        run(job)                          # noqa: F821 - defined by train_worker.py above
        globals()["_emit"] = emit
        best = final.get("best")
        if final.get("type") == "done" and best and _pl.Path(best).exists():
            emit({"type": "status", "message": "Exporting ONNX"})
            from ultralytics import YOLO
            onnx = YOLO(best).export(format="onnx", imgsz=int(job["imgsz"]))
            run_dir = _pl.Path(best).parent.parent
            put(best, "best.pt")
            put(onnx, "best.onnx")
            if (run_dir / "results.csv").exists():
                put(run_dir / "results.csv", "results.csv")
            done = {"run_id": run_id, "status": "done", "elapsed_s": final.get("elapsed_s")}
        else:
            done = {"run_id": run_id, "status": "error", "message": final.get("message") or "Training gave no weights"}
    except Exception as exc:  # noqa: BLE001 - always leave a done.json so Syntra can stop the pod
        done = {"run_id": run_id, "status": "error", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        stop.set()
        try:
            if events.exists():
                put(events, "events.jsonl")
            api.upload_file(path_or_fileobj=_json.dumps(done).encode(), path_in_repo=f"{prefix}/done.json",
                            repo_id=repo, repo_type="dataset", commit_message=f"Syntra GPU run {run_id}: {done['status']}")
        except Exception as exc:  # noqa: BLE001
            print("final upload failed:", exc, flush=True)
    return 0 if done["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(_pod_main())
