"""
Training subprocess: runs Ultralytics YOLO training for one job and streams
one JSON line per epoch to stdout. Launched as its own OS process by
app.training.trainer.TrainingJob (never imported into the FastAPI process
itself), so a training run — which can take hours and fully pins every CPU
core — can never freeze the API server or a concurrent prediction request,
the same reasoning that motivated the earlier upload fix via run_in_threadpool.

Ported methodology from the GeoSyntra AI QGIS plugin's workers/train_worker.py
(same idea: subprocess isolation + a one-JSON-object-per-epoch stdout
protocol, which the user confirmed "works great on CPU"). The QGIS-specific
process launching (QProcess) is replaced by a plain subprocess, and progress
goes to stdout instead of a Qt signal — everything else about *how* training
progresses is the same.

Usage: python -m app.training.train_worker <job.json>
Exits 0 on success/cancel, 1 on error (also reported as a JSON "error" line).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _cancel_requested(flag_path: str | None) -> bool:
    return bool(flag_path) and Path(flag_path).exists()


def run(job: dict) -> int:
    t0 = time.time()
    try:
        from ultralytics import YOLO  # lazy: only the subprocess needs this installed
    except ImportError as exc:
        _emit({"type": "error", "message": f"ultralytics is not installed: {exc}"})
        return 1

    cancel_flag = job.get("cancel_flag_path")
    canceled = {"flag": False}

    def on_epoch_end(trainer):
        m = trainer.metrics or {}
        _emit({
            "type": "epoch",
            "epoch": int(trainer.epoch) + 1,
            "epochs": int(trainer.epochs),
            "box_loss": _f(getattr(trainer, "loss_items", None), 0),
            "cls_loss": _f(getattr(trainer, "loss_items", None), 1),
            "dfl_loss": _f(getattr(trainer, "loss_items", None), 2),
            "precision": _f(m.get("metrics/precision(B)")),
            "recall": _f(m.get("metrics/recall(B)")),
            "map50": _f(m.get("metrics/mAP50(B)")),
            "map50_95": _f(m.get("metrics/mAP50-95(B)")),
            "elapsed_s": round(time.time() - t0, 1),
        })
        if _cancel_requested(cancel_flag):
            canceled["flag"] = True
            # graceful stop: let the current epoch finish, then exit the loop
            trainer.epochs = int(trainer.epoch) + 1

    try:
        model = YOLO(job["model"])
        model.add_callback("on_train_epoch_end", on_epoch_end)
        results = model.train(
            data=job["data_yaml"],
            epochs=int(job["epochs"]),
            imgsz=int(job["imgsz"]),
            batch=int(job["batch"]),
            workers=int(job.get("workers", 4)),
            device=job.get("device", "cpu"),
            optimizer=job.get("optimizer", "auto"),
            lr0=float(job.get("lr0", 0.01)),
            patience=int(job.get("patience", 50)),
            project=job.get("project"),
            name=job.get("name", "run"),
            exist_ok=True,
            verbose=False,
        )
        save_dir = Path(getattr(results, "save_dir", job.get("project", ".")))
        best = save_dir / "weights" / "best.pt"
        _emit({
            "type": "canceled" if canceled["flag"] else "done",
            "best": str(best) if best.exists() else None,
            "results_csv": str(save_dir / "results.csv"),
            "save_dir": str(save_dir),
            "elapsed_s": round(time.time() - t0, 1),
        })
        return 0
    except Exception as exc:  # noqa: BLE001 - must always report, never crash silently
        _emit({"type": "error", "message": str(exc), "elapsed_s": round(time.time() - t0, 1)})
        return 1


def _f(loss_items, idx: int | None = None):
    try:
        if loss_items is None:
            return None
        if idx is None:
            return float(loss_items)
        return float(loss_items[idx])
    except Exception:
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        _emit({"type": "error", "message": "usage: train_worker.py <job.json>"})
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        job_cfg = json.load(fh)
    sys.exit(run(job_cfg))
