"""
Launches and supervises one training run as its own subprocess
(app.training.train_worker), parsing its one-JSON-object-per-line stdout
protocol and handing each event to a caller-supplied callback — the same
job-store/callback shape app/api/predict.py already uses for detection runs.

Running training out-of-process (rather than importing ultralytics/torch
into the FastAPI worker) is the core thing ported from the QGIS plugin: a
multi-hour CPU training run must never be able to freeze the API server,
the same reasoning behind the run_in_threadpool fix for large image uploads.

`subprocess`/`psutil` are only touched inside functions, not at import time,
so this module stays importable (and its config-building / event-parsing
logic stays testable) without those being a hard requirement.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

from .autoparams import suggest_training_params

# small built-in base-checkpoint catalogue, matched to the reference tool's
# "works great on CPU" smallest models by default
BASE_MODELS = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
}


class TrainingError(RuntimeError):
    pass


def build_job_config(
    data_yaml: str,
    out_dir: str,
    *,
    num_train_images: int,
    device: str = "cpu",
    model_size: str = "nano",
    tile_size: int = 640,
    cpu_threads: int | None = None,
    free_vram_mb: int | None = None,
    available_ram_mb: int | None = None,
    overrides: dict | None = None,
    cancel_flag_path: str | None = None,
) -> dict:
    """Auto-fills epochs/batch/workers/imgsz the way the reference Train step
    does, then applies any explicit overrides from the caller/user."""
    suggested = suggest_training_params(
        device=device, imgsz=tile_size, num_train_images=num_train_images,
        cpu_threads=cpu_threads, free_vram_mb=free_vram_mb, available_ram_mb=available_ram_mb,
    )
    cfg = {
        "data_yaml": data_yaml,
        "model": BASE_MODELS.get(model_size, model_size),
        "device": device,
        "project": out_dir,
        "name": "run",
        "optimizer": "auto",
        "lr0": 0.01,
        "patience": 50,
        "cancel_flag_path": cancel_flag_path,
        **suggested,
    }
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def request_cancel(cancel_flag_path: str) -> None:
    Path(cancel_flag_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cancel_flag_path).touch()


def run_training(
    job_config: dict,
    on_event: Callable[[dict], None],
    *,
    cmd: list[str] | None = None,
    cwd: str | None = None,
) -> dict:
    """Blocking: launches the worker subprocess, streams its stdout as
    parsed JSON events to `on_event`, and returns the final event
    ("done" / "canceled" / "error"). Intended to be called from inside a
    background task/thread, same as app.api.predict's `_run_job`.

    `cmd` lets tests substitute a fake worker script in place of
    `python -m app.training.train_worker`, so the subprocess-supervision and
    line-parsing logic here is testable without ultralytics/torch installed.
    """
    import subprocess

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(job_config, fh)
        job_path = fh.name

    if cmd is None:
        cmd = [sys.executable, "-m", "app.training.train_worker"]
    full_cmd = [*cmd, job_path]

    final: dict = {"type": "error", "message": "worker produced no output"}
    try:
        proc = subprocess.Popen(
            full_cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            final = event
            on_event(event)
        proc.wait()
        stderr = proc.stderr.read() if proc.stderr else ""
        if proc.returncode != 0 and final.get("type") not in ("error", "canceled"):
            final = {"type": "error", "message": stderr.strip() or f"worker exited with code {proc.returncode}"}
            on_event(final)
    finally:
        try:
            os.unlink(job_path)
        except OSError:
            pass
    return final
