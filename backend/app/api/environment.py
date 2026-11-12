"""
Environment / dependency management API — backs the Detection Models tab's
"Library Status" + "Installation Options" panel.

GET  /api/environment                 -> real dependency/CUDA/PROJ report
POST /api/environment/actions         -> kick off a real pip operation (background job)
GET  /api/environment/actions/{job_id} -> poll that job's status + live log tail

Every action here runs an actual `pip` subprocess against *this* interpreter
(sys.executable -m pip) — there is no simulated progress bar and no faked
"installed" result. If pip fails, the job reports returncode != 0 and the
real stderr tail.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from enum import Enum
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.diagnostics.environment import build_report, resolve_install_targets

router = APIRouter(prefix="/api/environment", tags=["environment"])

REQUIREMENTS_PATH = Path(__file__).resolve().parent.parent.parent / "requirements.txt"

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_LOG_TAIL_LINES = 400

# Published PyTorch wheel indexes for CPU-only vs CUDA builds — same choice
# the "Build" dropdown in the reference UI exposes. "auto" installs whatever
# PyPI's default torch wheel resolves to (no extra index).
_TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": "https://download.pytorch.org/whl/cu121",
}


class ActionKind(str, Enum):
    INSTALL = "install"
    REPAIR = "repair"
    UPDATE = "update"


class BuildOption(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class ActionRequest(BaseModel):
    action: ActionKind
    build: BuildOption = BuildOption.AUTO
    # Optional list of module names (e.g. "cv2", "ultralytics") to install/repair/
    # update individually instead of the whole requirements.txt. Each name is
    # resolved to its *pinned* line in requirements.txt, so a per-library install
    # still gets the exact version the project is tested against — never an
    # unpinned "latest".
    targets: list[str] | None = None


def _pip_command(action: ActionKind, build: BuildOption, targets: list[str] | None = None) -> list[str]:
    if targets:
        cmd = [sys.executable, "-m", "pip", "install", *resolve_install_targets(targets)]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)]
    if action == ActionKind.UPDATE:
        cmd.append("--upgrade")
    elif action == ActionKind.REPAIR:
        cmd += ["--force-reinstall", "--no-cache-dir"]
    if build in _TORCH_INDEX:
        cmd += ["--extra-index-url", _TORCH_INDEX[build]]
    return cmd


def _run_job(job_id: str, cmd: list[str]):
    with _LOCK:
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["command"] = " ".join(cmd)
    log_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            log_lines.append(line.rstrip("\n"))
            with _LOCK:
                _JOBS[job_id]["log_tail"] = log_lines[-_LOG_TAIL_LINES:]
        returncode = proc.wait()
        with _LOCK:
            _JOBS[job_id].update(
                status="completed" if returncode == 0 else "error",
                returncode=returncode,
                log_tail=log_lines[-_LOG_TAIL_LINES:],
            )
    except FileNotFoundError as exc:
        with _LOCK:
            _JOBS[job_id].update(status="error", returncode=None, error=str(exc), log_tail=log_lines)
    except Exception as exc:  # pip itself crashing, permissions, etc.
        with _LOCK:
            _JOBS[job_id].update(status="error", returncode=None, error=str(exc), log_tail=log_lines)


@router.get("")
def get_environment():
    return build_report()


@router.post("/actions")
def start_action(req: ActionRequest, background_tasks: BackgroundTasks):
    if not REQUIREMENTS_PATH.exists():
        raise HTTPException(status_code=500, detail=f"requirements.txt not found at {REQUIREMENTS_PATH}")

    try:
        cmd = _pip_command(req.action, req.build, req.targets)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    job_id = str(uuid.uuid4())
    with _LOCK:
        _JOBS[job_id] = {
            "status": "queued", "action": req.action.value, "build": req.build.value,
            "targets": req.targets or [], "log_tail": [],
        }
    background_tasks.add_task(_run_job, job_id, cmd)
    return {"job_id": job_id, "status": "queued", "command": " ".join(cmd)}


@router.get("/actions/{job_id}")
def action_status(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job
