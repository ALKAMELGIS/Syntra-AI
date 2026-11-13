"""
Request-level safety helpers shared by every API router.

Three concerns live here:

1. **Path confinement** — any filesystem path that arrives in a request body
   (image paths, dataset folders, output folders) must resolve *inside* one
   of the allowed data roots. Without this, `POST /api/predict` with
   `image_path="C:/Windows/..."` or `../../etc/passwd` would read arbitrary
   files on the server, and `out_dir` fields would write anywhere.
   Allowed roots: `settings.DATA_DIR` (uploads, results, datasets, runs) plus
   any extra folders listed in `SYNTRA_EXTRA_DATA_ROOTS` — e.g. a local
   folder of labelled training data the operator wants to point at.

2. **Upload filename sanitising** — the client-supplied filename is reduced
   to a plain basename with a conservative character set before it touches
   the disk, so `../../x.tif` can never escape `UPLOAD_DIR`.

3. **Optional shared API key** — when `SYNTRA_API_KEY` is set, every
   `/api/*` request (except `/api/health`) must carry it in the `X-API-Key`
   header or `api_key` query parameter (the query form exists because
   `<img>`/`<a download>` URLs cannot set headers). This is a first gate for
   LAN/on-prem deployments, not a replacement for per-user authentication.
"""
from __future__ import annotations

import hmac
import os
import re
from pathlib import Path

from fastapi import HTTPException, Request

from app.core.config import settings

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._\- ]+")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def allowed_roots() -> list[Path]:
    roots = [settings.DATA_DIR]
    extra = os.getenv("SYNTRA_EXTRA_DATA_ROOTS", "")
    roots += [Path(p.strip()) for p in extra.split(os.pathsep) if p.strip()]
    return [r.resolve() for r in roots]


def confine_path(raw: str | os.PathLike, *, field: str = "path", must_exist: bool = False) -> Path:
    """Resolve `raw` and refuse it unless it lies inside an allowed data root.

    Raises HTTP 403 for paths outside the roots and HTTP 404 when
    `must_exist` is set and nothing is there. Returns the resolved path.
    """
    if raw is None or str(raw).strip() == "":
        raise HTTPException(status_code=400, detail=f"'{field}' must not be empty.")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = settings.DATA_DIR / candidate
    resolved = candidate.resolve()
    if not any(resolved == root or root in resolved.parents for root in allowed_roots()):
        raise HTTPException(
            status_code=403,
            detail=(
                f"'{field}' must point inside the Syntra data directory. "
                "To allow another folder, add it to SYNTRA_EXTRA_DATA_ROOTS in backend/.env."
            ),
        )
    if must_exist and not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {field}")
    return resolved


def safe_filename(name: str | None, *, default: str = "upload") -> str:
    """Reduce a client-supplied filename to a harmless basename."""
    base = Path((name or "").replace("\\", "/")).name
    base = _SAFE_CHARS.sub("_", base).strip(" .")
    return base[:120] or default


# --------------------------------------------------------------------------
# API key
# --------------------------------------------------------------------------
def configured_api_key() -> str | None:
    key = os.getenv("SYNTRA_API_KEY", "").strip()
    return key or None


def require_api_key(request: Request) -> None:
    """FastAPI dependency: enforce SYNTRA_API_KEY when it is configured."""
    expected = configured_api_key()
    if expected is None:
        return
    supplied = request.headers.get("x-api-key") or request.query_params.get("api_key") or ""
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


# --------------------------------------------------------------------------
# Privileged actions (pip install / repair / update)
# --------------------------------------------------------------------------
def is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in _LOOPBACK_HOSTS


def environment_actions_allowed(request: Request) -> bool:
    """pip actions run arbitrary package installs on the server.

    SYNTRA_ALLOW_ENV_ACTIONS=1 -> always allowed, =0 -> never allowed,
    unset -> allowed only for requests from this machine (the desktop /
    single-user setup the Library Status panel was built for).
    """
    flag = os.getenv("SYNTRA_ALLOW_ENV_ACTIONS", "").strip()
    if flag == "1":
        return True
    if flag == "0":
        return False
    return is_loopback(request)
