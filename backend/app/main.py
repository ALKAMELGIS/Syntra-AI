"""
Syntra AI — Detection Engine API.

Phase 1 priority per the product spec: a working detection engine first,
dashboard second. This file wires the FastAPI app; the engine itself lives
in app/engine and app/models and is fully usable from plain Python/tests
without any HTTP layer at all.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import annotations, auth, basemaps, cloud, samples, segment, library, remote_sensing, billing, admin, oauth, benchmark, compute, datasets, environment, images, jobs, models, predict, results, training, usage, vector
from app.core.config import settings
from app.core.jobs import get_store
from app.core.auth import COOKIE as COOKIE_NAME, require_access


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    try:                                     # once: uploads/ -> images/<key>/, label files -> classes/
        from app.core.migrate import migrate_layout
        migrate_layout()
    except Exception:  # noqa: BLE001 - never block start-up on this; files stay where they were
        logging.getLogger(__name__).exception("Data folder migration failed")
    # Jobs a previous process left running can't still be running: their
    # worker threads died with it. Say so instead of showing them as live.
    get_store().mark_interrupted()
    if os.getenv("SYNTRA_AUTO_INSTALL_SAM") == "1":    # hosted: the click tool's model, fetched once per start
        import threading

        def _sam():
            try:
                from app.engine import click_segment
                if not click_segment.status()["ready"]:
                    click_segment.install()
            except Exception:  # noqa: BLE001 - the Install button stays as the fallback
                logging.getLogger(__name__).exception("Installing SAM at start-up failed")
        threading.Thread(target=_sam, name="sam-install", daemon=True).start()
    try:                                     # hosted, reset disk: the owner account from its secrets
        from app.core.auth import ensure_owner_from_env
        ensure_owner_from_env()
    except Exception:  # noqa: BLE001 - never block start-up on this
        logging.getLogger(__name__).exception("Owner account from the environment failed")
    try:                                     # GPU runs on RunPod keep going while Syntra is closed
        from app.api.cloud import resume_watchers
        resume_watchers()
    except Exception:  # noqa: BLE001 - never block start-up on this
        pass
    yield


app = FastAPI(title="Syntra AI Detection Engine", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)

@app.middleware("http")
async def same_origin_writes(request, call_next):
    """The session cookie is SameSite=None over https (it must work inside the Space's frame),
    so a form on another site could post with it. Browsers send Origin on such posts: a write
    that carries the cookie and comes from another origin is refused."""
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.cookies.get(COOKIE_NAME):
        origin = request.headers.get("origin")
        if origin and origin != "null":
            from urllib.parse import urlsplit
            host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
            if urlsplit(origin).netloc != host and origin not in settings.CORS_ORIGINS:
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
    return await call_next(request)


_protected = [Depends(require_access)]

app.include_router(auth.router)
app.include_router(oauth.router)              # open: how a session starts
app.include_router(billing.webhook_router)     # open: Stripe calls it, signature checked
app.include_router(billing.router, dependencies=_protected)
app.include_router(admin.router, dependencies=_protected)

app.include_router(models.router, dependencies=_protected)
app.include_router(basemaps.router, dependencies=_protected)
app.include_router(images.router, dependencies=_protected)
app.include_router(vector.router, dependencies=_protected)
app.include_router(predict.router, dependencies=_protected)
app.include_router(results.router, dependencies=_protected)
app.include_router(compute.router, dependencies=_protected)
app.include_router(datasets.router, dependencies=_protected)
app.include_router(benchmark.router, dependencies=_protected)
app.include_router(environment.router, dependencies=_protected)
app.include_router(usage.router, dependencies=_protected)
app.include_router(training.router, dependencies=_protected)
app.include_router(annotations.router, dependencies=_protected)
app.include_router(jobs.router, dependencies=_protected)
app.include_router(samples.router, dependencies=_protected)
app.include_router(cloud.router, dependencies=_protected)
app.include_router(segment.router, dependencies=_protected)
app.include_router(library.router, dependencies=_protected)
app.include_router(remote_sensing.router, dependencies=_protected)


@app.exception_handler(Exception)
async def explain_errors(request, exc):
    """An unexpected error still says what went wrong, as JSON the interface can
    show, instead of a bare "Internal Server Error". A full disk is named, since
    large uploads are the usual way to hit it."""
    import errno
    import logging

    from fastapi.responses import JSONResponse

    logging.getLogger("syntra").exception("Unhandled error on %s", request.url.path)
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return JSONResponse(status_code=507, content={"detail": (
            "The disk is full. Free space on the drive that holds Syntra's data and Windows' temp folder, "
            "or open the image from the This PC tab, which needs no copy.")})
    return JSONResponse(status_code=500, content={"detail": f"The engine hit an error: {type(exc).__name__}: {exc}"})


@app.get("/api/health")
def health():
    # api: bumped when the interface needs routes an older running engine lacks
    return {"status": "ok", "service": "syntra-ai-detection-engine", "api": API_LEVEL}


API_LEVEL = 5          # 2: /api/rs · 3: accounts, credits, admin · 4: box prompts · 5: UTF-8 training on Windows


# One process for both: when SYNTRA_STATIC_DIR names the built interface (frontend/dist),
# the engine serves it too (hosted installs: a Hugging Face Space, a single server).
def _serve_interface() -> None:
    import os
    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    root = os.getenv("SYNTRA_STATIC_DIR")
    if not root or not (Path(root) / "index.html").is_file():
        return
    root = Path(root).resolve()
    if (root / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def interface(path: str):
        if path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not found")
        f = (root / path).resolve()
        if path and path != "index.html" and root in f.parents and f.is_file():
            return FileResponse(f)
        # the app's own routes; never kept by the browser, so a deploy shows at once (assets are hashed)
        return FileResponse(root / "index.html", headers={"Cache-Control": "no-cache"})


_serve_interface()

