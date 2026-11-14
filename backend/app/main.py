"""
Syntra AI — Detection Engine API.

Phase 1 priority per the product spec: a working detection engine first,
dashboard second. This file wires the FastAPI app; the engine itself lives
in app/engine and app/models and is fully usable from plain Python/tests
without any HTTP layer at all.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import annotations, auth, benchmark, compute, datasets, environment, images, jobs, models, predict, results, training, usage
from app.core.config import settings
from app.core.jobs import get_store
from app.core.auth import require_access


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    # Jobs a previous process left running can't still be running: their
    # worker threads died with it. Say so instead of showing them as live.
    get_store().mark_interrupted()
    yield


app = FastAPI(title="Syntra AI Detection Engine", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_protected = [Depends(require_access)]

app.include_router(auth.router)

app.include_router(models.router, dependencies=_protected)
app.include_router(images.router, dependencies=_protected)
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


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "syntra-ai-detection-engine"}

