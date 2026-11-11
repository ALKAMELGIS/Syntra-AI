"""
Syntra AI — Detection Engine API.

Phase 1 priority per the product spec: a working detection engine first,
dashboard second. This file wires the FastAPI app; the engine itself lives
in app/engine and app/models and is fully usable from plain Python/tests
without any HTTP layer at all.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import benchmark, compute, datasets, environment, images, models, predict, results
from app.core.config import settings

app = FastAPI(title="Syntra AI Detection Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(images.router)
app.include_router(predict.router)
app.include_router(results.router)
app.include_router(compute.router)
app.include_router(datasets.router)
app.include_router(benchmark.router)
app.include_router(environment.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "syntra-ai-detection-engine"}


@app.on_event("startup")
def startup():
    settings.ensure_dirs()
