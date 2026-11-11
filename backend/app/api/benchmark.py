"""Benchmark endpoint — computes real metrics from submitted predictions + ground truth."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel

from app.engine.benchmark import run_benchmark

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


class DetectionRecord(BaseModel):
    image_id: str
    box: tuple[float, float, float, float]
    class_id: int
    score: float = 1.0  # ignored for ground truth


class BenchmarkRequest(BaseModel):
    model_id: str
    predictions: list[DetectionRecord]
    ground_truth: list[DetectionRecord]
    num_images: int
    inference_times_ms: list[float]
    gpu_memory_mb: float | None = None


@router.post("")
def benchmark(req: BenchmarkRequest):
    report = run_benchmark(
        model_id=req.model_id,
        predictions=[p.model_dump() for p in req.predictions],
        ground_truth=[g.model_dump() for g in req.ground_truth],
        num_images=req.num_images,
        inference_times_ms=req.inference_times_ms,
        gpu_memory_mb=req.gpu_memory_mb,
    )
    return asdict(report)
