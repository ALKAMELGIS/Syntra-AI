"""
Usage summary — backs the top-bar usage counter.

Per the product decision: no real payment/billing integration exists (or is
requested) yet, so this intentionally reports a real, derived usage count
(total prediction runs across every model) rather than a fabricated dollar
balance. Pulled straight from app.models.stats, the same persisted
per-model run history the Detection Models tab's "Processing statistics"
already displays — nothing new is tracked, this just aggregates it.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.models.stats import get_all_stats

router = APIRouter(prefix="/api/usage", tags=["usage"])


@router.get("")
def usage_summary():
    all_stats = get_all_stats(settings.DATA_DIR)
    total_runs = sum(s.run_count for s in all_stats.values())
    total_errors = sum(s.error_count for s in all_stats.values())
    total_tiles = sum(s.total_tiles_processed for s in all_stats.values())
    total_detections = sum(s.total_detections for s in all_stats.values())
    total_duration = sum(s.total_duration_seconds for s in all_stats.values())
    return {
        "total_runs": total_runs,
        "total_errors": total_errors,
        "total_tiles_processed": total_tiles,
        "total_detections": total_detections,
        "total_duration_seconds": round(total_duration, 1),
        "models_with_runs": len([s for s in all_stats.values() if s.run_count > 0]),
    }
