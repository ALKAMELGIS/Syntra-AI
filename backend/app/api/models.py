"""GET endpoints for the Detection Models dashboard (Phase 4)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.models.custom_registry import delete_custom_model, get_custom_model
from app.models.registry import get_metadata, list_catalog
from app.models.stats import get_stats

router = APIRouter(prefix="/api/models", tags=["models"])


def _serialize(meta) -> dict:
    stats = get_stats(settings.DATA_DIR, meta.model_id)
    return {
        "model_id": meta.model_id,
        "name": meta.name,
        "model_type": meta.model_type,
        "version": meta.version,
        "supported_gsd_m": [meta.supported_gsd.min_m, meta.supported_gsd.max_m],
        "supported_image_sources": meta.supported_image_sources,
        "supported_formats": meta.supported_formats,
        "status": meta.status.value,
        "hardware_compatibility": meta.hardware_compatibility,
        "default_confidence_threshold": meta.default_confidence_threshold,
        "notes": meta.notes,
        "processing_statistics": stats.to_dict(),
    }


@router.get("")
def list_models():
    return [_serialize(m) for m in list_catalog()]


@router.get("/{model_id}")
def get_model(model_id: str):
    try:
        return _serialize(get_metadata(model_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/{model_id}")
def delete_model(model_id: str):
    """Only a model registered from a training run (app.models.
    custom_registry) can be removed here — the static catalog above is a
    fixed product category list, not user data."""
    if get_custom_model(settings.DATA_DIR, model_id) is None:
        raise HTTPException(
            status_code=404 if model_id not in {m.model_id for m in list_catalog()} else 409,
            detail=f"'{model_id}' is not a trained model you registered — nothing to delete here.",
        )
    delete_custom_model(settings.DATA_DIR, model_id)
    return {"deleted": model_id}


@router.get("/{model_id}/gsd-check")
def gsd_check(model_id: str, gsd: float):
    try:
        meta = get_metadata(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    from app.models.base import DetectionModel

    # gsd_compatibility_report is an instance method; build a throwaway shim
    class _Shim(DetectionModel):
        def load_model(self, device): ...
        def validate_model(self): return True, "shim"
        def predict(self, *a, **k): return []
        def export_results(self, *a, **k): return {}

    shim = _Shim()
    shim.metadata = meta
    return shim.gsd_compatibility_report(gsd)
