"""
Model factory — maps a catalog model_id to a concrete, loadable DetectionModel
instance. This is the ONE place that decides which runtime backs each model
id, so swapping an architecture or wiring in freshly trained weights never
touches the pipeline or API code.
"""
from __future__ import annotations

from app.models.base import DetectionModel, ModelStatus
from app.models.registry import get_metadata


def build_model(model_id: str) -> DetectionModel:
    metadata = get_metadata(model_id)

    if model_id in ("car_detection", "ship_detection", "aircraft_detection"):
        from app.models.generic_detector import GenericYoloDetector
        return GenericYoloDetector(metadata)

    if model_id == "field_boundary_segmentation":
        from app.models.segmentation import FieldBoundarySegmenter
        return FieldBoundarySegmenter(metadata)

    if metadata.status == ModelStatus.NOT_PRODUCTION_READY:
        # No runtime exists yet — these are NOT_PRODUCTION_READY with no
        # weights to load at all. Raise a clear, typed error rather than
        # silently returning something that looks runnable.
        from app.models.segmentation import NotProductionReadyError
        raise NotProductionReadyError(
            f"{metadata.name}: no model runtime/weights exist yet. {metadata.notes}"
        )

    raise KeyError(f"No factory entry for model_id '{model_id}'.")
