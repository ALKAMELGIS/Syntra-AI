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
    from app.core.config import settings
    from app.models.custom_registry import get_custom_model

    custom_entry = get_custom_model(settings.DATA_DIR, model_id)
    if custom_entry is not None:
        from app.models.custom_detector import CustomYoloDetector
        return CustomYoloDetector(get_metadata(model_id), weights_path=custom_entry.weights_path)

    metadata = get_metadata(model_id)

    from app.models.packs import is_installed, pack_for_model, pack_path
    pack = pack_for_model(model_id)
    if pack is not None:
        if not is_installed(pack):
            from app.models.segmentation import NotProductionReadyError
            raise NotProductionReadyError(f"{metadata.name}: model pack '{pack.pack_id}' is not installed.")
        if pack.runtime == "geodeep_detector":
            from app.models.geodeep_onnx import GeoDeepDetector
            return GeoDeepDetector(metadata, str(pack_path(pack)))
        if pack.runtime == "yolo_seg":
            from app.models.drone_seg_detector import DroneSegDetector
            return DroneSegDetector(metadata, str(pack_path(pack)), training_gsd_m=pack.training_gsd_m or 0.04)
        raise KeyError(f"{model_id} is a segmentation pack; it runs through app.engine.segmentation_run.")

    from app.models.aerial_obb_detector import DOTA_CLASS_MAP
    if model_id in DOTA_CLASS_MAP:
        from app.models.aerial_obb_detector import AerialObbDetector
        return AerialObbDetector(metadata)

    if model_id == "vegetation_index":
        raise KeyError("vegetation_index is a spectral-index analysis run by app.engine.vegetation, not a tile detector.")

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
