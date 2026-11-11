"""
Field-boundary segmentation model (U-Net / DeepLab) — Phase 1 scaffold.

No trained weights exist yet, so this class deliberately raises on
load_model()/predict() rather than returning fabricated polygons. It still
implements the full DetectionModel interface so the orchestration pipeline,
the model-management dashboard, and the API can treat it uniformly with
every other model while it reports NOT_PRODUCTION_READY.
"""
from __future__ import annotations

import numpy as np

from app.engine.geo import GeoTransform
from app.models.base import DetectionModel, ModelMetadata, ModelStatus, RawDetection


class NotProductionReadyError(RuntimeError):
    """Raised when an unweighted model is asked to actually run inference."""


class FieldBoundarySegmenter(DetectionModel):
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self._model = None

    def load_model(self, device: str) -> None:
        if self.metadata.status == ModelStatus.NOT_PRODUCTION_READY:
            raise NotProductionReadyError(
                f"{self.metadata.name}: no trained segmentation weights are available yet. "
                "Train/fine-tune a U-Net or DeepLab head on labeled field-boundary masks "
                "(app/models/segmentation.py is the integration point) before enabling this model."
            )
        # Real path, once weights exist:
        # import segmentation_models_pytorch as smp
        # self._model = smp.Unet(encoder_name="resnet34", classes=1, activation="sigmoid")
        # self._model.load_state_dict(torch.load(weights_path, map_location=device))
        # self._model.to(device).eval()

    def validate_model(self) -> tuple[bool, str]:
        return False, "No trained weights loaded — field-boundary segmentation is not production-ready."

    def predict(self, tile_rgb: np.ndarray, confidence_threshold: float | None = None) -> list[RawDetection]:
        raise NotProductionReadyError(
            "field_boundary_segmentation has no trained weights; refusing to fabricate polygons. "
            "See registry.py notes for what training data is required."
        )

    def export_results(self, out_dir: str, detections_geo: list[dict], extra_meta: dict | None = None) -> dict:
        raise NotProductionReadyError("Cannot export results for a model with no trained weights.")

    # --- Real, tested postprocessing math — ready for when weights exist ---------------------
    #
    # predict() above has to raise (no trained weights), but the geometry
    # math that turns a segmentation mask into area/perimeter-bearing
    # polygons does NOT depend on having weights, and IS implemented and
    # tested today (app/engine/measurements.py + tests/test_measurements_
    # offline.py, using cv2 — which, unlike torch, is actually installed in
    # this build sandbox). Once a real model produces a mask, swap predict()
    # to call its forward pass and feed the resulting mask through this.
    def mask_to_results(self, mask: np.ndarray, transform: GeoTransform, min_area_px: int = 16) -> list[dict]:
        """Tile-level binary/probability mask -> export-ready polygon dicts (ring_geo, area_sqm, perimeter_m, class_name)."""
        from app.engine.measurements import mask_to_geo_polygons

        class_name = self.metadata.class_names.get(0, "field")
        polygons = mask_to_geo_polygons(mask, transform, min_area_px=min_area_px)
        return [
            {"ring_geo": p.ring_geo, "area_sqm": p.area_sqm, "perimeter_m": p.perimeter_m, "class_name": class_name}
            for p in polygons
        ]
