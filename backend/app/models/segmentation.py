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

    # --- Reference implementation notes for the real pipeline, once weights exist -----------
    #
    # def postprocess(self, mask_prob: np.ndarray) -> list[dict]:
    #     """mask_prob: (H, W) float32 in [0,1]. Returns polygons with area/perimeter in pixel space."""
    #     import cv2
    #     binary = (mask_prob > 0.5).astype(np.uint8)
    #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     polygons = []
    #     for cnt in contours:
    #         if cv2.contourArea(cnt) < 16:  # drop speckle
    #             continue
    #         ring = cnt.reshape(-1, 2).tolist()
    #         polygons.append({"ring_px": ring, "area_px": cv2.contourArea(cnt), "perimeter_px": cv2.arcLength(cnt, True)})
    #     return polygons
