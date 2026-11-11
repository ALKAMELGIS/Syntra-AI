"""
Generic YOLO-family detector.

This is REAL inference code (ultralytics), used as the working placeholder
for car_detection / ship_detection (see registry.py notes on why those two
and not the others — COCO has 'car' and 'boat' classes to borrow; it has no
palm/date/tree/field class at all).

`ultralytics`/`torch` are imported lazily so this module stays importable
(for registry/metadata inspection, API schema generation, tests) in any
environment, even ones without a GPU or without torch installed at all.
"""
from __future__ import annotations

import numpy as np

from app.models.base import DetectionModel, ModelMetadata, ModelStatus, RawDetection

# Map our catalog's class name -> COCO class index in the stock weights.
_COCO_CLASS_MAP = {
    "car": 2,     # COCO 'car'
    "ship": 8,    # COCO 'boat' (closest available proxy)
}


class GenericYoloDetector(DetectionModel):
    """
    Thin wrapper standardizing ultralytics YOLO under the common
    DetectionModel interface (load_model / validate_model / predict /
    postprocess / georeference / export_results).
    """

    def __init__(self, metadata: ModelMetadata, weights: str = "yolov8n.pt"):
        self.metadata = metadata
        self.weights = weights
        self._model = None
        self._device = "cpu"
        target_class = metadata.class_names.get(0)
        if target_class not in _COCO_CLASS_MAP:
            raise ValueError(f"No COCO proxy class configured for '{target_class}'.")
        self._coco_class_id = _COCO_CLASS_MAP[target_class]

    def load_model(self, device: str) -> None:
        from ultralytics import YOLO  # lazy import — requires torch + ultralytics installed

        self._device = device
        if self._model is None:
            self._model = YOLO(self.weights)
        self._model.to(device)
        self.metadata.status = ModelStatus.READY if self.metadata.status != ModelStatus.VALIDATION else self.metadata.status

    def validate_model(self) -> tuple[bool, str]:
        if self._model is None:
            return False, "Model not loaded — call load_model() first."
        try:
            names = self._model.names
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"Model loaded but failed introspection: {exc}"
        if self._coco_class_id not in names:
            return False, f"Expected class id {self._coco_class_id} not found in model's class list."
        return True, f"OK — '{names[self._coco_class_id]}' available as class {self._coco_class_id}."

    def predict(self, tile_rgb: np.ndarray, confidence_threshold: float | None = None) -> list[RawDetection]:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first.")
        conf = confidence_threshold if confidence_threshold is not None else self.metadata.default_confidence_threshold

        results = self._model.predict(
            source=tile_rgb,
            conf=conf,
            classes=[self._coco_class_id],
            device=self._device,
            verbose=False,
        )
        detections: list[RawDetection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), score in zip(xyxy, scores):
                detections.append(RawDetection(box=(float(x1), float(y1), float(x2), float(y2)),
                                                score=float(score), class_id=0))
        return detections

    def export_results(self, out_dir: str, detections_geo: list[dict], extra_meta: dict | None = None) -> dict:
        from app.engine.postprocess import export_results

        meta = {
            "model_version": self.metadata.version,
            "model_status": self.metadata.status.value,
            "notes": self.metadata.notes,
            **(extra_meta or {}),
        }
        return export_results(out_dir, self.metadata.model_id, detections_geo, geometry="polygon_detection", extra_meta=meta)
