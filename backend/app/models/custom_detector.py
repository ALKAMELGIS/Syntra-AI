"""
Trained-model runtime.

Loads a user's own fine-tuned YOLO weights — produced by POST
/api/training/train (see app/training/train_worker.py), registered via
app.models.custom_registry — and runs real inference with them, exactly
like app.models.generic_detector.GenericYoloDetector does for the
COCO-pretrained car/ship/aircraft placeholders.

The one real difference: a custom-trained model has no COCO proxy mapping
to do. Its classes ARE the classes the user labeled with the Annotate tool
and exported in that same order (app.training.dataset_builder.
write_data_yaml / read_data_yaml_names), so ultralytics' own class index
on each detection is used directly as class_id — no remapping needed.

`ultralytics`/`torch` stay lazily imported so this module is importable
(for registry/factory wiring, tests) without those installed.
"""
from __future__ import annotations

import numpy as np

from app.models.base import DetectionModel, ModelMetadata, ModelStatus, RawDetection


class CustomYoloDetector(DetectionModel):
    def __init__(self, metadata: ModelMetadata, weights_path: str):
        self.metadata = metadata
        self.weights_path = weights_path
        self._model = None
        self._device = "cpu"

    def load_model(self, device: str) -> None:
        from ultralytics import YOLO  # lazy import — requires torch + ultralytics installed

        self._device = device
        if self._model is None:
            self._model = YOLO(self.weights_path)
        self._model.to(device)

    def validate_model(self) -> tuple[bool, str]:
        if self._model is None:
            return False, "Model not loaded — call load_model() first."
        try:
            names = self._model.names
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"Model loaded but failed introspection: {exc}"
        expected = len(self.metadata.class_names)
        if expected and len(names) != expected:
            return False, (
                f"Weights report {len(names)} class(es) but the catalog entry expects "
                f"{expected} — this registration may point at the wrong weights file."
            )
        return True, f"OK — {len(names)} class(es): {list(names.values())}."

    def predict(self, tile_rgb: np.ndarray, confidence_threshold: float | None = None) -> list[RawDetection]:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first.")
        conf = confidence_threshold if confidence_threshold is not None else self.metadata.default_confidence_threshold

        results = self._model.predict(source=tile_rgb, conf=conf, device=self._device, verbose=False)
        detections: list[RawDetection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes):
                detections.append(RawDetection(
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score), class_id=int(cls),
                ))
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
