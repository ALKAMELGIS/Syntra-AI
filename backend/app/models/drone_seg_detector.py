"""
Drone object outlines: a YOLO11 instance-segmentation model run through
onnxruntime (app.models.onnx_seg), installed as a model pack.

The shipped pack was fine-tuned by Syntra on the TU Graz Semantic Drone
Dataset (nadir drone photos, about 4 cm per pixel as the model sees them),
with its per-pixel labels turned into one outline per connected object.
Its classes are read by name from the ONNX metadata, so a re-trained file
with the same names drops in.

Tiles are read at the model's training scale (`preferred_tile_size`), so
a 2 cm orthomosaic is shrunk and an 8 cm one is enlarged before inference.
"""
from __future__ import annotations

import threading

import numpy as np

from app.models.base import DetectionModel, ModelMetadata, RawDetection

TRAINING_GSD_M = 0.04
_CACHE: dict[tuple[str, int, float], object] = {}
_LOCK = threading.Lock()


def _load(path: str, device: str):
    import os

    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime)
    with _LOCK:
        if key not in _CACHE:
            from app.models.onnx_seg import OnnxSegModel

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else None
            _CACHE[key] = OnnxSegModel(path, providers=providers)
        return _CACHE[key]


class DroneSegDetector(DetectionModel):
    def __init__(self, metadata: ModelMetadata, weights_path: str, training_gsd_m: float = TRAINING_GSD_M):
        self.metadata = metadata
        self.weights_path = weights_path
        self.training_gsd_m = training_gsd_m
        self._model = None
        self._map: dict[int, int] = {}

    def load_model(self, device: str) -> None:
        self._model = _load(self.weights_path, device)
        ours = {v.lower(): k for k, v in self.metadata.class_names.items()}
        self._map = {k: ours[v.lower()] for k, v in self._model.names.items() if v.lower() in ours}

    def validate_model(self) -> tuple[bool, str]:
        if self._model is None:
            return False, "Model not loaded."
        missing = sorted(set(v.lower() for v in self.metadata.class_names.values())
                         - {v.lower() for v in self._model.names.values()})
        if missing:
            return False, f"The model file has no class named {', '.join(missing)}."
        return True, "OK"

    def preferred_tile_size(self, gsd_m: float | None) -> int:
        size = self._model.imgsz if self._model else 640
        if not gsd_m or gsd_m <= 0:
            return size
        return int(np.clip(round(size * self.training_gsd_m / gsd_m), 64, 4096))

    def predict(self, tile_rgb: np.ndarray, confidence_threshold: float | None = None) -> list[RawDetection]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        conf = confidence_threshold if confidence_threshold is not None else self.metadata.default_confidence_threshold
        out = []
        for d in self._model.predict(np.ascontiguousarray(tile_rgb), conf=conf,
                                     nominal_tile=getattr(self, "nominal_tile", None)):
            if d.cls in self._map:
                out.append(RawDetection(box=d.box, score=d.score, class_id=self._map[d.cls], polygon=d.polygon))
        return out

    def export_results(self, out_dir: str, detections_geo: list[dict], extra_meta: dict | None = None) -> dict:
        from app.engine.postprocess import export_results

        meta = {"model_version": self.metadata.version, "model_status": self.metadata.status.value,
                "weights": self.weights_path, "notes": self.metadata.notes, **(extra_meta or {})}
        return export_results(out_dir, self.metadata.model_id, detections_geo, geometry="polygon_detection", extra_meta=meta)
