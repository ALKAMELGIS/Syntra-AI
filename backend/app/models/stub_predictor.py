"""
Deterministic offline stub — FOR TESTS ONLY, never wired into the live API.

Exercises the full DetectionModel contract (load/validate/predict/
postprocess/georeference/export_results) and the tiling + cross-tile NMS +
GeoJSON export pipeline end-to-end, without requiring torch/ultralytics/
rasterio. It finds bright, roughly-square blobs in a tile using plain NumPy
thresholding — a deterministic, inspectable stand-in for a trained detector,
used to prove the *orchestration* logic is correct in environments (like this
build sandbox) that can't install the real ML stack.

This module must never be selected by app/models/factory.py for a real
prediction request; it exists only so `backend/tests/test_pipeline_offline.py`
can run without GPU/network access.
"""
from __future__ import annotations

import numpy as np

from app.models.base import DetectionModel, ModelMetadata, RawDetection


class OfflineStubDetector(DetectionModel):
    def __init__(self, metadata: ModelMetadata, brightness_threshold: int = 200, min_blob_px: int = 9):
        self.metadata = metadata
        self.brightness_threshold = brightness_threshold
        self.min_blob_px = min_blob_px
        self._loaded = False

    def load_model(self, device: str) -> None:
        self._loaded = True  # nothing to actually load

    def validate_model(self) -> tuple[bool, str]:
        return self._loaded, "stub predictor ready" if self._loaded else "call load_model() first"

    def predict(self, tile_rgb: np.ndarray, confidence_threshold: float | None = None) -> list[RawDetection]:
        """Connected-components over a brightness threshold — deterministic, no ML involved."""
        if not self._loaded:
            raise RuntimeError("call load_model() first")

        gray = tile_rgb.mean(axis=-1) if tile_rgb.ndim == 3 else tile_rgb
        binary = (gray >= self.brightness_threshold).astype(np.uint8)

        detections: list[RawDetection] = []
        visited = np.zeros_like(binary, dtype=bool)
        h, w = binary.shape
        for y in range(h):
            for x in range(w):
                if binary[y, x] and not visited[y, x]:
                    # flood fill (4-connectivity) via simple stack — tiles are small in tests
                    stack = [(y, x)]
                    pixels = []
                    visited[y, x] = True
                    while stack:
                        cy, cx = stack.pop()
                        pixels.append((cy, cx))
                        for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                    if len(pixels) < self.min_blob_px:
                        continue
                    ys = [p[0] for p in pixels]
                    xs = [p[1] for p in pixels]
                    x1, x2, y1, y2 = min(xs), max(xs) + 1, min(ys), max(ys) + 1
                    score = min(1.0, len(pixels) / (self.min_blob_px * 4))
                    detections.append(RawDetection(box=(float(x1), float(y1), float(x2), float(y2)),
                                                    score=float(score), class_id=0))
        return detections

    def export_results(self, out_dir: str, detections_geo: list[dict], extra_meta: dict | None = None) -> dict:
        from app.engine.postprocess import export_results

        meta = {"model_status": "offline_stub_do_not_use_in_production", **(extra_meta or {})}
        return export_results(out_dir, self.metadata.model_id, detections_geo, geometry="polygon_detection", extra_meta=meta)
