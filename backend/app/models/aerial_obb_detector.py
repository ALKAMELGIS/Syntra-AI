"""
Aerial oriented-box detector (YOLO11-OBB pretrained on DOTA v1).

Why this replaces the COCO proxies for car / ship / aircraft:
COCO is street-level photography. A car seen from the side in a photo looks
nothing like a car seen from 500 km up, so the old "car" and "boat" proxies
were a placeholder at best. DOTA is a large public dataset of *aerial and
satellite* scenes labelled with rotated boxes, so these weights have
actually seen nadir imagery of the objects we care about.

DOTA v1 also covers two catalog entries that had no runnable model at all:
"storage tank" (Oil Tanks) and "bridge" (Bridge).

What this still is NOT: fine-tuned or benchmarked on the user's own imagery
(region, sensor, GSD). Every model backed by it stays VALIDATION until a
benchmark against labelled local data says otherwise.

Each catalog model selects the DOTA classes it wants by *name* and maps them
to its own class ids, so a different OBB checkpoint with the same class
names (e.g. a fine-tune) drops in without code changes.

`ultralytics`/`torch` are imported lazily, as everywhere else in this repo.
"""
from __future__ import annotations

import threading

import numpy as np

from app.models.base import DetectionModel, ModelMetadata, RawDetection

# catalog model_id -> {DOTA class name: our class id}
DOTA_CLASS_MAP: dict[str, dict[str, int]] = {
    "car_detection": {"small vehicle": 0, "large vehicle": 1},
    "ship_detection": {"ship": 0},
    "aircraft_detection": {"plane": 0, "helicopter": 1},
    "oil_tank_detection": {"storage tank": 0},
    "bridge_detection": {"bridge": 0},
}

# One loaded network per (weights, device), shared by every catalog model
# that uses it: a run of "cars + ships + aircraft" loads the weights once.
_MODEL_CACHE: dict[tuple[str, str], object] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_yolo(weights: str, device: str):
    """ultralytics YOLO for .pt weights; torch-free onnxruntime for .onnx."""
    key = (weights, device)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            if str(weights).lower().endswith(".onnx"):
                from app.models.onnx_obb import OnnxObbModel

                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else None
                model = OnnxObbModel(weights, providers=providers)
            else:
                from ultralytics import YOLO  # lazy: needs torch + ultralytics

                model = YOLO(weights)
                model.to(device)
            _MODEL_CACHE[key] = model
        return model


def resolve_obb_weights(configured: str) -> str:
    """The configured path when it exists; otherwise the same model found in a
    model folder (ONNX preferred: it needs no torch). Falls back to the
    configured name, which ultralytics downloads on first use."""
    from pathlib import Path

    if configured and Path(configured).is_file():
        return configured
    from app.models.packs import find_model_file

    stem = Path(configured or "yolo11n-obb.pt").stem
    found = find_model_file(f"{stem}.onnx", f"{stem}.pt", Path(configured).name if configured else "")
    return str(found) if found else configured


def _names_dict(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


class AerialObbDetector(DetectionModel):
    def __init__(self, metadata: ModelMetadata, weights: str | None = None):
        from app.core.config import settings

        if metadata.model_id not in DOTA_CLASS_MAP:
            raise ValueError(f"No DOTA class mapping configured for '{metadata.model_id}'.")
        self.metadata = metadata
        self.weights = weights or resolve_obb_weights(settings.AERIAL_OBB_WEIGHTS)
        self.wanted = DOTA_CLASS_MAP[metadata.model_id]
        self._model = None
        self._device = "cpu"
        self._dota_to_ours: dict[int, int] = {}

    # ------------------------------------------------------------ lifecycle
    def load_model(self, device: str) -> None:
        self._device = device
        self._model = _load_yolo(self.weights, device)
        self._bind_classes()

    def _bind_classes(self) -> None:
        names = _names_dict(self._model.names)
        by_name = {v.lower(): k for k, v in names.items()}
        self._dota_to_ours = {
            by_name[dota_name]: our_id
            for dota_name, our_id in self.wanted.items()
            if dota_name in by_name
        }

    def validate_model(self) -> tuple[bool, str]:
        if self._model is None:
            return False, "Model not loaded — call load_model() first."
        task = getattr(self._model, "task", "obb")
        if task != "obb":
            return False, f"'{self.weights}' is a '{task}' model; an oriented-box (obb) model is required."
        missing = [n for n in self.wanted if n not in {v.lower() for v in _names_dict(self._model.names).values()}]
        if missing:
            return False, f"'{self.weights}' has no class named {', '.join(repr(m) for m in missing)}."
        return True, "OK — " + ", ".join(f"'{n}'" for n in self.wanted) + " available."

    # ------------------------------------------------------------ inference
    def predict(self, tile_rgb: np.ndarray, confidence_threshold: float | None = None) -> list[RawDetection]:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first.")
        if not self._dota_to_ours:
            return []
        conf = confidence_threshold if confidence_threshold is not None else self.metadata.default_confidence_threshold

        # ultralytics treats numpy input as BGR (OpenCV order); our tiles are RGB.
        tile_bgr = np.ascontiguousarray(tile_rgb[..., ::-1])
        from app.models.onnx_obb import OnnxObbModel
        if isinstance(self._model, OnnxObbModel):
            dets = self._model.predict(tile_bgr, conf=conf, classes=sorted(self._dota_to_ours),
                                       nominal_tile=getattr(self, "nominal_tile", None))
            return [
                RawDetection(
                    box=(float(d.corners[:, 0].min()), float(d.corners[:, 1].min()),
                         float(d.corners[:, 0].max()), float(d.corners[:, 1].max())),
                    score=d.score,
                    class_id=self._dota_to_ours[d.cls],
                    polygon=[(float(x), float(y)) for x, y in d.corners],
                )
                for d in dets if d.cls in self._dota_to_ours
            ]
        results = self._model.predict(
            source=tile_bgr,
            conf=conf,
            classes=sorted(self._dota_to_ours),
            device=self._device,
            imgsz=max(tile_rgb.shape[:2]),
            verbose=False,
        )
        return self.detections_from_results(results)

    def detections_from_results(self, results) -> list[RawDetection]:
        """Convert ultralytics OBB results to RawDetections (split out so it
        can be tested without torch)."""
        out: list[RawDetection] = []
        for r in results:
            obb = getattr(r, "obb", None)
            if obb is None or len(obb) == 0:
                continue
            corners = _to_numpy(obb.xyxyxyxy)   # (N, 4, 2) rotated corners, pixels
            envelopes = _to_numpy(obb.xyxy)     # (N, 4) axis-aligned envelope
            classes = _to_numpy(obb.cls).astype(int)
            scores = _to_numpy(obb.conf)
            for quad, env, cls, score in zip(corners, envelopes, classes, scores):
                ours = self._dota_to_ours.get(int(cls))
                if ours is None:
                    continue
                out.append(RawDetection(
                    box=tuple(float(v) for v in env),
                    score=float(score),
                    class_id=ours,
                    polygon=[(float(x), float(y)) for x, y in quad],
                ))
        return out

    def export_results(self, out_dir: str, detections_geo: list[dict], extra_meta: dict | None = None) -> dict:
        from app.engine.postprocess import export_results

        meta = {
            "model_version": self.metadata.version,
            "model_status": self.metadata.status.value,
            "weights": self.weights,
            "notes": self.metadata.notes,
            **(extra_meta or {}),
        }
        return export_results(out_dir, self.metadata.model_id, detections_geo, geometry="polygon_detection", extra_meta=meta)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)
