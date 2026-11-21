"""
Standardized model interface.

Every detection/segmentation model in the catalog — whatever its underlying
architecture (YOLO, RT-DETR, U-Net, DeepLab, ...) — implements this same
contract, so it is independently loadable, swappable, and deployable, and so
the orchestration pipeline (app/engine/pipeline.py) never needs to know a
model's internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from app.engine.geo import GeoTransform


class ModelStatus(str, Enum):
    READY = "ready"
    LOADING = "loading"
    PROCESSING = "processing"
    TRAINING = "training"
    VALIDATION = "validation"
    ERROR = "error"
    DISABLED = "disabled"
    NOT_PRODUCTION_READY = "not_production_ready"   # no validated weights/dataset yet
    NOT_INSTALLED = "not_installed"                 # open weights exist (app.models.packs); not downloaded yet


@dataclass
class GsdRange:
    min_m: float
    max_m: float

    def supports(self, gsd: float) -> bool:
        return self.min_m <= gsd <= self.max_m


@dataclass
class ModelMetadata:
    model_id: str
    name: str
    model_type: str                 # "detection" | "segmentation"
    version: str
    supported_gsd: GsdRange
    supported_image_sources: list[str]
    supported_formats: list[str]
    status: ModelStatus
    hardware_compatibility: list[str]   # e.g. ["cpu", "cuda"]
    default_confidence_threshold: float
    class_names: dict[int, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class RawDetection:
    box: tuple[float, float, float, float]   # pixel space, tile-local (x1,y1,x2,y2)
    score: float
    class_id: int
    mask: np.ndarray | None = None            # optional, for instance segmentation
    # Optional rotated outline in tile-local pixels, e.g. the 4 corners of an
    # oriented bounding box. When set, it becomes the exported geometry and
    # `box` is its axis-aligned envelope (still used for cross-tile NMS).
    polygon: list[tuple[float, float]] | None = None


class DetectionModel(ABC):
    """Abstract base every model pipeline must implement."""

    metadata: ModelMetadata

    @abstractmethod
    def load_model(self, device: str) -> None:
        """Load weights onto the given device ('cpu' | 'cuda:N'). Must be idempotent."""

    @abstractmethod
    def validate_model(self) -> tuple[bool, str]:
        """Sanity-check the loaded model (shape probe, class-count check, etc). Returns (ok, message)."""

    @abstractmethod
    def predict(self, tile_rgb: np.ndarray, confidence_threshold: float | None = None) -> list[RawDetection]:
        """Run inference on a single RGB tile (H, W, 3) and return raw, tile-local detections."""

    def postprocess(self, detections: list[RawDetection]) -> list[RawDetection]:
        """Per-model postprocessing hook (default: identity). Override for class-specific logic."""
        return detections

    def georeference(self, detections: list[RawDetection], transform: GeoTransform, class_names: dict[int, str]) -> list[dict]:
        """Default georeferencing delegates to the shared helper in engine.postprocess."""
        from app.engine.postprocess import tile_detections_to_global

        raw = [
            {"box": d.box, "score": d.score, "class_id": d.class_id, "polygon": d.polygon}
            for d in detections
        ]
        return tile_detections_to_global(raw, transform, class_names)

    @abstractmethod
    def export_results(self, out_dir: str, detections_geo: list[dict], extra_meta: dict | None = None) -> dict:
        """Write the final artifact(s) (GeoJSON, manifest, ...). Returns output file paths."""

    def gsd_compatibility_report(self, gsd: float) -> dict:
        ok = self.metadata.supported_gsd.supports(gsd)
        return {
            "gsd": gsd,
            "supported": ok,
            "supported_range_m": [self.metadata.supported_gsd.min_m, self.metadata.supported_gsd.max_m],
            "message": (
                f"{self.metadata.name} is validated for {gsd:.2f} m/px."
                if ok
                else (
                    f"{self.metadata.name} has NOT been validated at {gsd:.2f} m/px "
                    f"(supported range: {self.metadata.supported_gsd.min_m}-{self.metadata.supported_gsd.max_m} m). "
                    "Results at this resolution may be unreliable."
                )
            ),
        }
