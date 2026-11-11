"""
Dataset & Training Pipeline module — spec Section 9.

Pure dataclasses + plain Python so this whole module is importable and
testable without rasterio/torch — only the actual model-training step
(not implemented yet; no trained models exist per registry.py) needs them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AnnotationFormat(str, Enum):
    YOLO = "yolo"
    COCO = "coco"
    GEOJSON = "geojson"


@dataclass
class BoundingBox:
    """Normalized (0..1) YOLO-style box, OR absolute pixel box — `normalized` says which."""
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int
    normalized: bool = True

    def to_xyxy(self, image_w: int | None = None, image_h: int | None = None) -> tuple[float, float, float, float]:
        if self.normalized:
            if image_w is None or image_h is None:
                raise ValueError("image_w/image_h required to convert a normalized box to pixel xyxy.")
            cx, cy, w, h = self.x_center * image_w, self.y_center * image_h, self.width * image_w, self.height * image_h
        else:
            cx, cy, w, h = self.x_center, self.y_center, self.width, self.height
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


@dataclass
class DatasetSample:
    """One labeled image (or raster tile) in a dataset."""
    sample_id: str
    image_path: str
    boxes: list[BoundingBox] = field(default_factory=list)
    polygon_rings: list[list[tuple[float, float]]] = field(default_factory=list)  # for segmentation
    # Geo footprint (lon/lat or projected bbox), used by the geographically
    # separated splitter to keep adjacent tiles out of different splits.
    geo_bounds: tuple[float, float, float, float] | None = None  # (minx, miny, maxx, maxy)
    image_w: int | None = None
    image_h: int | None = None
    source_gsd_m: float | None = None


@dataclass
class ClassBalanceReport:
    counts: dict[int, int]
    total: int
    imbalance_ratio: float   # max_count / min_count (>= 1.0; 1.0 == perfectly balanced)
    underrepresented_classes: list[int]  # classes below 10% of the mean count

    def summary(self) -> str:
        if not self.counts:
            return "No annotated instances found."
        lines = [f"class {c}: {n} ({n / self.total:.1%})" for c, n in sorted(self.counts.items())]
        warn = f" WARNING: underrepresented classes {self.underrepresented_classes}" if self.underrepresented_classes else ""
        return "; ".join(lines) + warn


@dataclass
class DatasetSplit:
    train: list[str]   # sample_ids
    val: list[str]
    test: list[str]
    geographically_separated: bool
    notes: str = ""


@dataclass
class DatasetVersion:
    version_id: str
    created_at: str
    model_id: str
    sample_count: int
    class_balance: ClassBalanceReport
    split: DatasetSplit
    annotation_format: AnnotationFormat
    source_paths: list[str] = field(default_factory=list)
