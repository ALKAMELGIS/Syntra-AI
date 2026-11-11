"""
Model catalog — Phase 1 of the roadmap: the six initial detection categories.

STATUS HONESTY (per the product spec's final requirement): a model is only
ever marked READY once it has (a) real fine-tuned weights trained on labeled
satellite/drone imagery for that class, and (b) a benchmark report (see
app/engine/benchmark.py placeholder, Phase 6). Until then it is reported as
NOT_PRODUCTION_READY — the engine will refuse to claim detections for it
unless `--allow-unvalidated` style flags are explicitly set by a caller who
understands the risk (see ModelRegistry.get_runnable).
"""
from __future__ import annotations

from app.models.base import GsdRange, ModelMetadata, ModelStatus

# ---------------------------------------------------------------------------
# Catalog entries. `runtime` points at the concrete DetectionModel subclass
# key used by the model factory (app/models/factory.py).
# ---------------------------------------------------------------------------

CATALOG: dict[str, ModelMetadata] = {
    "palm_tree_detection": ModelMetadata(
        model_id="palm_tree_detection",
        name="Palm Tree Detection",
        model_type="detection",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 0.75),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.35,
        class_names={0: "palm_tree"},
        notes=(
            "No fine-tuned weights or benchmark yet. Architecture target: YOLOv8/RT-DETR "
            "with dense-canopy NMS tuning for plantation rows. Needs a labeled palm dataset."
        ),
    ),
    "date_tree_detection": ModelMetadata(
        model_id="date_tree_detection",
        name="Date Tree Detection",
        model_type="detection",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 0.75),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.35,
        class_names={0: "date_palm"},
        notes=(
            "No fine-tuned weights. Distinguishing date palms from other palm species requires "
            "class-labeled training data; without it this collapses to generic palm detection."
        ),
    ),
    "tree_detection": ModelMetadata(
        model_id="tree_detection",
        name="Tree Detection",
        model_type="detection",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 1.00),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.30,
        class_names={0: "tree"},
        notes=(
            "No aerial-imagery fine-tuned weights. A generic COCO-pretrained detector has no "
            "'tree' class, so there is no usable unvalidated stand-in here either — disabled "
            "until a labeled vegetation dataset is available."
        ),
    ),
    "car_detection": ModelMetadata(
        model_id="car_detection",
        name="Car Detection",
        model_type="detection",
        version="0.1.0-generic-coco",
        supported_gsd=GsdRange(0.25, 0.50),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.VALIDATION,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.40,
        class_names={0: "car"},
        notes=(
            "Running on generic YOLOv8 COCO-pretrained weights ('car' class) as a working "
            "placeholder — NOT fine-tuned on aerial/nadir imagery and NOT benchmarked. Treat "
            "counts as indicative only until fine-tuned on a labeled aerial car dataset."
        ),
    ),
    "ship_detection": ModelMetadata(
        model_id="ship_detection",
        name="Ship Detection",
        model_type="detection",
        version="0.1.0-generic-coco",
        supported_gsd=GsdRange(0.30, 1.00),
        supported_image_sources=["satellite"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.VALIDATION,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.40,
        class_names={0: "ship"},
        notes=(
            "Running on generic YOLOv8 COCO-pretrained weights ('boat' class as proxy) as a "
            "working placeholder — NOT fine-tuned for maritime satellite imagery (scale/angle "
            "mismatch vs. COCO's photos is significant). Needs a labeled ship dataset."
        ),
    ),
    "field_boundary_segmentation": ModelMetadata(
        model_id="field_boundary_segmentation",
        name="Field Agriculture Boundary Detection",
        model_type="segmentation",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 1.00),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.50,
        class_names={0: "field"},
        notes=(
            "U-Net/DeepLab segmentation head is scaffolded (app/models/segmentation.py) but has "
            "no trained weights. Needs labeled field-boundary masks or polygon annotations."
        ),
    ),
}


def list_catalog() -> list[ModelMetadata]:
    return list(CATALOG.values())


def get_metadata(model_id: str) -> ModelMetadata:
    if model_id not in CATALOG:
        raise KeyError(f"Unknown model_id '{model_id}'. Known: {list(CATALOG)}")
    return CATALOG[model_id]
