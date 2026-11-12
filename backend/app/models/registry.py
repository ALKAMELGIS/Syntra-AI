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
from app.models.custom_registry import list_custom_models

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
    # ------------------------------------------------------------------
    # Extended catalog (Phase 5) — added to match the full category list
    # the product's reference UI exposes. Same honesty rule as Phase 1:
    # a category is only ever READY once real fine-tuned weights + a
    # benchmark exist. "Aircraft" gets the same generic-COCO-proxy
    # treatment as car/ship because COCO already has an 'airplane' class;
    # none of the other categories below have any usable COCO/ImageNet
    # equivalent, so they are honestly NOT_PRODUCTION_READY rather than
    # faked with an unrelated pretrained class.
    # ------------------------------------------------------------------
    "building_detection": ModelMetadata(
        model_id="building_detection",
        name="Buildings",
        model_type="detection",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.10, 0.50),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.40,
        class_names={0: "building"},
        notes=(
            "No fine-tuned weights. COCO has no 'building' class, so there is no honest generic "
            "stand-in — needs a labeled building-footprint dataset (e.g. a SpaceNet-style corpus)."
        ),
    ),
    "vacant_land_detection": ModelMetadata(
        model_id="vacant_land_detection",
        name="Vacant Land",
        model_type="segmentation",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 1.00),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.50,
        class_names={0: "vacant_land"},
        notes=(
            "A land-use segmentation task (vacant vs. built/vegetated), not a bounding-box "
            "detector. No trained weights or labeled land-use masks exist yet."
        ),
    ),
    "street_centerline_extraction": ModelMetadata(
        model_id="street_centerline_extraction",
        name="Street (centerline)",
        model_type="segmentation",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.10, 0.50),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.45,
        class_names={0: "street_centerline"},
        notes=(
            "Road-surface segmentation plus a skeletonization step to reduce a paved-road mask "
            "to a single centerline per segment. No trained weights yet, and the skeletonization "
            "postprocessing step itself isn't written."
        ),
    ),
    "construction_waste_detection": ModelMetadata(
        model_id="construction_waste_detection",
        name="Construction Waste",
        model_type="detection",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.10, 0.30),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.40,
        class_names={0: "construction_waste"},
        notes=(
            "No fine-tuned weights or labeled dataset exist. Debris piles have no usable COCO "
            "proxy class, unlike car/ship/aircraft."
        ),
    ),
    "aircraft_detection": ModelMetadata(
        model_id="aircraft_detection",
        name="Aircraft",
        model_type="detection",
        version="0.1.0-generic-coco",
        supported_gsd=GsdRange(0.25, 0.75),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.VALIDATION,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.40,
        class_names={0: "airplane"},
        notes=(
            "Running on generic YOLOv8 COCO-pretrained weights ('airplane' class) as a working "
            "placeholder — NOT fine-tuned on aerial/nadir imagery and NOT benchmarked. Treat "
            "counts as indicative only until fine-tuned on a labeled aerial-aircraft dataset."
        ),
    ),
    "unpaved_road_detection": ModelMetadata(
        model_id="unpaved_road_detection",
        name="Unpaved Road",
        model_type="segmentation",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 1.00),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.45,
        class_names={0: "unpaved_road"},
        notes=(
            "Texture-based segmentation (unpaved vs. paved vs. bare ground) — no trained weights "
            "or labeled masks exist yet."
        ),
    ),
    "wall_detection": ModelMetadata(
        model_id="wall_detection",
        name="Walls",
        model_type="detection",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.10, 0.30),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.40,
        class_names={0: "wall"},
        notes="No fine-tuned weights or labeled dataset exist. Linear/thin structures need a dedicated line-detector, not a stock box detector.",
    ),
    "oil_tank_detection": ModelMetadata(
        model_id="oil_tank_detection",
        name="Oil Tanks",
        model_type="detection",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 0.75),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.45,
        class_names={0: "oil_tank"},
        notes=(
            "Circular storage tanks have no usable COCO proxy class. Needs a labeled oil-tank "
            "dataset (e.g. a Kaggle 'Airbus Oil Storage' style corpus) before this can run."
        ),
    ),
    "oil_tank_shadow_analysis": ModelMetadata(
        model_id="oil_tank_shadow_analysis",
        name="Oil Shadows",
        model_type="measurement",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 0.75),
        supported_image_sources=["satellite"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu"],
        default_confidence_threshold=0.50,
        class_names={},
        notes=(
            "A derived measurement (floating-roof shadow geometry + sun elevation/azimuth from "
            "image metadata), not a standalone detector — depends entirely on 'Oil Tanks' "
            "detections it doesn't have yet. Not implemented."
        ),
    ),
    "tank_volume_estimation": ModelMetadata(
        model_id="tank_volume_estimation",
        name="Tank Volume",
        model_type="measurement",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 0.75),
        supported_image_sources=["satellite"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu"],
        default_confidence_threshold=0.50,
        class_names={},
        notes=(
            "Derived from 'Oil Shadows' + known tank diameter/height assumptions to estimate "
            "fill level. Chained on two models that don't exist yet — not implemented."
        ),
    ),
    "bridge_detection": ModelMetadata(
        model_id="bridge_detection",
        name="Bridge",
        model_type="detection",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 1.00),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.40,
        class_names={0: "bridge"},
        notes="No fine-tuned weights or labeled dataset exist. No usable COCO proxy class.",
    ),
    "change_detection": ModelMetadata(
        model_id="change_detection",
        name="Change Detection",
        model_type="change_detection",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.25, 1.00),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.50,
        class_names={},
        notes=(
            "Needs a bi-temporal pipeline (two co-registered images of the same area, a Siamese "
            "change-detection network or diffing of per-image detections) — a fundamentally "
            "different input contract from every other single-image model in this catalog. "
            "Neither the pipeline nor any weights exist yet."
        ),
    ),
    "container_detection": ModelMetadata(
        model_id="container_detection",
        name="Containers",
        model_type="detection",
        version="0.0.0-unweighted",
        supported_gsd=GsdRange(0.10, 0.50),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.NOT_PRODUCTION_READY,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.40,
        class_names={0: "container"},
        notes=(
            "Shipping/storage containers have no usable COCO proxy class. Needs a labeled "
            "container-yard dataset before this can run."
        ),
    ),
}


def _custom_catalog() -> dict[str, ModelMetadata]:
    """The dynamic half of the catalog: models actually trained via the
    wizard's Train step (app.models.custom_registry), turned into the same
    ModelMetadata shape as every static CATALOG entry above. Re-read on
    every call (like app.core.image_library) rather than cached, so a model
    trained or deleted a moment ago shows up immediately with no restart."""
    from app.core.config import settings

    out: dict[str, ModelMetadata] = {}
    for e in list_custom_models(settings.DATA_DIR):
        out[e.model_id] = ModelMetadata(
            model_id=e.model_id,
            name=e.name,
            model_type="segmentation" if e.task == "segment" else "detection",
            version=f"trained-{e.created_at or 'unknown'}",
            # GSD compatibility for a user-trained model hasn't been measured —
            # a wide, honestly-unvalidated range rather than a guessed-tight one.
            supported_gsd=GsdRange(0.02, 5.0),
            supported_image_sources=["satellite", "drone"],
            supported_formats=["GeoTIFF", "COG"],
            status=ModelStatus.VALIDATION,
            hardware_compatibility=["cpu", "cuda"],
            default_confidence_threshold=0.25,
            class_names=e.class_names,
            notes=(
                e.notes
                or "Trained via the Model Setup wizard's Train step on your own annotated "
                "imagery — real fine-tuned weights, but no benchmark report has been run "
                "yet. Treat detections as indicative until validated against a held-out set."
            ),
        )
    return out


def list_catalog() -> list[ModelMetadata]:
    return list(CATALOG.values()) + list(_custom_catalog().values())


def get_metadata(model_id: str) -> ModelMetadata:
    if model_id in CATALOG:
        return CATALOG[model_id]
    custom = _custom_catalog()
    if model_id in custom:
        return custom[model_id]
    raise KeyError(f"Unknown model_id '{model_id}'. Known: {list(CATALOG) + list(custom)}")
