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
        name="Palms",
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
        name="Date palms",
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
        name="Trees",
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
    "vegetation_index": ModelMetadata(
        model_id="vegetation_index",
        name="Vegetation",
        model_type="index",
        version="1.0.0-ndvi",
        supported_gsd=GsdRange(0.05, 10.0),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.VALIDATION,
        hardware_compatibility=["cpu"],
        default_confidence_threshold=0.20,
        class_names={0: "vegetation"},
        notes=(
            "Spectral index, no trained model: NDVI from the red and near-infrared bands (VARI for "
            "RGB-only images). Outputs vegetation polygons with area and an estimated count of tree "
            "crowns. The 0.20 threshold is a general default, not calibrated for your area."
        ),
    ),
    "car_detection": ModelMetadata(
        model_id="car_detection",
        name="Cars",
        model_type="detection",
        version="0.2.0-dota-obb",
        supported_gsd=GsdRange(0.1, 0.6),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.VALIDATION,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.35,
        class_names={0: 'car', 1: 'large_vehicle'},
        notes=(
            "Detects small vehicles (cars, vans) and large vehicles (trucks, buses) as separate classes."
            " Pretrained on DOTA v1 aerial imagery (YOLO11-OBB, rotated boxes), not yet fine-tuned or benchmarked on your own imagery: check results before relying on counts."
        ),
    ),
    "ship_detection": ModelMetadata(
        model_id="ship_detection",
        name="Ships",
        model_type="detection",
        version="0.2.0-dota-obb",
        supported_gsd=GsdRange(0.15, 1.5),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.VALIDATION,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.35,
        class_names={0: 'ship'},
        notes=(
            "Detects vessels at sea and moored in ports, with their heading as a rotated box."
            " Pretrained on DOTA v1 aerial imagery (YOLO11-OBB, rotated boxes), not yet fine-tuned or benchmarked on your own imagery: check results before relying on counts."
        ),
    ),
    "drone_objects": ModelMetadata(
        model_id="drone_objects",
        name="Drone objects",
        model_type="detection",
        version="0.1.0-semantic-drone",
        supported_gsd=GsdRange(0.01, 0.15),
        supported_image_sources=["drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.VALIDATION,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.30,
        class_names={0: "tree", 1: "building", 2: "vehicle", 3: "person", 4: "fence", 5: "bicycle"},
        notes="Outlines of trees, roofs, vehicles, people, fences and bicycles in drone orthomosaics.",
    ),
    "field_boundary_segmentation": ModelMetadata(
        model_id="field_boundary_segmentation",
        name="Field boundaries",
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
        name="Vacant land",
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
        name="Street centerlines",
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
        name="Construction waste",
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
        version="0.2.0-dota-obb",
        supported_gsd=GsdRange(0.1, 1.0),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.VALIDATION,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.35,
        class_names={0: 'airplane', 1: 'helicopter'},
        notes=(
            "Detects fixed-wing aircraft and helicopters on aprons, runways and helipads."
            " Pretrained on DOTA v1 aerial imagery (YOLO11-OBB, rotated boxes), not yet fine-tuned or benchmarked on your own imagery: check results before relying on counts."
        ),
    ),
    "unpaved_road_detection": ModelMetadata(
        model_id="unpaved_road_detection",
        name="Unpaved roads",
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
        name="Oil tanks",
        model_type="detection",
        version="0.2.0-dota-obb",
        supported_gsd=GsdRange(0.15, 1.5),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.VALIDATION,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.35,
        class_names={0: 'oil_tank'},
        notes=(
            "Detects DOTA's 'storage tank' class, which covers oil and other cylindrical storage tanks, so it can include water or chemical tanks."
            " Pretrained on DOTA v1 aerial imagery (YOLO11-OBB, rotated boxes), not yet fine-tuned or benchmarked on your own imagery: check results before relying on counts."
        ),
    ),
    "oil_tank_shadow_analysis": ModelMetadata(
        model_id="oil_tank_shadow_analysis",
        name="Tank shadows",
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
        name="Tank volume",
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
        name="Bridges",
        model_type="detection",
        version="0.2.0-dota-obb",
        supported_gsd=GsdRange(0.15, 2.0),
        supported_image_sources=["satellite", "drone"],
        supported_formats=["GeoTIFF", "COG"],
        status=ModelStatus.VALIDATION,
        hardware_compatibility=["cpu", "cuda"],
        default_confidence_threshold=0.3,
        class_names={0: 'bridge'},
        notes=(
            "Detects bridges over water and roads. Long, thin bridges can be split across tiles; overlap helps."
            " Pretrained on DOTA v1 aerial imagery (YOLO11-OBB, rotated boxes), not yet fine-tuned or benchmarked on your own imagery: check results before relying on counts."
        ),
    ),
    "change_detection": ModelMetadata(
        model_id="change_detection",
        name="Change",
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


def _with_pack(meta: ModelMetadata) -> ModelMetadata:
    """Categories backed by an installable pack (app.models.packs) report
    whether the weights are on disk instead of a fixed status."""
    from dataclasses import replace

    from app.models.packs import is_installed, pack_for_model

    pack = pack_for_model(meta.model_id)
    if pack is None:
        return meta
    caveats = " ".join(pack.caveats)
    if is_installed(pack):
        return replace(meta, status=ModelStatus.VALIDATION, version=f"pack:{pack.pack_id}",
                       notes=f"{pack.source} Licence: {pack.licence}. {caveats}".strip())
    return replace(meta, status=ModelStatus.NOT_INSTALLED, version="not-installed",
                   notes=f"Open weights available: {pack.title} ({pack.licence}). Install to run. {pack.source}")


def list_catalog() -> list[ModelMetadata]:
    return [_with_pack(m) for m in CATALOG.values()] + list(_custom_catalog().values())


def get_metadata(model_id: str) -> ModelMetadata:
    if model_id in CATALOG:
        return _with_pack(CATALOG[model_id])
    custom = _custom_catalog()
    if model_id in custom:
        return custom[model_id]
    raise KeyError(f"Unknown model_id '{model_id}'. Known: {list(CATALOG) + list(custom)}")
