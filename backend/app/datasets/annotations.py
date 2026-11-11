"""
Annotation parsing + validation for the three formats the spec requires:
YOLO (.txt per image), COCO (.json), GeoJSON (polygons, for field-boundary
or any geo-referenced annotation set).

All parsing here is pure Python/stdlib (json, csv-free text parsing) — no
GDAL/rasterio/pycocotools dependency — so dataset validation can run before
a full ML environment is installed.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.datasets.schema import AnnotationFormat, BoundingBox, DatasetSample


class AnnotationError(ValueError):
    pass


def parse_yolo_txt(txt_path: str | Path, image_path: str, image_w: int, image_h: int) -> DatasetSample:
    """Each line: `class_id x_center y_center width height` (all normalized 0..1)."""
    txt_path = Path(txt_path)
    boxes: list[BoundingBox] = []
    if not txt_path.exists():
        raise AnnotationError(f"YOLO annotation file not found: {txt_path}")

    for lineno, raw in enumerate(txt_path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise AnnotationError(f"{txt_path}:{lineno}: expected 5 fields, got {len(parts)}: '{line}'")
        cls_id, xc, yc, w, h = parts
        try:
            cls_id = int(cls_id)
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)
        except ValueError as exc:
            raise AnnotationError(f"{txt_path}:{lineno}: non-numeric field: {exc}")
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
            raise AnnotationError(f"{txt_path}:{lineno}: YOLO coordinates must be normalized in [0,1], got {parts[1:]}")
        boxes.append(BoundingBox(x_center=xc, y_center=yc, width=w, height=h, class_id=cls_id, normalized=True))

    return DatasetSample(sample_id=txt_path.stem, image_path=image_path, boxes=boxes, image_w=image_w, image_h=image_h)


def parse_coco_json(coco_path: str | Path, images_dir: str | Path) -> list[DatasetSample]:
    """Standard COCO detection format: {images: [...], annotations: [...], categories: [...]}."""
    coco_path = Path(coco_path)
    if not coco_path.exists():
        raise AnnotationError(f"COCO annotation file not found: {coco_path}")

    data = json.loads(coco_path.read_text())
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise AnnotationError(f"COCO file missing required key '{key}'.")

    images_by_id = {img["id"]: img for img in data["images"]}
    samples: dict[int, DatasetSample] = {}

    for img_id, img in images_by_id.items():
        if "width" not in img or "height" not in img or "file_name" not in img:
            raise AnnotationError(f"COCO image entry {img_id} missing width/height/file_name.")
        samples[img_id] = DatasetSample(
            sample_id=str(img_id),
            image_path=str(Path(images_dir) / img["file_name"]),
            image_w=img["width"],
            image_h=img["height"],
        )

    for ann in data["annotations"]:
        img_id = ann.get("image_id")
        if img_id not in samples:
            raise AnnotationError(f"Annotation references unknown image_id {img_id}.")
        bbox = ann.get("bbox")  # COCO: [x, y, width, height], absolute pixels, top-left origin
        if not bbox or len(bbox) != 4:
            raise AnnotationError(f"Annotation {ann.get('id')} has invalid bbox: {bbox}")
        x, y, w, h = bbox
        sample = samples[img_id]
        box = BoundingBox(
            x_center=x + w / 2, y_center=y + h / 2, width=w, height=h,
            class_id=ann["category_id"], normalized=False,
        )
        sample.boxes.append(box)

    return list(samples.values())


def parse_geojson_annotations(geojson_path: str | Path, class_field: str = "class") -> list[DatasetSample]:
    """
    Geo-referenced polygon annotations (field boundaries, or any georeferenced
    label set). Each Feature's geometry ring becomes `polygon_rings`, and the
    feature's bbox becomes `geo_bounds` for the geographic splitter.
    """
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        raise AnnotationError(f"GeoJSON file not found: {geojson_path}")

    data = json.loads(geojson_path.read_text())
    if data.get("type") != "FeatureCollection":
        raise AnnotationError("Expected a GeoJSON FeatureCollection.")

    samples = []
    for i, feat in enumerate(data.get("features", [])):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            raise AnnotationError(f"Feature {i}: only Polygon geometries are supported, got {geom.get('type')}.")
        ring = geom["coordinates"][0]
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
        samples.append(
            DatasetSample(
                sample_id=str(feat.get("id", i)),
                image_path="",  # geo annotations may not be tied to one raster file
                polygon_rings=[[tuple(c) for c in ring]],
                geo_bounds=(min(xs), min(ys), max(xs), max(ys)),
            )
        )
    return samples


def validate_dataset(samples: list[DatasetSample]) -> list[str]:
    """Returns a list of human-readable problems (empty list == dataset looks sane)."""
    problems: list[str] = []
    if not samples:
        return ["Dataset is empty — no samples found."]

    seen_ids = set()
    for s in samples:
        if s.sample_id in seen_ids:
            problems.append(f"Duplicate sample_id: {s.sample_id}")
        seen_ids.add(s.sample_id)

        if not s.boxes and not s.polygon_rings:
            problems.append(f"Sample {s.sample_id} has no annotations at all.")

        for b in s.boxes:
            if b.width <= 0 or b.height <= 0:
                problems.append(f"Sample {s.sample_id}: non-positive box size (w={b.width}, h={b.height}).")

    return problems
