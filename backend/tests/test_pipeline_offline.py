"""
Offline, dependency-light smoke test for the detection engine's
*orchestration* logic: tiling, georeferencing, cross-tile NMS, and GeoJSON
export.

Deliberately avoids rasterio / torch / ultralytics so it runs in any
environment (including CI containers and this build sandbox) with only
numpy installed. It proves the pipeline wiring is correct; it says nothing
about real detection accuracy, which depends on trained weights (see
app/models/registry.py statuses).

Run with:  python -m pytest backend/tests/test_pipeline_offline.py -v
       or:  python backend/tests/test_pipeline_offline.py   (plain script mode)
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on sys.path

from app.core.compute import DeviceOption
from app.engine.geo import GeoTransform
from app.engine.nms import merge_tile_duplicates
from app.engine.pipeline import DetectionEngine
from app.engine.postprocess import tile_detections_to_global
from app.engine.raster_io import iter_tiles_from_array
from app.models.base import ModelMetadata, GsdRange, ModelStatus
from app.models.stub_predictor import OfflineStubDetector


def make_synthetic_orthomosaic(size: int = 256, n_blobs: int = 12, seed: int = 7) -> np.ndarray:
    """Dark background with bright square 'objects' scattered around — a stand-in orthomosaic."""
    rng = np.random.default_rng(seed)
    img = (rng.integers(20, 60, size=(size, size, 3))).astype(np.uint8)  # dark noisy background
    for _ in range(n_blobs):
        s = rng.integers(6, 14)
        y = rng.integers(0, size - s)
        x = rng.integers(0, size - s)
        img[y:y + s, x:x + s] = rng.integers(220, 255, size=3)
    return img


def test_end_to_end_pipeline():
    image = make_synthetic_orthomosaic()
    transform = GeoTransform.identity_with_gsd(gsd=0.30, origin_x=500000.0, origin_y=2700000.0)

    metadata = ModelMetadata(
        model_id="offline_test_blob_detector",
        name="Offline Test Blob Detector",
        model_type="detection",
        version="test",
        supported_gsd=GsdRange(0.25, 0.75),
        supported_image_sources=["synthetic"],
        supported_formats=["array"],
        status=ModelStatus.READY,
        hardware_compatibility=["cpu"],
        default_confidence_threshold=0.3,
        class_names={0: "bright_blob"},
    )
    model = OfflineStubDetector(metadata)
    engine = DetectionEngine(model)

    tmp_out = Path(tempfile.mkdtemp(prefix="syntra_offline_test_"))
    try:
        tiles = list(iter_tiles_from_array(image, transform, tile_size=128, overlap=0.25))
        assert len(tiles) >= 4, "Expected the 256x256 image to split into multiple overlapping tiles"

        result = engine.run(tiles, tmp_out, device_option=DeviceOption.CPU1)

        assert result.tiles_processed == len(tiles)
        assert result.raw_detection_count >= result.final_detection_count, (
            "Cross-tile NMS should never increase detection count"
        )
        assert result.final_detection_count > 0, "Should detect at least some synthetic bright blobs"
        assert Path(result.geojson_path).exists()

        import json
        geojson = json.loads(Path(result.geojson_path).read_text())
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == result.final_detection_count
        for feat in geojson["features"]:
            ring = feat["geometry"]["coordinates"][0]  # polygon_detection -> Polygon features
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            assert 500000.0 <= min(xs) and max(xs) <= 500000.0 + 256 * 0.30 + 1
            assert 2700000.0 - 256 * 0.30 - 1 <= min(ys) and max(ys) <= 2700000.0

        print(f"OK: {result.tiles_processed} tiles, {result.raw_detection_count} raw -> "
              f"{result.final_detection_count} after cross-tile NMS, device={result.device_used}")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)


def test_cross_tile_dedup_merges_overlapping_boxes_under_north_up_transform():
    """
    Regression test for a real bug: tile_detections_to_global() used to build
    box_geo as (ring_geo[0].x, ring_geo[0].y, ring_geo[2].x, ring_geo[2].y)
    verbatim. Under a standard north-up affine transform (negative pixel
    height), that flips the y-axis and produces (xmin, ymax, xmax, ymin)
    instead of (xmin, ymin, xmax, ymax) -- silently breaking nms.py's IoU
    math (negative height -> intersection/area miscomputed) so cross-tile
    duplicate suppression never fired for any real north-up GeoTIFF, even
    though the original test_end_to_end_pipeline() above kept passing
    (it only asserts raw >= final, which is trivially true when NMS does
    nothing at all).
    """
    # North-up transform: pixel height (e) is negative, exactly like a real GeoTIFF.
    transform = GeoTransform.identity_with_gsd(gsd=1.0, origin_x=0.0, origin_y=100.0)
    assert transform.e < 0, "this test only exercises the bug when e is negative (north-up)"

    heavily_overlapping = [
        {"box": (10, 10, 30, 30), "score": 0.9, "class_id": 0},
        {"box": (12, 12, 32, 32), "score": 0.8, "class_id": 0},  # >70% IoU with the first
    ]
    geo_dets = tile_detections_to_global(heavily_overlapping, transform, {0: "car"})

    # box_geo must be true (xmin, ymin, xmax, ymax), whatever the transform's
    # axis orientation -- never raw corner-0/corner-2 coordinates.
    for d in geo_dets:
        x1, y1, x2, y2 = d["box_geo"]
        assert x1 <= x2 and y1 <= y2, f"box_geo is not (xmin, ymin, xmax, ymax): {d['box_geo']}"

    merged = merge_tile_duplicates(geo_dets, iou_threshold=0.45)
    assert len(merged) == 1, (
        f"expected the two >70%-overlapping detections to merge into 1, got {len(merged)} -- "
        "cross-tile duplicate suppression is not working under a north-up transform"
    )

    # Sanity check: two genuinely non-overlapping detections must NOT be merged.
    non_overlapping = [
        {"box": (10, 10, 20, 20), "score": 0.9, "class_id": 0},
        {"box": (100, 100, 120, 120), "score": 0.8, "class_id": 0},
    ]
    geo_dets2 = tile_detections_to_global(non_overlapping, transform, {0: "car"})
    merged2 = merge_tile_duplicates(geo_dets2, iou_threshold=0.45)
    assert len(merged2) == 2, f"expected 2 distinct detections to survive, got {len(merged2)}"


if __name__ == "__main__":
    test_end_to_end_pipeline()
    test_cross_tile_dedup_merges_overlapping_boxes_under_north_up_transform()
    print("test_pipeline_offline: PASSED")
