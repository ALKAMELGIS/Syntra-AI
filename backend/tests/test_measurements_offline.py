"""
Offline test for field-boundary measurement math (spec Section 1, Model 06:
"Calculate field area in square meters and hectares. Extract perimeter.").

Uses a synthetic binary mask with a known square field, a known GSD, and
hand-computed expected area/perimeter — this is real cv2 contour extraction
(opencv-python IS installed in this sandbox) plus the shoelace-formula area
and perimeter math, not a mock.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.geo import GeoTransform
from app.engine.measurements import mask_to_geo_polygons, polygon_perimeter, shoelace_area
from app.models.base import GsdRange, ModelMetadata, ModelStatus
from app.models.segmentation import FieldBoundarySegmenter


def test_shoelace_area_known_square():
    # A simple 10x10 axis-aligned square: area = 100, perimeter = 40.
    ring = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    assert abs(shoelace_area(ring) - 100.0) < 1e-9
    assert abs(polygon_perimeter(ring) - 40.0) < 1e-9
    print("Shoelace area/perimeter on a known square: OK")


def test_mask_to_geo_polygons_known_field():
    # 300x300 px mask, a 200x200 px square field in the middle, GSD = 0.5 m/px.
    # Expected real-world size: 100m x 100m = 10,000 sqm = 1.0 ha, perimeter = 400 m.
    # (Using a large square keeps the ~1px contour-tracing boundary effect well under 5% relative error.)
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[50:250, 50:250] = 1  # 200x200 px square

    gsd = 0.5
    transform = GeoTransform.identity_with_gsd(gsd=gsd, origin_x=500_000.0, origin_y=2_700_000.0)

    polygons = mask_to_geo_polygons(mask, transform, min_area_px=16, simplify_epsilon_px=0.0)
    assert len(polygons) == 1, f"expected exactly one field polygon, got {len(polygons)}"

    p = polygons[0]
    expected_area_sqm = 200 * 200 * (gsd ** 2)   # 10,000.0
    expected_perimeter_m = 4 * 200 * gsd          # 400.0

    # cv2 contour tracing is pixel-accurate but not sub-pixel-exact (boundary
    # tracing convention can be off by up to ~1 px per side) — allow a small
    # tolerance rather than demanding bit-exact equality.
    assert abs(p.area_sqm - expected_area_sqm) / expected_area_sqm < 0.05, (
        f"area off by more than 5%: got {p.area_sqm}, expected ~{expected_area_sqm}"
    )
    assert abs(p.perimeter_m - expected_perimeter_m) / expected_perimeter_m < 0.05, (
        f"perimeter off by more than 5%: got {p.perimeter_m}, expected ~{expected_perimeter_m}"
    )
    print(f"mask_to_geo_polygons: area={p.area_sqm:.2f} sqm (~{expected_area_sqm}), "
          f"perimeter={p.perimeter_m:.2f} m (~{expected_perimeter_m}), area_ha={p.area_sqm / 10_000:.4f}")


def test_mask_to_results_via_segmentation_model():
    """Exercises FieldBoundarySegmenter.mask_to_results — the real postprocessing
    math wired through the model class, independent of predict() (which still
    correctly refuses to run without trained weights)."""
    metadata = ModelMetadata(
        model_id="field_boundary_segmentation", name="Field Agriculture Boundary Detection",
        model_type="segmentation", version="test", supported_gsd=GsdRange(0.25, 1.0),
        supported_image_sources=["test"], supported_formats=["array"],
        status=ModelStatus.NOT_PRODUCTION_READY, hardware_compatibility=["cpu"],
        default_confidence_threshold=0.5, class_names={0: "field"},
    )
    model = FieldBoundarySegmenter(metadata)

    mask = np.zeros((150, 150), dtype=np.uint8)
    mask[25:125, 25:125] = 1  # 100x100 px square — large enough that 1px boundary-tracing error is <2%
    transform = GeoTransform.identity_with_gsd(gsd=1.0)  # 1 m/px for simple numbers

    results = model.mask_to_results(mask, transform)
    assert len(results) == 1
    assert results[0]["class_name"] == "field"
    assert abs(results[0]["area_sqm"] - 10_000.0) / 10_000.0 < 0.05
    print(f"FieldBoundarySegmenter.mask_to_results: {results[0]['area_sqm']:.1f} sqm "
          f"({results[0]['area_sqm'] / 10_000:.4f} ha), perimeter={results[0]['perimeter_m']:.1f} m")

    # predict() must still refuse — no trained weights exist.
    try:
        model.predict(np.zeros((10, 10, 3), dtype=np.uint8))
        raise AssertionError("predict() should have raised NotProductionReadyError")
    except Exception as exc:
        assert "not production-ready" in str(exc).lower() or "no trained weights" in str(exc).lower()
        print("FieldBoundarySegmenter.predict() correctly still refuses to run:", exc)


if __name__ == "__main__":
    test_shoelace_area_known_square()
    test_mask_to_geo_polygons_known_field()
    test_mask_to_results_via_segmentation_model()
    print("test_measurements_offline: PASSED")
