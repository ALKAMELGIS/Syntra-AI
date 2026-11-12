"""
Offline tests for the pure-NumPy geometry/augmentation helpers ported from
the GeoSyntra AI QGIS plugin (app.training.geometry / app.training.augment).
No rasterio/PIL/torch needed — these modules are plain NumPy.
"""
import numpy as np

from app.training.augment import ALL_VARIANTS, augment_image, augment_points
from app.training.geometry import (
    clip_polygon_rect,
    min_area_rect,
    polygon_area,
    simplify_rdp,
    tile_starts,
    tiles_for_bbox,
)


def test_tile_starts_covers_length_without_padding():
    starts = tile_starts(1000, 300, 50)
    assert starts[0] == 0
    assert starts[-1] + 300 == 1000               # last tile ends exactly at the edge
    assert all(b - a <= 300 - 50 for a, b in zip(starts, starts[1:]))  # overlap respected


def test_tile_starts_small_raster_single_tile():
    starts = tile_starts(200, 640, 64)
    assert list(starts) == [0]


def test_tiles_for_bbox_finds_intersecting_tiles():
    starts = tile_starts(2000, 500, 100)
    hit = list(tiles_for_bbox(starts, 500, 450, 550))
    assert len(hit) >= 1
    for i in hit:
        assert starts[i] <= 550 and starts[i] + 500 >= 450


def test_polygon_area_square():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert polygon_area(sq) == 100.0


def test_clip_polygon_rect_keeps_only_intersection():
    sq = np.array([[-5, -5], [15, -5], [15, 15], [-5, 15]])
    clipped = clip_polygon_rect(sq, 0, 0, 10, 10)
    assert polygon_area(clipped) == 100.0          # clipped down to the 10x10 tile


def test_clip_polygon_rect_outside_returns_empty():
    sq = np.array([[100, 100], [110, 100], [110, 110], [100, 110]])
    clipped = clip_polygon_rect(sq, 0, 0, 10, 10)
    assert len(clipped) == 0


def test_min_area_rect_axis_aligned_box_is_itself():
    box = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float)
    rect = min_area_rect(box)
    assert polygon_area(rect) == 50.0


def test_simplify_rdp_keeps_at_least_three_points():
    ring = np.array([[0, 0], [1, 0.01], [2, -0.01], [10, 0], [10, 10], [0, 10]], dtype=float)
    out = simplify_rdp(ring, eps=0.5)
    assert len(out) >= 3


def test_augment_round_trip_orig_is_identity():
    img = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    assert np.array_equal(augment_image(img, "orig"), img)
    pts = np.array([[0.2, 0.3], [0.8, 0.9]])
    assert np.allclose(augment_points(pts, "orig"), pts)


def test_augment_rot90_point_mapping():
    # a point at the top-right corner (1, 0) rotated 90 CW lands top-left-ish (0, 0)
    pts = np.array([[1.0, 0.0]])
    out = augment_points(pts, "rot90")
    assert np.allclose(out, [[1.0, 1.0]])


def test_augment_fliplr_mirrors_x_only():
    pts = np.array([[0.2, 0.7]])
    out = augment_points(pts, "fliplr")
    assert np.allclose(out, [[0.8, 0.7]])


def test_all_variants_are_applicable_without_error():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    pts = np.array([[0.1, 0.9], [0.9, 0.1]])
    for op in ALL_VARIANTS:
        out_img = augment_image(img, op)
        out_pts = augment_points(pts, op)
        assert out_img.shape == (8, 8, 3)
        assert out_pts.shape == (2, 2)
