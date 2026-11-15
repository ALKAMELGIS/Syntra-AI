"""Outline geometry: masks and rough rings -> the shape the object really has."""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
shapely = pytest.importorskip("shapely")
from shapely import affinity  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402

from app.engine.shapes import (  # noqa: E402
    RefineOptions, iou, mask_to_polygon, refine, right_angles, shape_for_class,
)


def raster(poly, size=400):
    """Burn a polygon into a mask with a ragged 1 px edge, like a model's mask."""
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly.exterior.coords)).astype(np.int32)], 1)
    rng = np.random.default_rng(0)
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    m[edge & (rng.random(m.shape) < 0.35)] ^= 1
    return m


def corner_angles(poly):
    pts = np.asarray(poly.exterior.coords)[:-1]
    out = []
    for i in range(len(pts)):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
        v1, v2 = a - b, c - b
        out.append(math.degrees(math.acos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))
    return out


L_SHAPE = Polygon([(100, 100), (300, 100), (300, 180), (200, 180), (200, 300), (100, 300)])


def test_rotated_building_gets_right_angles_and_keeps_its_shape():
    truth = affinity.rotate(L_SHAPE, 23, origin="centroid")
    traced = mask_to_polygon(raster(truth))
    assert len(traced.exterior.coords) > 100                      # the ragged trace
    res = refine(traced, RefineOptions(shape="right_angles", px_m=0.3))
    out = res["polygon"]
    assert res["shape"] == "right_angles" and res["warning"] is None
    assert res["vertices"] == 6
    assert all(abs(a - 90) < 0.5 for a in corner_angles(out))
    assert iou(out, truth) > 0.96
    assert res["area_m2"] == pytest.approx(truth.area * 0.09, rel=0.05)


def test_rectangle_and_circle_fitters():
    rect = affinity.rotate(box(120, 150, 280, 230), -35, origin="centroid")
    res = refine(mask_to_polygon(raster(rect)), RefineOptions(shape="rectangle"))
    assert res["vertices"] == 4 and iou(res["polygon"], rect) > 0.95

    disc = Polygon([(200 + 70 * math.cos(t), 200 + 70 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 200)])
    res = refine(mask_to_polygon(raster(disc)), RefineOptions(shape="circle"))
    assert res["shape"] == "circle" and res["vertices"] == 64 and iou(res["polygon"], disc) > 0.96


def test_a_fit_that_changes_the_shape_too_much_is_refused():
    disc = Polygon([(200 + 70 * math.cos(t), 200 + 70 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 200)])
    res = refine(mask_to_polygon(raster(disc)), RefineOptions(shape="rectangle"))
    assert res["shape"] == "polygon" and "Kept the traced outline" in res["warning"]


def test_smooth_outline_for_trees_stays_close():
    blob = Polygon([(200 + (60 + 12 * math.sin(5 * t)) * math.cos(t), 200 + (60 + 12 * math.sin(5 * t)) * math.sin(t))
                    for t in np.linspace(0, 2 * math.pi, 300)])
    res = refine(mask_to_polygon(raster(blob)), RefineOptions(shape="smooth", px_m=0.3))
    assert res["shape"] == "smooth" and iou(res["polygon"], blob) > 0.95


def test_sizes_are_in_metres():
    sq = box(100, 100, 200, 200)                                    # 100 px at 0.5 m = 50 m
    grown = refine(sq, RefineOptions(shape="right_angles", px_m=0.5, grow_m=1.0))["polygon"]
    assert grown.bounds == pytest.approx((98, 98, 202, 202), abs=0.01)   # 1 m = 2 px, corners stay square
    with pytest.raises(ValueError, match="minimum size"):
        refine(sq, RefineOptions(px_m=0.5, min_area_m2=5000))


def test_mask_to_polygon_keeps_the_part_under_the_click_and_maps_back():
    m = np.zeros((100, 100), np.uint8)
    m[10:30, 10:30] = 1          # small, clicked
    m[50:95, 50:95] = 1          # bigger, not clicked
    p = mask_to_polygon(m, seed_xy=(20, 20), offset=(1000, 2000), scale=2.0)
    minx, miny, maxx, maxy = p.bounds
    assert (minx, miny) == pytest.approx((1020, 2020), abs=1) and (maxx, maxy) == pytest.approx((1060, 2060), abs=1)


def test_right_angles_on_a_noisy_rectangle_is_a_rectangle():
    rect = affinity.rotate(box(0, 0, 120, 60), 12)
    out = right_angles(mask_to_polygon(raster(affinity.translate(rect, 150, 150))))
    assert len(out.exterior.coords) - 1 == 4


def test_default_shape_follows_the_class():
    assert shape_for_class("Building") == "right_angles"
    assert shape_for_class("solar_panel") == "rectangle"
    assert shape_for_class("oil tank") == "circle"
    assert shape_for_class("palm") == "smooth"
    assert shape_for_class("ship") == "polygon"


def test_a_label_outside_the_image_is_refused(tmp_path):
    from app.annotations import store

    b = {"west": 55.0, "east": 55.01, "south": 25.0, "north": 25.01}
    with pytest.raises(ValueError, match="outside the image"):
        store.add_annotation(tmp_path / "x.tif", "building", [[54.99, 25.005], [54.995, 25.005], [54.995, 25.004]], b, 100, 100)
    ok = store.add_annotation(tmp_path / "x.tif", "building", [[54.99, 25.005], [55.005, 25.005], [55.005, 25.004]], b, 100, 100,
                              extra={"tool": "rect", "shape": "rectangle"})
    assert ok["tool"] == "rect" and ok["shape"] == "rectangle"
