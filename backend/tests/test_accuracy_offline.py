"""Accuracy areas: matching predictions to labels, and areas kept out of training."""
import pytest

pytest.importorskip("shapely")


def sq(x, y, s=1.0):
    return [[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]


def test_match_counts_tp_fp_fn_per_class():
    from app.engine.accuracy import _poly, match

    truth = [{"class": "palm", "geom": _poly(sq(0, 0))}, {"class": "palm", "geom": _poly(sq(5, 0))},
             {"class": "car", "geom": _poly(sq(10, 0))}]
    preds = [{"class": "palm", "score": 0.9, "geom": _poly(sq(0.1, 0))},     # hit
             {"class": "palm", "score": 0.8, "geom": _poly(sq(0.2, 0))},     # duplicate of the same palm: FP
             {"class": "palm", "score": 0.7, "geom": _poly(sq(20, 0))}]      # nothing there: FP
    r = match(preds, truth, 0.5)
    assert r["per_class"]["palm"] == {"tp": 1, "fp": 2, "fn": 1, "labels": 2, "predictions": 3,
                                      "precision": 0.333, "recall": 0.5, "f1": 0.4}
    assert r["per_class"]["car"]["fn"] == 1 and r["per_class"]["car"]["recall"] == 0.0
    assert r["overall"]["tp"] == 1 and r["overall"]["fn"] == 2


def test_one_class_model_matches_one_class_labels_by_position():
    from app.engine.accuracy import class_aliases

    assert class_aliases(["ship"], ["boat"]) == {"ship": "boat"}
    assert class_aliases(["tree", "building"], ["Tree"]) == {"tree": "Tree"}
    assert class_aliases(["car", "ship"], ["boat"]) == {}


def test_evaluate_only_scores_inside_the_accuracy_area():
    from app.engine.accuracy import evaluate

    area = {"class_name": "__check_area__", "points_lonlat": sq(0, 0, 10)}
    labels = [{"class_name": "boat", "points_lonlat": sq(1, 1)}, {"class_name": "boat", "points_lonlat": sq(20, 20)}]
    gj = {"features": [
        {"geometry": {"type": "Polygon", "coordinates": [sq(1, 1)]}, "properties": {"class": "ship", "score": 0.9}},
        {"geometry": {"type": "Polygon", "coordinates": [sq(20, 20)]}, "properties": {"class": "ship", "score": 0.9}},
    ]}
    r = evaluate(gj, labels + [area], [area], ["ship"])
    assert r["overall"] == {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_labels_inside_accuracy_areas_are_never_trained_on():
    from app.annotations import store

    anns = [
        {"class_name": "__check_area__", "points_px": sq(0, 0, 100)},
        {"class_name": "__train_area__", "points_px": sq(200, 0, 100)},
        {"class_name": "palm", "points_px": sq(10, 10, 5)},       # inside the accuracy area
        {"class_name": "palm", "points_px": sq(210, 10, 5)},
    ]
    names, objs = store.to_export_payload(anns)
    assert names == ["palm"] and len(objs) == 1 and objs[0]["points_px"][0] == [210, 10]
    assert store.class_names_used(anns) == ["palm"]


def test_area_filter_masks_outside_training_and_inside_accuracy_areas():
    pytest.importorskip("cv2")
    from app.api.training import area_filter

    f, bounds = area_filter([sq(0, 0, 64)], [sq(0, 0, 16)], 64, 1000, 1000)
    assert bounds == (0, 0, 64, 64)
    m = f(0, 0)
    assert m is not None and m is not False
    assert not m[5, 5] and m[40, 40]          # accuracy area cut out, training area kept
    assert f(500, 500) is False               # far outside the training area
    f2, b2 = area_filter(None, None, 64, 100, 100)
    assert f2 is None and b2 is None


def test_area_window_is_aligned_to_the_whole_image_tile_grid(tmp_path):
    """An area run must read the same tiles a whole-image run reads, so results inside the area match."""
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    import numpy as np
    from rasterio.transform import from_origin

    from app.engine.aoi import prepare_aoi
    from app.engine.raster_io import _window_grid

    p = tmp_path / "img.tif"
    tf = from_origin(330000, 2773000, 0.3, 0.3)
    with rasterio.open(p, "w", driver="GTiff", width=3000, height=2000, count=3, dtype="uint8", crs="EPSG:32640", transform=tf) as d:
        d.write(np.zeros((3, 2000, 3000), np.uint8))
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:32640", "EPSG:4326", always_xy=True)
    x0, y0 = t.transform(330000 + 0.3 * 1100, 2773000 - 0.3 * 700)
    x1, y1 = t.transform(330000 + 0.3 * 1500, 2773000 - 0.3 * 1000)
    aoi = {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
    a = prepare_aoi(str(p), aoi, grid=(1024, 0.25))
    c0, r0, w, h = a.window
    stride = 768
    assert c0 % stride == 0 and r0 % stride == 0
    full = {(c, r) for c, r, _, _ in _window_grid(3000, 2000, 1024, 0.25)}
    with rasterio.open(p) as ds:
        mine = {(c + c0, r + r0) for c, r, ww, hh in _window_grid(w, h, 1024, 0.25)
                if a.tile_intersects(ds.transform, c + c0, r + r0, ww, hh)}
    assert mine and mine <= full          # every tile read is a tile of the whole-image grid
