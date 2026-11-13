"""Offline tests for the DOTA-pretrained oriented-box detector wrapper and
rotated-polygon georeferencing. No torch/ultralytics needed: the network's
output is faked with plain numpy arrays shaped like ultralytics' OBB results."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.engine.geo import GeoTransform
from app.engine.postprocess import tile_detections_to_global
from app.models.aerial_obb_detector import AerialObbDetector
from app.models.base import RawDetection
from app.models.registry import get_metadata

DOTA_NAMES = {
    0: "plane", 1: "ship", 2: "storage tank", 3: "baseball diamond", 4: "tennis court",
    5: "basketball court", 6: "ground track field", 7: "harbor", 8: "bridge",
    9: "large vehicle", 10: "small vehicle", 11: "helicopter", 12: "roundabout",
    13: "soccer ball field", 14: "swimming pool",
}


class FakeYolo:
    task = "obb"
    names = DOTA_NAMES


def _detector(model_id: str) -> AerialObbDetector:
    det = AerialObbDetector(get_metadata(model_id), weights="fake-obb.pt")
    det._model = FakeYolo()
    det._bind_classes()
    return det


def _obb_result(rows):
    """rows: (cls, conf, quad[[x,y]*4])"""
    quads = np.array([r[2] for r in rows], dtype=float)
    env = np.stack([quads[..., 0].min(1), quads[..., 1].min(1), quads[..., 0].max(1), quads[..., 1].max(1)], axis=1)
    obb = SimpleNamespace(
        xyxyxyxy=quads, xyxy=env,
        cls=np.array([r[0] for r in rows], dtype=float),
        conf=np.array([r[1] for r in rows], dtype=float),
    )
    return SimpleNamespace(obb=_Sized(obb, len(rows)))


class _Sized:
    def __init__(self, inner, n):
        self.__dict__.update(inner.__dict__)
        self._n = n

    def __len__(self):
        return self._n


SQUARE = [[10, 10], [30, 10], [30, 30], [10, 30]]
DIAMOND = [[50, 40], [60, 50], [50, 60], [40, 50]]


def test_car_model_maps_small_and_large_vehicles_and_drops_others():
    det = _detector("car_detection")
    out = det.detections_from_results([_obb_result([
        (10, 0.9, SQUARE),   # small vehicle -> car (0)
        (9, 0.8, DIAMOND),   # large vehicle -> large_vehicle (1)
        (1, 0.95, SQUARE),   # ship: not asked for by the car model
    ])])
    assert [d.class_id for d in out] == [0, 1]
    assert out[0].score == 0.9
    assert out[1].polygon == [(50.0, 40.0), (60.0, 50.0), (50.0, 60.0), (40.0, 50.0)]
    assert out[1].box == (40.0, 40.0, 60.0, 60.0)  # envelope of the diamond


def test_oil_tank_model_uses_storage_tank_class():
    det = _detector("oil_tank_detection")
    assert det._dota_to_ours == {2: 0}
    out = det.detections_from_results([_obb_result([(2, 0.7, SQUARE), (8, 0.9, SQUARE)])])
    assert len(out) == 1 and out[0].class_id == 0


def test_validate_model_reports_missing_class_and_wrong_task():
    det = _detector("bridge_detection")
    assert det.validate_model()[0] is True

    class NoBridge(FakeYolo):
        names = {k: v for k, v in DOTA_NAMES.items() if v != "bridge"}

    det._model = NoBridge()
    ok, msg = det.validate_model()
    assert not ok and "bridge" in msg

    class Detect(FakeYolo):
        task = "detect"

    det._model = Detect()
    ok, msg = det.validate_model()
    assert not ok and "obb" in msg


def test_empty_results_give_no_detections():
    det = _detector("ship_detection")
    assert det.detections_from_results([SimpleNamespace(obb=None)]) == []
    assert det.detections_from_results([SimpleNamespace(obb=_Sized(SimpleNamespace(), 0))]) == []


def test_rotated_polygon_is_exported_and_envelope_used_for_nms():
    # north-up transform: 0.5 m pixels, origin (500000, 2700000)
    t = GeoTransform(a=0.5, b=0.0, c=500000.0, d=0.0, e=-0.5, f=2700000.0)
    raw = RawDetection(box=(40, 40, 60, 60), score=0.9, class_id=0,
                       polygon=[(50, 40), (60, 50), (50, 60), (40, 50)])
    [geo] = tile_detections_to_global(
        [{"box": raw.box, "score": raw.score, "class_id": raw.class_id, "polygon": raw.polygon}],
        t, {0: "ship"},
    )
    ring = geo["polygon_geo"]
    assert len(ring) == 5 and ring[0] == ring[-1]          # closed ring of the 4 rotated corners
    assert ring[0] == (500025.0, 2699980.0)                 # (50, 40) px
    assert geo["centroid_geo"] == (500025.0, 2699975.0)     # (50, 50) px
    xmin, ymin, xmax, ymax = geo["box_geo"]
    assert xmin < xmax and ymin < ymax                       # valid envelope under negative e
    assert (xmin, ymin, xmax, ymax) == (500020.0, 2699970.0, 500030.0, 2699980.0)


def test_axis_aligned_detections_unchanged_without_polygon():
    t = GeoTransform(a=1.0, b=0.0, c=0.0, d=0.0, e=-1.0, f=100.0)
    [geo] = tile_detections_to_global([{"box": (0, 0, 10, 10), "score": 0.5, "class_id": 0}], t, {0: "car"})
    assert len(geo["polygon_geo"]) == 5
    assert geo["box_geo"] == (0.0, 90.0, 10.0, 100.0)
