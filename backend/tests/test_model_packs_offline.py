"""Model packs: install/upload/uninstall, catalog status, and the GeoDeep
ONNX runtime, using tiny synthetic ONNX models in the same metadata format."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper  # noqa: E402

from app.core.config import settings  # noqa: E402


def _meta(m, **kv):
    for k, v in kv.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v


def make_retinanet(path, size=64, boxes=((8, 8, 24, 24),), scores=(0.9,), res_cm=10):
    """Detector: ignores the image, returns fixed boxes (xyxy in input px)."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])
    outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in ("boxes", "scores", "labels")]
    nodes = [
        helper.make_node("Constant", [], ["boxes"], value=helper.make_tensor("b", TensorProto.FLOAT, [len(boxes), 4], [v for b in boxes for v in b])),
        helper.make_node("Constant", [], ["scores"], value=helper.make_tensor("s", TensorProto.FLOAT, [len(scores)], list(scores))),
        helper.make_node("Constant", [], ["labels"], value=helper.make_tensor("l", TensorProto.FLOAT, [len(scores)], [0.0] * len(scores))),
    ]
    g = helper.make_graph(nodes, "det", [inp], outs)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    _meta(m, model_type='"Detector"', det_type='"retinanet"', det_conf="0.3", resolution=str(res_cm),
          class_names='{"0": "tree"}')
    onnx.save(m, path)


def make_segmentor(path, size=64, res_cm=50):
    """Segmentor: class 1 where the red channel > 0.5, two output channels."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, size, size])
    nodes = [
        helper.make_node("Slice", ["input", "s", "e", "a"], ["red"]),
        helper.make_node("Sub", ["one", "red"], ["bg"]),
        helper.make_node("Concat", ["bg", "red"], ["out"], axis=1),
    ]
    inits = [helper.make_tensor("s", TensorProto.INT64, [1], [0]), helper.make_tensor("e", TensorProto.INT64, [1], [1]),
             helper.make_tensor("a", TensorProto.INT64, [1], [1]), helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])]
    g = helper.make_graph(nodes, "seg", [inp], [out], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    _meta(m, model_type='"Segmentor"', seg_thresh="0.5", resolution=str(res_cm), class_names='{"1": "building"}')
    onnx.save(m, path)


@pytest.fixture(autouse=True)
def iso(tmp_path, monkeypatch):
    from app.models import geodeep_onnx
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "data" / "results")
    monkeypatch.setattr(settings, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.delenv("SYNTRA_API_KEY", raising=False)
    geodeep_onnx._SESSIONS.clear()
    settings.ensure_dirs()


def test_catalog_status_follows_install(tmp_path):
    from app.models.packs import PACKS, pack_path
    from app.models.registry import get_metadata

    assert get_metadata("tree_detection").status.value == "not_installed"
    make_retinanet(str(pack_path(PACKS["deepforest-trees"])))
    m = get_metadata("tree_detection")
    assert m.status.value == "validation" and "MIT" in m.notes


def test_tile_scale_for_model_resolution(tmp_path):
    from app.models.geodeep_onnx import load_session, tile_px_for_gsd

    make_retinanet(str(tmp_path / "t.onnx"), size=400, res_cm=10)
    _, cfg = load_session(str(tmp_path / "t.onnx"))
    assert tile_px_for_gsd(cfg, 0.5) == 80         # 50 cm image: 80 px window upsampled to 400
    assert tile_px_for_gsd(cfg, 0.05) == 800       # 5 cm drone image: 800 px window downsampled
    assert tile_px_for_gsd(cfg, None) == 400


def test_detector_scales_boxes_back_to_tile(tmp_path):
    from app.models.geodeep_onnx import GeoDeepDetector
    from app.models.registry import get_metadata

    make_retinanet(str(tmp_path / "t.onnx"), size=64, boxes=((8, 8, 24, 24), (9, 9, 25, 25)), scores=(0.9, 0.8))
    det = GeoDeepDetector(get_metadata("tree_detection"), str(tmp_path / "t.onnx"))
    det.load_model("cpu")
    assert det.validate_model()[0]
    out = det.predict(np.zeros((16, 16, 3), dtype=np.uint8))      # 16 px tile upsampled 4x to 64
    assert len(out) == 1                                          # overlapping duplicate removed
    assert out[0].box == pytest.approx((2, 2, 6, 6))


def test_http_install_from_file_and_uninstall(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    good, bad = tmp_path / "good.onnx", tmp_path / "bad.onnx"
    make_segmentor(str(good))
    make_retinanet(str(bad))
    with TestClient(app) as c:
        packs = {p["pack_id"]: p for p in c.get("/api/models/packs").json()}
        assert packs["ramp-buildings"]["installed"] is False and packs["ramp-buildings"]["commercial_use"] is False
        r = c.post("/api/predict", json={"model_id": "building_detection", "image_path": str(good)})
        assert r.status_code in (404, 409)
        # wrong kind of model is refused
        r = c.post("/api/models/packs/ramp-buildings/file", files={"file": ("x.onnx", bad.read_bytes())})
        assert r.status_code == 400 and "Segmentor" in r.json()["detail"]
        r = c.post("/api/models/packs/ramp-buildings/file", files={"file": ("x.txt", b"hi")})
        assert r.status_code == 400
        r = c.post("/api/models/packs/ramp-buildings/file", files={"file": ("x.onnx", good.read_bytes())})
        assert r.status_code == 200 and r.json()["installed"] is True
        m = next(m for m in c.get("/api/models").json() if m["model_id"] == "building_detection")
        assert m["status"] == "validation" and m["pack"]["installed"] is True
        assert c.delete("/api/models/packs/ramp-buildings").json()["installed"] is False
        assert c.post("/api/models/packs/nope/install").status_code == 404


rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")


def test_segmentation_run_produces_wgs84_polygons_with_area(tmp_path):
    from rasterio.transform import from_origin

    from app.engine.segmentation_run import run_segmentation

    img = np.zeros((3, 300, 300), dtype=np.uint8)
    img[:] = 40
    img[0, 50:90, 60:120] = 230          # "building" A: 40x60 px = 20 x 30 m at 0.5 m
    img[0, 200:220, 200:220] = 230       # "building" B: 10 x 10 m
    path = tmp_path / "town.tif"
    with rasterio.open(path, "w", driver="GTiff", height=300, width=300, count=3, dtype="uint8",
                       crs="EPSG:32640", transform=from_origin(323000, 2774000, 0.5, 0.5)) as d:
        d.write(img)
    model = tmp_path / "seg.onnx"
    make_segmentor(str(model), size=64, res_cm=50)
    res = run_segmentation(str(path), tmp_path / "out", "building_detection", str(model), class_names={1: "building"})
    feats = json.loads((tmp_path / "out" / "building_detection.geojson").read_text())["features"]
    areas = sorted(f["properties"]["area_m2"] for f in feats)
    assert len(feats) == 2, areas
    assert areas[0] == pytest.approx(100, rel=0.15) and areas[1] == pytest.approx(600, rel=0.1)
    lon, lat = feats[0]["geometry"]["coordinates"][0][0]
    assert 55.2 < lon < 55.3 and 25.0 < lat < 25.1
    assert res["summary"]["tile_px"] == 64 and feats[0]["properties"]["class"] == "building"
