"""YOLO-seg ONNX decoding, the drone pack wiring, and the drone-model scale guard."""
import numpy as np
import pytest

pytest.importorskip("cv2")


def _fake_outputs(nc=6, imgsz=640, box=(320, 320, 200, 100), cls=0, score=0.9):
    n = 8
    out = np.zeros((4 + nc + 32, n), np.float32)
    out[:4, 0] = box
    out[4 + cls, 0] = score
    out[4 + nc, 0] = 8.0            # mask = sigmoid(8 * proto0) ~ 1 everywhere
    protos = np.zeros((32, imgsz // 4, imgsz // 4), np.float32)
    protos[0] = 1.0
    return out, protos


def test_decode_seg_returns_box_shaped_outline_in_original_pixels():
    from app.models.onnx_seg import decode_seg

    # a 640x427 image letterboxed into 640x640: ratio 1, padded 106 px top
    out, protos = _fake_outputs(box=(320, 320, 200, 100))
    dets = decode_seg(out, protos, 6, ratio=1.0, left=0, top=106, orig_hw=(427, 640), imgsz=640)
    assert len(dets) == 1
    d = dets[0]
    xs, ys = [p[0] for p in d.polygon], [p[1] for p in d.polygon]
    assert abs(min(xs) - 220) <= 4 and abs(max(xs) - 420) <= 4
    assert abs(min(ys) - (270 - 106)) <= 4 and abs(max(ys) - (370 - 106)) <= 4
    assert d.cls == 0 and d.score == pytest.approx(0.9)


def test_decode_seg_drops_low_scores():
    from app.models.onnx_seg import decode_seg

    out, protos = _fake_outputs(score=0.1)
    assert decode_seg(out, protos, 6, 1.0, 0, 0, (640, 640), 640, conf=0.25) == []


def test_drone_pack_is_listed_and_not_installed_by_default():
    from app.models.packs import pack_for_model
    from app.models.registry import get_metadata

    pack = pack_for_model("drone_objects")
    assert pack is not None and pack.runtime == "yolo_seg" and pack.commercial_use is False
    assert get_metadata("drone_objects").status.value in ("not_installed", "validation")


def test_tile_size_follows_image_resolution():
    from app.models.drone_seg_detector import DroneSegDetector
    from app.models.registry import get_metadata

    d = DroneSegDetector(get_metadata("drone_objects"), "unused.onnx", training_gsd_m=0.04)
    assert d.preferred_tile_size(0.04) == 640
    assert d.preferred_tile_size(0.02) == 1280      # finer image: read more pixels, shrink to 640
    assert d.preferred_tile_size(0.08) == 320


def test_drone_model_refused_on_satellite_scale(monkeypatch):
    from fastapi import HTTPException

    import app.engine.segmentation_run as seg
    from app.api.predict import _refuse_wrong_scale
    from app.models.registry import get_metadata

    meta = get_metadata("drone_objects")
    monkeypatch.setattr(seg, "ground_gsd_m", lambda p: 0.5)
    with pytest.raises(HTTPException) as e:
        _refuse_wrong_scale(meta, "x.tif")
    assert e.value.status_code == 409 and "drone model" in e.value.detail
    monkeypatch.setattr(seg, "ground_gsd_m", lambda p: 0.03)
    _refuse_wrong_scale(meta, "x.tif")                # drone scale: allowed
    _refuse_wrong_scale(get_metadata("ship_detection"), "x.tif")


def test_weak_classes_in_the_file_are_not_returned(monkeypatch):
    """The ONNX file has six classes; the catalog keeps four (fence, bicycle scored mAP50 < 0.2)."""
    import app.models.drone_seg_detector as mod
    from app.models.onnx_seg import SegDetection
    from app.models.registry import get_metadata

    class Fake:
        imgsz = 640
        names = {0: "tree", 1: "building", 2: "vehicle", 3: "person", 4: "fence", 5: "bicycle"}

        def predict(self, img, conf=0.25, nominal_tile=None):
            sq = [(0, 0), (10, 0), (10, 10)]
            return [SegDetection((0, 0, 10, 10), sq, c, 0.9) for c in range(6)]

    monkeypatch.setattr(mod, "_load", lambda path, device: Fake())
    d = mod.DroneSegDetector(get_metadata("drone_objects"), "x.onnx")
    d.load_model("cpu")
    got = sorted(r.class_id for r in d.predict(np.zeros((64, 64, 3), np.uint8)))
    assert got == [0, 1, 2, 3]


def test_partial_tiles_are_scaled_like_full_tiles():
    """A 320x200 edge tile of a 640 px tiling must keep scale 1, not be enlarged 2x to fill the input."""
    from app.models.onnx_seg import _letterbox

    img = np.zeros((200, 320, 3), np.uint8)
    _, r, _ = _letterbox(img, 640, nominal_tile=640)
    assert r == 1.0
    _, r, _ = _letterbox(img, 640, nominal_tile=1280)      # 2 cm image read at 1280 px: shrink by 2
    assert r == 0.5
    _, r_old, _ = _letterbox(img, 640)                     # no tiling known: fit (old behaviour)
    assert r_old == 2.0


def test_obb_partial_tile_not_enlarged():
    pytest.importorskip("onnxruntime")
    from app.models.onnx_obb import OnnxObbModel

    m = OnnxObbModel.__new__(OnnxObbModel)
    m.imgsz = 1024
    _, r, _ = m._letterbox(np.zeros((432, 576, 3), np.uint8), nominal_tile=1024)
    assert r == 1.0
    _, r, _ = m._letterbox(np.zeros((432, 576, 3), np.uint8))
    assert r == 1.0


def test_geodeep_partial_tile_padded_to_full_tile():
    from app.models.geodeep_onnx import _to_input

    _, s, _ = _to_input(np.zeros((100, 300, 3), np.uint8), 512, nominal_tile=1024)
    assert s == 2.0                       # same factor as a full 1024 tile
