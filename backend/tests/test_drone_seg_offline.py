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
