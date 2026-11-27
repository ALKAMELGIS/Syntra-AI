"""Date palm pack: the network matches the published SegFormer-B3 file, crowns split per tree."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("scipy")
cv2 = pytest.importorskip("cv2")

# read from the published file's header (brakuta/date-palm-wv3-models, segformer_b3_rgb.safetensors)
PUBLISHED_KEYS = 548
PUBLISHED_SHAPES = {
    "backbone.layers.0.0.projection.weight": (64, 3, 7, 7),
    "backbone.layers.0.1.2.attn.sr.weight": (64, 64, 8, 8),
    "backbone.layers.0.1.0.ffn.layers.1.weight": (256, 1, 3, 3),
    "backbone.layers.1.1.3.attn.attn.in_proj_weight": (384, 128),
    "backbone.layers.2.1.17.attn.sr.weight": (320, 320, 2, 2),
    "backbone.layers.2.1.17.ffn.layers.4.weight": (320, 1280, 1, 1),
    "backbone.layers.2.2.weight": (320,),
    "backbone.layers.3.0.projection.weight": (512, 320, 3, 3),
    "backbone.layers.3.1.2.attn.attn.out_proj.weight": (512, 512),
    "decode_head.convs.2.conv.weight": (256, 320, 1, 1),
    "decode_head.fusion_conv.conv.weight": (256, 1024, 1, 1),
    "decode_head.fusion_conv.bn.running_var": (256,),
    "decode_head.conv_seg.weight": (150, 256, 1, 1),
}
ABSENT = ("backbone.layers.3.1.0.attn.sr.weight", "backbone.layers.3.1.0.attn.norm.weight",
          "decode_head.convs.0.conv.bias")


def test_network_has_the_published_tensors():
    from app.models.palm_segformer import build_network

    sd = build_network(150).state_dict()
    assert len(sd) == PUBLISHED_KEYS
    for k, shape in PUBLISHED_SHAPES.items():
        assert tuple(sd[k].shape) == shape, k
    for k in ABSENT:
        assert k not in sd


def test_strict_load_and_refusals(tmp_path):
    from safetensors.torch import save_file

    from app.models.palm_segformer import build_network, palm_probability, verify_weights, load_network

    sd = {k: v.contiguous() for k, v in build_network(150).state_dict().items()}
    good = tmp_path / "segformer_b3_rgb.safetensors"
    save_file(sd, str(good))
    info = verify_weights(good)
    assert info["class_names"] == ["background", "date_palm"] and info["tile_px"] == 512
    prob = palm_probability(load_network(str(good)), np.full((300, 200, 3), 120, np.uint8))
    assert prob.shape == (300, 200) and 0 <= prob.min() <= prob.max() <= 1

    missing = dict(sd); missing.pop("decode_head.fusion_conv.bn.running_mean")
    save_file(missing, str(tmp_path / "m.safetensors"))
    with pytest.raises(RuntimeError):
        verify_weights(tmp_path / "m.safetensors")
    ms = dict(sd); ms["backbone.layers.0.0.projection.weight"] = torch.zeros(64, 8, 7, 7)
    save_file(ms, str(tmp_path / "ms.safetensors"))
    with pytest.raises(ValueError, match="RGB"):
        verify_weights(tmp_path / "ms.safetensors")


def test_touching_crowns_split_one_per_tree():
    from app.models.palm_segformer import split_crowns

    prob = np.zeros((200, 260), np.float32)
    for cx in (60, 90, 120):                      # three 9 m crowns (r = 14.5 px at 0.31 m) touching in a row
        cv2.circle(prob, (cx, 100), 15, 0.9, -1)
    cv2.circle(prob, (220, 40), 2, 0.9, -1)       # a 1.2 m blob: too small for a palm
    crowns = split_crowns(prob, 0.31)
    assert len(crowns) == 3
    xs = sorted(c[:, 0].mean() for c, _ in crowns)
    assert xs[0] < 75 < xs[1] < 105 < xs[2]
    assert all(abs(s - 0.9) < 1e-5 for _, s in crowns)


def test_detector_reads_at_training_scale(monkeypatch):
    from app.models import palm_segformer as ps
    from app.models.registry import get_metadata

    seen = {}

    def fake_prob(net, rgb):
        seen["shape"] = rgb.shape
        p = np.zeros(rgb.shape[:2], np.float32)
        cv2.circle(p, (256, 256), 20, 0.8, -1)
        return p

    monkeypatch.setattr(ps, "palm_probability", fake_prob)
    d = ps.PalmSegformerDetector(get_metadata("date_tree_detection"), "x")
    d._net = object()
    assert d.preferred_tile_size(0.5) == 317 and d.preferred_tile_size(0.31) == 512
    d.nominal_tile = 317
    out = d.predict(np.zeros((317, 317, 3), np.uint8))
    assert seen["shape"] == (512, 512, 3) and len(out) == 1
    x0, y0, x1, y1 = out[0].box                    # 40 model px wide -> about 25 native px
    assert 20 < x1 - x0 < 30 and abs((x0 + x1) / 2 - 158.5) < 3
    assert d.predict(np.zeros((317, 317, 3), np.uint8), confidence_threshold=0.9) == []


def test_pack_checks_sha_and_file_type(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.models.packs import PACKS, check_sha256

    pack = PACKS["date-palm-wv3"]
    f = tmp_path / "x.safetensors"
    f.write_bytes(b"not the model")
    with pytest.raises(ValueError, match="SHA-256"):
        check_sha256(f, pack)
    monkeypatch.setattr(settings, "WEIGHTS_DIR", tmp_path / "w")
    from app.main import app

    c = TestClient(app)
    r = c.post("/api/models/packs/date-palm-wv3/file", files={"file": ("m.onnx", b"x", "application/octet-stream")})
    assert r.status_code == 400 and ".safetensors" in r.json()["detail"]
    r = c.post("/api/models/packs/date-palm-wv3/file", files={"file": ("segformer_b3_rgb.safetensors", b"x", "application/octet-stream")})
    assert r.status_code == 400 and "SHA-256" in r.json()["detail"]
