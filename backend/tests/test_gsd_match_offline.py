"""Approved class models read tiles at their training scale when it is known."""
import numpy as np

from app.models.base import ModelMetadata
from app.models.custom_detector import CustomYoloDetector


def _det(tmp_path, gsd, imgsz=256):
    run = tmp_path / "v001"
    (run / "weights").mkdir(parents=True, exist_ok=True)
    (run / "weights" / "best.pt").write_bytes(b"x")
    (run / "args.yaml").write_text(f"task: detect\nimgsz: {imgsz}\n")
    meta = ModelMetadata.__new__(ModelMetadata)
    return CustomYoloDetector(meta, str(run / "weights" / "best.pt"), training_gsd_m=gsd)


def test_tile_scales_with_gsd(tmp_path):
    d = _det(tmp_path, 0.12)
    assert d.train_imgsz() == 256
    assert d.preferred_tile_size(0.12) == 256          # same scale: tile = training size
    assert d.preferred_tile_size(0.03) == 1024         # drone, 4x finer: 4x more pixels per tile
    assert d.preferred_tile_size(0.5) == 64            # coarser than trained: clamped


def test_unknown_gsd_keeps_request_tile(tmp_path):
    assert _det(tmp_path, None).preferred_tile_size(0.1) is None
    assert _det(tmp_path, 0.12).preferred_tile_size(None) is None


def test_edge_tile_padded_to_nominal(tmp_path):
    d = _det(tmp_path, 0.12)
    seen = {}

    class Fake:
        def predict(self, source, **kw):
            seen["shape"] = source.shape
            return []
    d._model = Fake()
    d.metadata = type("M", (), {"default_confidence_threshold": 0.25})()
    d.nominal_tile = 512
    d.predict(np.zeros((300, 512, 3), np.uint8))
    assert seen["shape"] == (512, 512, 3)
