"""
Runtime for ONNX models in the GeoDeep / Deepness format.

These files describe themselves in ONNX metadata: `model_type` (Detector or
Segmentor), `det_type` (retinanet, YOLO_v5_or_v7_default, YOLO_v8,
YOLO_v9), the ground resolution the model was trained at (`resolution`,
cm/px), thresholds and class names. Input is one RGB tile, NCHW float 0-1.

Written from that documented format (not vendored), so it runs with only
onnxruntime + numpy + OpenCV.

Resolution matters: a model trained at 10 cm sees a 50 cm tree as one fifth
of the size it learned. `tile_px_for_gsd()` picks the source window so that,
after resizing to the model's input size, objects appear at the scale the
model was trained on.
"""
from __future__ import annotations

import json
import threading

import numpy as np

from app.models.base import DetectionModel, ModelMetadata, RawDetection

_SESSIONS: dict[tuple[str, str], tuple[object, dict]] = {}
_LOCK = threading.Lock()


def load_session(path: str, device: str = "cpu"):
    import os

    st = os.stat(path)
    # size + mtime in the key: a re-installed or replaced file must not reuse the old model
    key = (path, device, st.st_size, st.st_mtime_ns)
    with _LOCK:
        if key in _SESSIONS:
            return _SESSIONS[key]
        import onnxruntime as ort

        from app.core.onnx_runtime import session_providers, trt_cache_dir
        providers = session_providers(device, trt_cache_dir())   # TensorRT > CUDA > DirectML > CPU
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
        inputs = sess.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Unsupported ONNX model: expected exactly one input.")
        meta = sess.get_modelmeta().custom_metadata_map

        def j(k, default):
            return json.loads(meta[k]) if k in meta else default

        shape = inputs[0].shape
        cfg = {
            "model_type": j("model_type", "Detector"),
            "det_type": j("det_type", "YOLO_v5_or_v7_default"),
            "det_conf": float(meta.get("det_conf", 0.3)),
            "det_iou_thresh": float(meta.get("det_iou_thresh", 0.5)),
            "seg_thresh": float(meta.get("seg_thresh", 0.5)),
            "resolution": float(meta.get("resolution", 10)),      # cm per pixel the model was trained at
            "class_names": {int(k): v for k, v in j("class_names", {}).items()},
            "tiles_size": int(shape[-1]) if isinstance(shape[-1], int) else 512,
            "input_name": inputs[0].name,
        }
        _SESSIONS[key] = (sess, cfg)
        return sess, cfg


def tile_px_for_gsd(cfg: dict, gsd_m: float | None) -> int:
    """Source pixels per tile so objects reach the model at its training scale."""
    size = cfg["tiles_size"]
    if not gsd_m or gsd_m <= 0:
        return size
    ratio = (cfg["resolution"] / 100.0) / gsd_m           # model_res / image_res
    return max(16, int(round(size * ratio)))


def _to_input(tile_rgb: np.ndarray, size: int, nominal_tile: int | None = None) -> tuple[np.ndarray, float, float]:
    import cv2

    h, w = tile_rgb.shape[:2]
    img = tile_rgb
    if (h, w) != (size, size):
        # pad partial edge tiles to the FULL tile's size, so they are resized by the
        # same factor as every other tile (not enlarged to fill the input)
        side = max(h, w, nominal_tile or 0)
        pad = np.zeros((side, side, 3), dtype=tile_rgb.dtype)
        pad[:h, :w] = tile_rgb
        img = cv2.resize(pad, (size, size), interpolation=cv2.INTER_CUBIC if side < size else cv2.INTER_AREA)
        scale = side / size
    else:
        scale = 1.0
    x = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), scale, scale


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        ov = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ov <= iou]
    return np.array(keep, dtype=int)


def decode_detections(outs: list[np.ndarray], cfg: dict, conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> boxes (N,4) xyxy in model-input pixels, scores (N,), classes (N,)"""
    t = cfg["det_type"]
    if t == "retinanet":
        boxes, scores = np.asarray(outs[0]).reshape(-1, 4), np.asarray(outs[1]).reshape(-1)
        classes = np.asarray(outs[2]).reshape(-1).astype(int) if len(outs) > 2 else np.zeros(len(scores), int)
    else:
        out = np.asarray(outs[0])
        if t in ("YOLO_v8", "YOLO_v9"):
            out = out.transpose(0, 2, 1)[0]            # (N, 4 + nc)
            cls_scores = out[:, 4:]
            classes = cls_scores.argmax(1)
            scores = cls_scores.max(1)
        else:                                          # YOLO v5/v7: x,y,w,h,obj,cls...
            out = out[0]
            cls_scores = out[:, 5:] if out.shape[1] > 5 else np.ones((len(out), 1))
            classes = cls_scores.argmax(1)
            scores = out[:, 4] * cls_scores.max(1) if out.shape[1] > 5 else out[:, 4]
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    keep = scores >= conf
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    k = _nms(boxes, scores, cfg["det_iou_thresh"])
    return boxes[k], scores[k], classes[k]


class GeoDeepDetector(DetectionModel):
    def __init__(self, metadata: ModelMetadata, weights_path: str):
        self.metadata = metadata
        self.weights_path = weights_path
        self._sess = None
        self._cfg: dict = {}

    def load_model(self, device: str) -> None:
        self._sess, self._cfg = load_session(self.weights_path, device)

    def validate_model(self) -> tuple[bool, str]:
        if self._sess is None:
            return False, "Model not loaded."
        if self._cfg["model_type"] != "Detector":
            return False, f"Expected a Detector, the file is a {self._cfg['model_type']}."
        return True, f"OK — {self._cfg['det_type']}, trained at {self._cfg['resolution']:g} cm/px."

    def preferred_tile_size(self, gsd_m: float | None) -> int | None:
        return tile_px_for_gsd(self._cfg or load_session(self.weights_path)[1], gsd_m)

    def predict(self, tile_rgb: np.ndarray, confidence_threshold: float | None = None) -> list[RawDetection]:
        size = self._cfg["tiles_size"]
        x, sx, sy = _to_input(tile_rgb, size, getattr(self, "nominal_tile", None))
        outs = self._sess.run(None, {self._cfg["input_name"]: x})
        conf = confidence_threshold if confidence_threshold is not None else self._cfg["det_conf"]
        boxes, scores, classes = decode_detections(outs, self._cfg, conf)
        h, w = tile_rgb.shape[:2]
        out = []
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
            x1, x2 = np.clip([x1 * sx, x2 * sx], 0, w)
            y1, y2 = np.clip([y1 * sy, y2 * sy], 0, h)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            cid = int(c) if int(c) in self.metadata.class_names else 0
            out.append(RawDetection(box=(float(x1), float(y1), float(x2), float(y2)), score=float(s), class_id=cid))
        return out

    def export_results(self, out_dir: str, detections_geo: list[dict], extra_meta: dict | None = None) -> dict:
        from app.engine.postprocess import export_results

        meta = {"model_version": self.metadata.version, "model_status": self.metadata.status.value,
                "weights": self.weights_path, "trained_at_cm": self._cfg.get("resolution"), **(extra_meta or {})}
        return export_results(out_dir, self.metadata.model_id, detections_geo, geometry="polygon_detection", extra_meta=meta)


def segment_tile(tile_rgb: np.ndarray, sess, cfg: dict, nominal_tile: int | None = None) -> np.ndarray:
    """-> (H, W) uint8 class mask for the tile (0 = background)."""
    import cv2

    h, w = tile_rgb.shape[:2]
    x, s, _ = _to_input(tile_rgb, cfg["tiles_size"], nominal_tile)
    out = np.asarray(sess.run(None, {cfg["input_name"]: x})[0])[0]     # (C, S, S)
    if out.shape[0] == 1:
        mask = (out[0] >= cfg["seg_thresh"]).astype(np.uint8)
    else:
        out = np.where(out >= cfg["seg_thresh"], out, 0)
        mask = out.argmax(0).astype(np.uint8)
    side = max(h, w, nominal_tile or 0)
    mask = cv2.resize(mask, (side, side), interpolation=cv2.INTER_NEAREST)
    return mask[:h, :w]
