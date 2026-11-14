"""
Torch-free inference for YOLO instance-segmentation models exported to ONNX.

Same reason as onnx_obb.py: onnxruntime + numpy + OpenCV run on a GIS
workstation without PyTorch.

Output layout of an ultralytics -seg ONNX graph:
  output0 (1, 4 + nc + 32, N): cx, cy, w, h, class scores x nc, 32 mask coefficients
  output1 (1, 32, mh, mw):     mask prototypes at 1/4 of the input size
An instance mask is sigmoid(coefficients @ prototypes), cropped to its box.
Each mask is returned as its outer polygon in original image pixels.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np


@dataclass
class SegDetection:
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in original pixels
    polygon: list[tuple[float, float]]
    cls: int
    score: float


class OnnxSegModel:
    task = "segment"

    def __init__(self, path: str, providers: list[str] | None = None):
        import onnxruntime as ort  # lazy

        self.session = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.meta = meta

    def predict(self, image_rgb: np.ndarray, conf: float = 0.25, iou: float = 0.5,
                max_det: int = 300) -> list[SegDetection]:
        blob, r, (left, top) = _letterbox(image_rgb, self.imgsz)
        out0, protos = self.session.run(None, {self.input_name: blob})
        h, w = image_rgb.shape[:2]
        return decode_seg(out0[0], protos[0], len(self.names) or out0.shape[1] - 36, r, left, top, (h, w),
                          self.imgsz, conf=conf, iou=iou, max_det=max_det)


def _letterbox(img: np.ndarray, s: int):
    import cv2

    h, w = img.shape[:2]
    r = min(s / h, s / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (s - nh) // 2, (s - nw) // 2
    canvas = np.full((s, s, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), r, (left, top)


def decode_seg(out: np.ndarray, protos: np.ndarray, nc: int, ratio: float, left: int, top: int,
               orig_hw: tuple[int, int], imgsz: int, *, conf: float = 0.25, iou: float = 0.5,
               max_det: int = 300, min_area_px: float = 4.0) -> list[SegDetection]:
    import cv2

    boxes = out[:4].T
    scores_all = out[4:4 + nc].T
    coefs = out[4 + nc:].T
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(cls)), cls]
    keep = scores >= conf
    if not keep.any():
        return []
    boxes, cls, scores, coefs = boxes[keep], cls[keep], scores[keep], coefs[keep]
    xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                     boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)

    # class-aware NMS: shift each class far apart so classes never suppress each other
    off = cls[:, None].astype(np.float32) * 10_000.0
    shifted = xyxy + off
    rects = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in shifted]
    idx = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), conf, iou)
    idx = np.array(idx).reshape(-1)[:max_det] if len(idx) else np.array([], dtype=int)
    if not len(idx):
        return []

    c, mh, mw = protos.shape
    masks = 1 / (1 + np.exp(-(coefs[idx] @ protos.reshape(c, -1)))).reshape(-1, mh, mw)
    sx, sy = mw / imgsz, mh / imgsz
    H, W = orig_hw
    nh, nw = int(round(H * ratio)), int(round(W * ratio))
    out_dets = []
    for m, b, k, s in zip(masks, xyxy[idx], cls[idx], scores[idx]):
        x1, y1, x2, y2 = b
        crop = np.zeros_like(m)
        cx1, cy1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        cx2, cy2 = min(mw, int(np.ceil(x2 * sx))), min(mh, int(np.ceil(y2 * sy)))
        crop[cy1:cy2, cx1:cx2] = m[cy1:cy2, cx1:cx2]
        full = cv2.resize(crop, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)[top:top + nh, left:left + nw]
        full = cv2.resize(full, (W, H), interpolation=cv2.INTER_LINEAR)
        binm = (full > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area_px:
            continue
        cnt = cv2.approxPolyDP(cnt, 1.0, True).reshape(-1, 2).astype(float)
        if len(cnt) < 3:
            continue
        bx = ((x1 - left) / ratio, (y1 - top) / ratio, (x2 - left) / ratio, (y2 - top) / ratio)
        out_dets.append(SegDetection(
            box=tuple(float(np.clip(v, 0, W if i % 2 == 0 else H)) for i, v in enumerate(bx)),
            polygon=[(float(x), float(y)) for x, y in cnt], cls=int(k), score=float(s)))
    return out_dets
