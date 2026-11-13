"""
Torch-free inference for YOLO oriented-box (OBB) models exported to ONNX.

Why: PyTorch is the heaviest and most fragile dependency on a Windows GIS
workstation (multi-GB wheels, CUDA mismatches, blocked package indexes).
An ONNX export of the same network runs on `onnxruntime` + numpy + OpenCV,
gives the same detections, and is fast enough on CPU.

Export once on any machine that has ultralytics + torch:
    yolo export model=yolo11n-obb.pt format=onnx imgsz=1024 opset=17
then point SYNTRA_AERIAL_OBB_WEIGHTS at the .onnx file.

Output layout of an ultralytics OBB ONNX graph: (1, 4 + nc + 1, N) with rows
[cx, cy, w, h, class scores x nc, angle (radians)] in letterboxed input pixels.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ObbDetection:
    corners: np.ndarray   # (4, 2) pixel coords in the original image
    cls: int
    score: float


class OnnxObbModel:
    task = "obb"

    def __init__(self, path: str, providers: list[str] | None = None, threads: int | None = None):
        import onnxruntime as ort  # lazy

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(path, sess_options=opts, providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 1024
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(meta["names"]) if "names" in meta else {}

    # ------------------------------------------------------------------ api
    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45,
                classes: list[int] | None = None, max_det: int = 1000) -> list[ObbDetection]:
        blob, ratio, (pad_x, pad_y) = self._letterbox(image_bgr)
        out = self.session.run(None, {self.input_name: blob})[0][0]   # (4+nc+1, N)
        return decode_obb(out, len(self.names) or out.shape[0] - 5, ratio, pad_x, pad_y,
                          conf=conf, iou=iou, classes=classes, max_det=max_det)

    # ------------------------------------------------------------- helpers
    def _letterbox(self, img: np.ndarray):
        import cv2

        h, w = img.shape[:2]
        s = self.imgsz
        r = min(s / h, s / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        if (nh, nw) != (h, w):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (s - nw) / 2, (s - nh) / 2
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = img
        blob = canvas[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0   # BGR->RGB, CHW
        return np.ascontiguousarray(blob), r, (left, top)


def _regularize(w: np.ndarray, h: np.ndarray, a: np.ndarray):
    """Same convention as ultralytics.regularize_rboxes: long side is w, angle in [0, pi)."""
    swap = w < h
    w2 = np.where(swap, h, w)
    h2 = np.where(swap, w, h)
    a2 = np.where(swap, a + math.pi / 2, a) % math.pi
    return w2, h2, a2


def _corners(cx, cy, w, h, a) -> np.ndarray:
    cos, sin = np.cos(a), np.sin(a)
    vx = np.stack([w / 2 * cos, w / 2 * sin], -1)
    vy = np.stack([-h / 2 * sin, h / 2 * cos], -1)
    c = np.stack([cx, cy], -1)
    return np.stack([c + vx + vy, c + vx - vy, c - vx - vy, c - vx + vy], 1)   # (N, 4, 2)


def decode_obb(out: np.ndarray, nc: int, ratio: float, pad_x: float, pad_y: float, *,
               conf: float = 0.25, iou: float = 0.45, classes: list[int] | None = None,
               max_det: int = 1000) -> list[ObbDetection]:
    import cv2

    boxes = out[:4].T                       # (N, 4) cx, cy, w, h
    scores_all = out[4:4 + nc].T            # (N, nc)
    angles = out[4 + nc]                    # (N,)
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(cls)), cls]
    keep = scores >= conf
    if classes is not None:
        keep &= np.isin(cls, classes)
    if not keep.any():
        return []
    boxes, cls, scores, angles = boxes[keep], cls[keep], scores[keep], angles[keep]

    w, h, a = _regularize(boxes[:, 2], boxes[:, 3], angles)
    cx, cy = boxes[:, 0], boxes[:, 1]

    # Class-aware rotated NMS: offset each class far apart so boxes of
    # different classes never suppress each other (same trick as ultralytics).
    off = cls.astype(np.float32) * 10_000.0
    rects = [((float(x + o), float(y + o)), (float(ww), float(hh)), float(math.degrees(t)))
             for x, y, ww, hh, t, o in zip(cx, cy, w, h, a, off)]
    idx = cv2.dnn.NMSBoxesRotated(rects, scores.astype(float).tolist(), conf, iou)
    idx = np.array(idx).reshape(-1)[:max_det] if len(idx) else np.array([], dtype=int)

    pts = _corners(cx[idx], cy[idx], w[idx], h[idx], a[idx])
    pts[..., 0] = (pts[..., 0] - pad_x) / ratio
    pts[..., 1] = (pts[..., 1] - pad_y) / ratio
    return [ObbDetection(corners=p, cls=int(c), score=float(s)) for p, c, s in zip(pts, cls[idx], scores[idx])]
