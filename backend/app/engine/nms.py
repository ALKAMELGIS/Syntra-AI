"""
Non-Maximum Suppression and cross-tile duplicate-detection removal.

Pure NumPy — no torch/torchvision dependency, so it's testable in any
environment and reusable for both box detections and (via their bounding
envelopes) segmentation instances.
"""
from __future__ import annotations

import numpy as np


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """boxes_* are (N, 4) / (M, 4) arrays of (x1, y1, x2, y2). Returns (N, M) IoU matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """Greedy class-agnostic NMS. Returns indices of boxes to keep, sorted by score desc."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i:i + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """NMS applied independently within each class — prevents suppressing a car behind a ship, etc."""
    keep_all: list[int] = []
    for cls in np.unique(class_ids):
        mask = np.where(class_ids == cls)[0]
        kept_local = nms(boxes[mask], scores[mask], iou_threshold)
        keep_all.extend(mask[kept_local].tolist())
    return np.array(sorted(keep_all), dtype=np.int64)


def merge_tile_duplicates(
    detections: list[dict],
    iou_threshold: float = 0.45,
) -> list[dict]:
    """
    Remove duplicate detections that straddle overlapping tile borders.

    `detections` is a flat list of dicts with keys: box_geo (x1,y1,x2,y2 in
    the mosaic/global coordinate frame — NOT per-tile pixel space), score,
    class_id. Call this *after* georeferencing every tile's detections back
    into a shared pixel or geo frame for the whole raster, so boxes from
    neighboring tiles are directly comparable.
    """
    if not detections:
        return []

    boxes = np.array([d["box_geo"] for d in detections], dtype=np.float32)
    scores = np.array([d["score"] for d in detections], dtype=np.float32)
    class_ids = np.array([d["class_id"] for d in detections])

    keep_idx = nms_per_class(boxes, scores, class_ids, iou_threshold)
    return [detections[i] for i in keep_idx]
