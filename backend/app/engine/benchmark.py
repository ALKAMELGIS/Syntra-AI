"""
Benchmarking — spec Phase 6 / Section 6 reporting requirements:
precision, recall, F1, mAP50, mAP50-95, inference time, throughput,
GPU memory consumption, false-positive rate.

The metric computation here is real (pure NumPy, COCO-style 101-point
interpolated AP, matched per-image so predictions from one image can never
spuriously match ground truth from another) and is unit-tested with
synthetic predictions/ground truth in tests/test_benchmark_offline.py.
What it explicitly does NOT do is invent numbers for a model that hasn't
actually been run: `run_benchmark()` requires real predictions and real
ground truth, and there is no code path that returns a BenchmarkReport
without both.

Per the spec: "Do not claim performance figures without actual benchmark
results." No model in the current catalog (see app/models/registry.py) has
been benchmarked — this module is the tool to do that once trained weights
and a labeled, held-out test set exist.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from app.engine.nms import iou_matrix


@dataclass
class BenchmarkReport:
    model_id: str
    num_images: int
    num_gt_instances: int
    num_predictions: int
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    false_positive_rate: float
    inference_time_ms_per_image: float
    throughput_fps: float
    gpu_memory_mb: float | None
    per_class_ap50: dict[int, float] = field(default_factory=dict)


def _match_at_iou(
    pred_boxes: np.ndarray, pred_scores: np.ndarray,
    gt_boxes: np.ndarray, iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy matching within ONE image, highest-confidence predictions first.
    Returns (tp, fp) boolean arrays aligned with `pred_boxes` in its
    original (caller-supplied) order.
    """
    order = pred_scores.argsort()[::-1]
    pred_boxes_sorted = pred_boxes[order]

    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes_sorted), dtype=bool)
    fp = np.zeros(len(pred_boxes_sorted), dtype=bool)

    if len(gt_boxes) == 0:
        fp[:] = True
    else:
        ious = iou_matrix(pred_boxes_sorted, gt_boxes)  # (P, G)
        for i in range(len(pred_boxes_sorted)):
            row = ious[i].copy()
            row[matched_gt] = -1  # already-matched GT can't be matched twice
            best_gt = int(row.argmax()) if len(row) else -1
            if best_gt >= 0 and row[best_gt] >= iou_threshold:
                tp[i] = True
                matched_gt[best_gt] = True
            else:
                fp[i] = True

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return tp[inverse], fp[inverse]


def _match_per_image(
    pred_items: list[tuple[str, tuple, float]],   # (image_id, box, score)
    gt_items: list[tuple[str, tuple]],             # (image_id, box)
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Groups predictions/GT by image_id and matches independently within each
    image — a prediction can never match a ground-truth box from a
    different image, however close the raw coordinates happen to be.

    Returns (scores, tp, fp, total_gt) — flat arrays spanning all images,
    ready for a single dataset-wide precision/recall curve.
    """
    preds_by_img: dict[str, list[tuple[tuple, float]]] = defaultdict(list)
    gts_by_img: dict[str, list[tuple]] = defaultdict(list)
    for img_id, box, score in pred_items:
        preds_by_img[img_id].append((box, score))
    for img_id, box in gt_items:
        gts_by_img[img_id].append(box)

    all_scores: list[float] = []
    all_tp: list[bool] = []
    all_fp: list[bool] = []

    # Every image that has predictions must be matched against its own GT
    # (possibly empty, meaning every prediction there is a false positive).
    for img_id, preds in preds_by_img.items():
        boxes = np.array([p[0] for p in preds], dtype=np.float32).reshape(-1, 4)
        scores = np.array([p[1] for p in preds], dtype=np.float32)
        gt_boxes = np.array(gts_by_img.get(img_id, []), dtype=np.float32).reshape(-1, 4)
        tp, fp = _match_at_iou(boxes, scores, gt_boxes, iou_threshold)
        all_scores.extend(scores.tolist())
        all_tp.extend(tp.tolist())
        all_fp.extend(fp.tolist())

    total_gt = sum(len(v) for v in gts_by_img.values())
    return np.array(all_scores), np.array(all_tp, dtype=bool), np.array(all_fp, dtype=bool), total_gt


def _ap_from_matches(scores: np.ndarray, tp: np.ndarray, fp: np.ndarray, total_gt: int) -> float:
    """COCO-style 101-point interpolated AP from dataset-wide (already-matched) tp/fp flags."""
    if total_gt == 0:
        return 0.0 if len(scores) > 0 else 1.0
    if len(scores) == 0:
        return 0.0

    order = scores.argsort()[::-1]
    tp_cum = np.cumsum(tp[order])
    fp_cum = np.cumsum(fp[order])

    recalls = tp_cum / total_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    recall_thresholds = np.linspace(0, 1, 101)
    interpolated = np.zeros_like(recall_thresholds)
    for i, rt in enumerate(recall_thresholds):
        mask = recalls >= rt
        interpolated[i] = precisions[mask].max() if mask.any() else 0.0

    return float(interpolated.mean())


def average_precision(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float) -> float:
    """
    COCO-style 101-point interpolated AP for a single class at a single IoU
    threshold, for predictions/GT already belonging to one image (or already
    safely pooled by the caller). For multi-image datasets, use
    `average_precision_dataset` instead so matching stays image-scoped.
    """
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) > 0 else 1.0
    if len(pred_boxes) == 0:
        return 0.0
    tp, fp = _match_at_iou(pred_boxes, pred_scores, gt_boxes, iou_threshold)
    return _ap_from_matches(pred_scores, tp, fp, len(gt_boxes))


def average_precision_dataset(
    pred_items: list[tuple[str, tuple, float]],
    gt_items: list[tuple[str, tuple]],
    iou_threshold: float,
) -> float:
    """AP for one class across a whole (multi-image) dataset, matched per image."""
    scores, tp, fp, total_gt = _match_per_image(pred_items, gt_items, iou_threshold)
    return _ap_from_matches(scores, tp, fp, total_gt)


def run_benchmark(
    model_id: str,
    predictions: list[dict],   # [{"image_id", "box": xyxy, "score", "class_id"}, ...]
    ground_truth: list[dict],  # [{"image_id", "box": xyxy, "class_id"}, ...]
    num_images: int,
    inference_times_ms: list[float],
    gpu_memory_mb: float | None = None,
    iou_range: tuple[float, float, float] = (0.5, 0.95, 0.05),
) -> BenchmarkReport:
    """
    Compute the full metric set the spec requires (Section 6). Requires real
    predictions from an actual model run and real ground-truth annotations —
    there is no fallback that fabricates a result. All matching is scoped
    per image_id, so this is correct for multi-image benchmark sets.
    """
    if not inference_times_ms:
        raise ValueError("inference_times_ms must contain at least one real measured timing.")

    class_ids = sorted({p["class_id"] for p in predictions} | {g["class_id"] for g in ground_truth})
    iou_start, iou_end, step = iou_range
    thresholds = [round(t, 2) for t in np.arange(iou_start, iou_end + 1e-9, step)]

    per_class_ap50: dict[int, float] = {}
    per_class_ap_range: dict[int, float] = {}

    all_tp = all_fp = 0
    total_gt = len(ground_truth)

    for cls in class_ids:
        pred_items = [(p["image_id"], tuple(p["box"]), p["score"]) for p in predictions if p["class_id"] == cls]
        gt_items = [(g["image_id"], tuple(g["box"])) for g in ground_truth if g["class_id"] == cls]

        per_class_ap50[cls] = average_precision_dataset(pred_items, gt_items, 0.5)
        per_class_ap_range[cls] = float(np.mean([average_precision_dataset(pred_items, gt_items, t) for t in thresholds]))

        _, tp, fp, _ = _match_per_image(pred_items, gt_items, 0.5)
        all_tp += int(tp.sum())
        all_fp += int(fp.sum())

    precision = all_tp / max(1, all_tp + all_fp)
    recall = all_tp / max(1, total_gt)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    fp_rate = all_fp / max(1, all_tp + all_fp)

    mean_time_ms = float(np.mean(inference_times_ms))

    return BenchmarkReport(
        model_id=model_id,
        num_images=num_images,
        num_gt_instances=total_gt,
        num_predictions=len(predictions),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        map50=round(float(np.mean(list(per_class_ap50.values()))) if per_class_ap50 else 0.0, 4),
        map50_95=round(float(np.mean(list(per_class_ap_range.values()))) if per_class_ap_range else 0.0, 4),
        false_positive_rate=round(fp_rate, 4),
        inference_time_ms_per_image=round(mean_time_ms, 3),
        throughput_fps=round(1000.0 / mean_time_ms, 3) if mean_time_ms > 0 else 0.0,
        gpu_memory_mb=gpu_memory_mb,
        per_class_ap50={k: round(v, 4) for k, v in per_class_ap50.items()},
    )
