"""
Offline test for app/engine/benchmark.py — proves the metric math (AP, mAP50,
mAP50-95, precision/recall/F1, false-positive rate) is computed correctly
against hand-constructed predictions/ground truth with known right answers.

This does NOT claim any real model's performance — see registry.py: no
model in the catalog has been benchmarked yet. This only validates the
*measurement tool* itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.benchmark import average_precision, average_precision_dataset, run_benchmark


def test_perfect_predictions_give_ap_one():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    pred_boxes = gt.copy()
    pred_scores = np.array([0.99, 0.95], dtype=np.float32)
    ap = average_precision(pred_boxes, pred_scores, gt, iou_threshold=0.5)
    assert ap == 1.0, f"expected AP=1.0 for perfect predictions, got {ap}"
    print("Perfect predictions -> AP50 = 1.0 : OK")


def test_no_predictions_give_ap_zero():
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    pred_boxes = np.zeros((0, 4), dtype=np.float32)
    pred_scores = np.zeros((0,), dtype=np.float32)
    ap = average_precision(pred_boxes, pred_scores, gt, iou_threshold=0.5)
    assert ap == 0.0
    print("No predictions against real GT -> AP50 = 0.0 : OK")


def test_matching_is_scoped_per_image_not_global():
    """
    Regression test: a prediction in image B must NOT be allowed to match a
    ground-truth box that actually lives in image A, just because the raw
    coordinates happen to coincide. Image A has one (unmatched, since it has
    no prediction) GT box; image B has one prediction at those exact same
    coordinates but NO ground truth of its own.

    If matching ignored image_id (pooled globally), B's prediction would
    spatially overlap A's leftover GT box and be scored as a true positive
    (AP -> 1.0) — silently crediting a detection in the wrong image. With
    correct per-image scoping, B's prediction is a false positive (nothing
    to match in B) and A's GT is simply a miss, so precision/recall are both
    0 and AP must be 0.0.
    """
    pred_items = [("imgB", (0, 0, 10, 10), 0.8)]
    gt_items = [("imgA", (0, 0, 10, 10))]

    ap = average_precision_dataset(pred_items, gt_items, iou_threshold=0.5)
    assert ap == 0.0, (
        f"expected AP=0.0 (prediction and GT are in different images, so no match "
        f"should be possible), got {ap} — matching may have leaked across images"
    )
    print(f"Per-image match scoping: AP={ap} (0.0 as expected — no cross-image leakage)")


def test_run_benchmark_end_to_end():
    # 2 images, 1 class. Image 1: perfect match. Image 2: one false positive,
    # one missed GT (false negative) -> known precision/recall by hand.
    predictions = [
        {"image_id": "img1", "box": [0, 0, 10, 10], "score": 0.9, "class_id": 0},     # TP
        {"image_id": "img2", "box": [100, 100, 110, 110], "score": 0.8, "class_id": 0},  # FP (no nearby GT)
    ]
    ground_truth = [
        {"image_id": "img1", "box": [0, 0, 10, 10], "class_id": 0},   # matched
        {"image_id": "img2", "box": [50, 50, 60, 60], "class_id": 0},  # missed -> FN
    ]

    report = run_benchmark(
        model_id="unit_test_model",
        predictions=predictions,
        ground_truth=ground_truth,
        num_images=2,
        inference_times_ms=[120.0, 110.0],
    )

    assert report.num_gt_instances == 2
    assert report.num_predictions == 2
    assert abs(report.precision - 0.5) < 1e-6, f"expected precision=0.5 (1 TP / 2 preds), got {report.precision}"
    assert abs(report.recall - 0.5) < 1e-6, f"expected recall=0.5 (1 TP / 2 GT), got {report.recall}"
    assert abs(report.false_positive_rate - 0.5) < 1e-6
    assert report.throughput_fps > 0
    assert 0.0 <= report.map50 <= 1.0
    assert 0.0 <= report.map50_95 <= 1.0
    print(f"Benchmark end-to-end: precision={report.precision}, recall={report.recall}, "
          f"f1={report.f1}, map50={report.map50}, map50_95={report.map50_95}, "
          f"throughput={report.throughput_fps} fps")


if __name__ == "__main__":
    test_perfect_predictions_give_ap_one()
    test_no_predictions_give_ap_zero()
    test_matching_is_scoped_per_image_not_global()
    test_run_benchmark_end_to_end()
    print("test_benchmark_offline: PASSED")
