"""Offline test for app/models/stats.py — persisted run accumulation, pure
stdlib (json/pathlib), no GPU/network needed."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.stats import get_stats, record_run


def test_stats_accumulate_across_runs():
    tmp = Path(tempfile.mkdtemp(prefix="syntra_stats_test_"))
    try:
        empty = get_stats(tmp, "car_detection")
        assert empty.run_count == 0
        assert empty.avg_duration_seconds == 0.0

        record_run(tmp, "car_detection", status="completed", tiles_processed=10,
                   detection_count=42, duration_seconds=5.0, device="cuda:0")
        record_run(tmp, "car_detection", status="completed", tiles_processed=12,
                   detection_count=30, duration_seconds=7.0, device="cuda:0")
        record_run(tmp, "car_detection", status="error")

        stats = get_stats(tmp, "car_detection")
        assert stats.run_count == 3
        assert stats.error_count == 1
        assert stats.total_tiles_processed == 22
        assert stats.total_detections == 72
        assert abs(stats.total_duration_seconds - 12.0) < 1e-9
        assert abs(stats.avg_duration_seconds - 4.0) < 1e-9       # 12.0 / 3 runs (error run counts as a run with 0 duration)
        assert abs(stats.avg_detections_per_run - 24.0) < 1e-9    # 72 / 3
        assert stats.last_status == "error"                        # most recent call was the error
        assert stats.last_run_at is not None

        # A different model_id must not share state.
        other = get_stats(tmp, "ship_detection")
        assert other.run_count == 0

        print(f"Stats accumulation: {stats.to_dict()}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_stats_accumulate_across_runs()
    print("test_stats_offline: PASSED")
