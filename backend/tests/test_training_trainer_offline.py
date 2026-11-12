"""
Offline tests for app.training.trainer: the subprocess-supervision and
event-parsing logic is exercised against tests/fixtures/fake_train_worker.py
(a tiny stand-in that speaks the real worker's exact stdout protocol), so
this runs with no ultralytics/torch installed.
"""
import sys
from pathlib import Path

from app.training.autoparams import (
    suggest_batch_size,
    suggest_epochs,
    suggest_imgsz,
    suggest_training_params,
    suggest_workers,
)
from app.training.trainer import build_job_config, request_cancel, run_training

FIXTURE = str(Path(__file__).parent / "fixtures" / "fake_train_worker.py")


def _fake_cmd():
    return [sys.executable, FIXTURE]


def test_suggest_workers_leaves_a_core_free_and_caps_at_eight():
    assert suggest_workers(4) == 3
    assert suggest_workers(32) == 8
    assert suggest_workers(1) == 0


def test_suggest_batch_size_shrinks_on_cpu_and_low_ram():
    gpu_batch = suggest_batch_size(640, "cuda:0", free_vram_mb=8000)
    cpu_batch = suggest_batch_size(640, "cpu")
    cpu_low_ram = suggest_batch_size(640, "cpu", available_ram_mb=2048)
    assert cpu_batch < gpu_batch
    assert cpu_low_ram < cpu_batch


def test_suggest_batch_size_respects_vram_cap():
    small_vram = suggest_batch_size(640, "cuda:0", free_vram_mb=100)
    large_vram = suggest_batch_size(640, "cuda:0", free_vram_mb=20000)
    assert small_vram <= large_vram


def test_suggest_imgsz_is_stride_32_aligned():
    assert suggest_imgsz(None, tile_size=640) % 32 == 0
    assert suggest_imgsz(None, tile_size=500) % 32 == 0


def test_suggest_epochs_scales_down_for_larger_datasets():
    assert suggest_epochs(50) > suggest_epochs(2000) > suggest_epochs(20000)


def test_suggest_training_params_bundle_has_all_keys():
    out = suggest_training_params(device="cpu", imgsz=640, num_train_images=300, cpu_threads=8)
    assert set(out) == {"batch", "workers", "epochs", "imgsz"}


def test_build_job_config_applies_overrides(tmp_path):
    cfg = build_job_config(
        str(tmp_path / "data.yaml"), str(tmp_path / "out"),
        num_train_images=100, device="cpu", overrides={"epochs": 5, "batch": 4},
    )
    assert cfg["epochs"] == 5
    assert cfg["batch"] == 4
    assert cfg["model"] == "yolov8n.pt"          # default model_size="nano"
    assert cfg["data_yaml"].endswith("data.yaml")


def test_run_training_streams_epoch_events_and_reports_done(tmp_path):
    cfg = build_job_config(
        str(tmp_path / "data.yaml"), str(tmp_path / "out"),
        num_train_images=50, device="cpu", overrides={"epochs": 4},
    )
    events = []
    final = run_training(cfg, events.append, cmd=_fake_cmd())
    epoch_events = [e for e in events if e["type"] == "epoch"]
    assert len(epoch_events) == 4
    assert [e["epoch"] for e in epoch_events] == [1, 2, 3, 4]
    assert final["type"] == "done"
    assert final["best"].endswith("best.pt")


def test_run_training_honors_cancel_flag(tmp_path):
    cancel_flag = str(tmp_path / "cancel.flag")
    cfg = build_job_config(
        str(tmp_path / "data.yaml"), str(tmp_path / "out"),
        num_train_images=50, device="cpu", overrides={"epochs": 50},
        cancel_flag_path=cancel_flag,
    )
    events = []

    def on_event(e):
        events.append(e)
        if e["type"] == "epoch" and e["epoch"] == 2:
            request_cancel(cancel_flag)

    final = run_training(cfg, on_event, cmd=_fake_cmd())
    epoch_events = [e for e in events if e["type"] == "epoch"]
    # stops within a couple of epochs of the cancel request, not after all 50
    assert len(epoch_events) <= 4
    assert final["type"] == "canceled"
