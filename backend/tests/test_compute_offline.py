"""
Offline tests for app.core.compute — the CPU0/CPU1/GPU/AUTO device resolver.

Runs with no GPU and no torch installed (this build sandbox), which is
exactly the fallback path these tests exist to lock down: resolve_device()
must never raise and must always degrade to a CPU profile with an honest
'reason' string when CUDA/torch are unavailable, rather than silently
pretending a GPU is in use.

Run directly: python tests/test_compute_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.compute import DeviceOption, DeviceProfile, list_devices, resolve_device  # noqa: E402


def test_cpu0_is_always_available():
    profile = resolve_device(DeviceOption.CPU0)
    assert profile.option == DeviceOption.CPU0
    assert profile.resolved == "cpu"
    assert profile.threads is not None
    assert profile.gpu_index is None


def test_cpu1_is_always_available():
    profile = resolve_device(DeviceOption.CPU1)
    assert profile.option == DeviceOption.CPU1
    assert profile.resolved == "cpu"


def test_gpu_request_falls_back_to_cpu1_without_cuda():
    # This sandbox has no torch installed, so _cuda_devices() must return [].
    profile = resolve_device(DeviceOption.GPU)
    assert profile.option == DeviceOption.CPU1
    assert profile.resolved == "cpu"
    assert "fell back to CPU1" in profile.reason
    assert "GPU requested but no CUDA device" in profile.reason


def test_auto_falls_back_to_cpu1_without_cuda():
    profile = resolve_device(DeviceOption.AUTO)
    assert profile.option == DeviceOption.CPU1
    assert profile.resolved == "cpu"
    assert "no CUDA device detected" in profile.reason


def test_auto_with_estimated_vram_still_falls_back_cleanly():
    # Passing an estimated_vram_mb hint must not change behaviour when
    # there is no GPU to evaluate it against.
    profile = resolve_device(DeviceOption.AUTO, estimated_vram_mb=4096)
    assert profile.option == DeviceOption.CPU1
    assert profile.resolved == "cpu"


def test_resolve_device_never_raises_and_always_returns_profile():
    for option in DeviceOption:
        profile = resolve_device(option)
        assert isinstance(profile, DeviceProfile)
        assert profile.reason  # always a human-readable explanation
        assert profile.resolved in ("cpu", "cuda") or profile.resolved.startswith("cuda:")


def test_list_devices_reports_zero_gpus_honestly():
    inventory = list_devices()
    assert inventory["gpu"] == []
    assert inventory["gpu_count"] == 0
    assert "threads" in inventory["cpu0"]
    assert "threads" in inventory["cpu1"]


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  OK  {t.__name__}")
    print(f"\n{passed}/{len(tests)} compute tests passed.")


if __name__ == "__main__":
    _run_all()
