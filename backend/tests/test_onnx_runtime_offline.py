"""GPU choice for onnxruntime: found without torch, best provider first, CPU always last."""
import pytest


@pytest.fixture()
def fake(monkeypatch):
    import app.core.onnx_runtime as rt

    state = {"providers": ["CPUExecutionProvider"], "cards": ()}
    monkeypatch.setattr(rt, "available_providers", lambda: list(state["providers"]))
    rt.nvidia_gpus.cache_clear()
    monkeypatch.setattr(rt, "nvidia_gpus", lambda: state["cards"])
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)            # no torch on this machine
    return state


def test_cpu_device_uses_cpu_only(fake):
    from app.core.onnx_runtime import session_providers

    fake["providers"] = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert session_providers("cpu") == ["CPUExecutionProvider"]


def test_gpu_order_tensorrt_cuda_directml_cpu(fake, tmp_path):
    from app.core.onnx_runtime import session_providers

    fake["providers"] = ["CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"]
    p = session_providers("cuda:1", tmp_path / "trt")
    names = [x[0] if isinstance(x, tuple) else x for x in p]
    assert names == ["TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    trt = dict(p[0][1])
    assert trt["device_id"] == 1 and trt["trt_fp16_enable"] and trt["trt_engine_cache_enable"]
    assert (tmp_path / "trt").is_dir()


def test_nvidia_card_found_without_torch(fake):
    from app.core.compute import DeviceOption, _cuda_devices, resolve_device

    fake["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    fake["cards"] = ({"index": 0, "name": "NVIDIA GeForce RTX 4060", "total_vram_mb": 8188, "free_vram_mb": 7400},)
    assert _cuda_devices()[0]["name"] == "NVIDIA GeForce RTX 4060"
    assert resolve_device(DeviceOption.GPU).resolved == "cuda:0"


def test_card_without_gpu_runtime_is_not_offered_and_says_how_to_fix(fake):
    from app.core.compute import _cuda_devices
    from app.core.onnx_runtime import gpu_status

    fake["cards"] = ({"index": 0, "name": "NVIDIA RTX A2000", "total_vram_mb": 6138, "free_vram_mb": 6000},)
    assert _cuda_devices() == []                                # CPU onnxruntime: cannot run there
    st = gpu_status()
    assert st["usable"] is False and "onnxruntime-gpu" in st["fix"]


def test_directml_only_machine(fake):
    from app.core.compute import _cuda_devices
    from app.core.onnx_runtime import session_providers

    fake["providers"] = ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert _cuda_devices()[0]["name"] == "GPU (DirectML)"
    assert [x[0] if isinstance(x, tuple) else x for x in session_providers("cuda:0")] == ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_nvidia_smi_parsing(monkeypatch):
    import subprocess

    import app.core.onnx_runtime as rt

    rt.nvidia_gpus.cache_clear()
    monkeypatch.setattr(rt.shutil, "which", lambda name: "nvidia-smi")

    class R:
        stdout = "0, NVIDIA GeForce RTX 4090, 24564, 23000\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert rt.nvidia_gpus() == ({"index": 0, "name": "NVIDIA GeForce RTX 4090", "total_vram_mb": 24564, "free_vram_mb": 23000},)
    rt.nvidia_gpus.cache_clear()
