#!/usr/bin/env python3
"""
Syntra AI — environment doctor.

Checks which of this project's dependencies are actually importable in the
*current* Python interpreter, and whether a CUDA GPU is visible to torch (if
torch is installed). Prints a plain pass/fail report with install hints —
nothing is assumed or faked.

This exists because the cloud build sandbox and the device bridge's sandboxed
VM both have NO package-registry network access, so the full stack
(fastapi/rasterio/torch/ultralytics/...) could never be installed or verified
there — only on your own machine. Run this on your real Windows/conda
environment to find out, honestly, what's missing before trying to launch
the API or run real inference.

Usage:
    python scripts/check_environment.py
    python scripts/check_environment.py --json     # machine-readable output
"""
from __future__ import annotations

import importlib
import json
import sys

# (import name, pip/conda package name, why it's needed)
REQUIRED = [
    ("fastapi", "fastapi", "backend API framework"),
    ("uvicorn", "uvicorn[standard]", "ASGI server to run the API"),
    ("pydantic", "pydantic", "request/response validation"),
    ("multipart", "python-multipart", "file upload handling"),
    ("rasterio", "rasterio", "reading GeoTIFF / windowed tiling"),
    ("shapely", "shapely", "polygon geometry for field boundaries"),
    ("pyproj", "pyproj", "CRS / coordinate reprojection"),
    ("affine", "affine", "pixel<->geo transforms (rasterio dependency)"),
    ("geopandas", "geopandas", "vector dataset I/O"),
    ("numpy", "numpy", "core array math — used everywhere"),
    ("cv2", "opencv-python-headless", "contour extraction for field polygons"),
    ("PIL", "pillow", "raster/image decoding"),
    ("torch", "torch", "model inference runtime (CPU or GPU)"),
    ("torchvision", "torchvision", "vision model utilities"),
    ("ultralytics", "ultralytics", "YOLO-family detectors"),
    ("segmentation_models_pytorch", "segmentation-models-pytorch", "field-boundary segmentation backbone"),
    ("onnxruntime", "onnxruntime-gpu", "optional faster inference path"),
    ("yaml", "pyyaml", "dataset/config YAML parsing"),
    ("dotenv", "python-dotenv", ".env config loading"),
]


def check_import(module_name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown version")
        return True, version
    except ImportError as exc:
        return False, str(exc)


def check_cuda() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch_installed": False, "cuda_available": False, "devices": []}

    available = torch.cuda.is_available()
    devices = []
    if available:
        for i in range(torch.cuda.device_count()):
            devices.append({"index": i, "name": torch.cuda.get_device_name(i)})
    return {"torch_installed": True, "cuda_available": available, "devices": devices}


def main() -> int:
    as_json = "--json" in sys.argv

    results = []
    for module_name, pip_name, purpose in REQUIRED:
        ok, detail = check_import(module_name)
        results.append({
            "module": module_name,
            "package": pip_name,
            "purpose": purpose,
            "installed": ok,
            "detail": detail if ok else None,
            "error": None if ok else detail,
        })

    cuda_info = check_cuda()
    missing = [r for r in results if not r["installed"]]

    report = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "results": results,
        "missing_count": len(missing),
        "total_count": len(results),
        "cuda": cuda_info,
        "production_ready": len(missing) == 0,
    }

    if as_json:
        print(json.dumps(report, indent=2))
        return 1 if missing else 0

    print(f"Syntra AI environment check — Python {report['python']} on {report['platform']}\n")
    for r in results:
        status = "OK   " if r["installed"] else "MISSING"
        extra = f"({r['detail']})" if r["installed"] else f"-- pip install {r['package']}"
        print(f"  [{status}] {r['module']:<28} {r['purpose']:<48} {extra}")

    print()
    if cuda_info["torch_installed"]:
        if cuda_info["cuda_available"]:
            names = ", ".join(d["name"] for d in cuda_info["devices"])
            print(f"GPU: CUDA available — {len(cuda_info['devices'])} device(s): {names}")
        else:
            print("GPU: torch is installed but torch.cuda.is_available() is False "
                  "(no GPU, or a CPU-only torch build, or missing drivers).")
    else:
        print("GPU: torch is not installed, so CUDA availability cannot be checked yet.")

    print()
    if missing:
        print(f"Result: {len(missing)}/{len(results)} dependencies missing. "
              f"This machine is NOT ready to run the Syntra AI backend yet.")
        print("Install the full stack with:\n    pip install -r backend/requirements.txt")
    else:
        print("Result: all dependencies present. "
              "This does not by itself confirm trained model weights exist — "
              "see backend/app/models/registry.py for per-model production-readiness status.")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
