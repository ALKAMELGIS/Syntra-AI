"""
Syntra AI — environment / dependency doctor.

Single source of truth for "is this machine's Python environment actually
able to run the Syntra AI backend" — checks which required packages are
*importable* in the current interpreter (not just listed in requirements.txt),
whether a CUDA GPU is visible to torch, and whether rasterio/pyproj can
actually build a real CRS (catches the PROJ-database-shadowing gotcha found
on a real Windows deployment).

Used by two callers that must never drift apart:
  - scripts/check_environment.py          (CLI, run by a human before deploying)
  - app.api.environment (GET /api/environment)   (the Detection Models tab's
    "Library Status" panel)

Nothing here is faked or assumed: every row reflects a real `importlib.import_module`
call, a real `torch.cuda.is_available()` call, or a real `CRS.from_epsg(4326)` call
against whatever interpreter this process is actually running in.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# (import name, pip/conda package name, why it's needed)
REQUIRED: list[tuple[str, str, str]] = [
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

BACKEND_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


def load_backend_env() -> dict:
    """
    Loads backend/.env the same way app.core.config does, so a standalone
    caller (the CLI script) sees the same PROJ_LIB/PROJ_DATA overrides the
    real server would. Safe to call more than once.
    """
    found = BACKEND_ENV_PATH.exists()
    if found:
        try:
            from dotenv import load_dotenv

            load_dotenv(BACKEND_ENV_PATH)
        except ImportError:
            pass
    return {"path": str(BACKEND_ENV_PATH), "found": found}


def check_import(module_name: str) -> dict:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown version")
        path = getattr(mod, "__file__", None)
        return {"installed": True, "detail": version, "path": str(Path(path).parent) if path else None, "error": None}
    except ImportError as exc:
        return {"installed": False, "detail": None, "path": None, "error": str(exc)}


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


def check_proj_database() -> dict:
    """
    Confirms rasterio/pyproj can actually build a CRS, not just import.

    Real-world failure mode this catches: a system-wide PROJ_LIB/PROJ_DATA
    env var set by another installed GIS tool (PostgreSQL/PostGIS, QGIS,
    Anaconda's own GDAL, ArcGIS) can shadow pyproj's own bundled proj.db
    with an incompatible/older one. Both rasterio and pyproj import fine in
    that case — this only breaks the moment you actually try to use a CRS,
    e.g. rasterio.open(..., crs="EPSG:4326").
    """
    try:
        from rasterio.crs import CRS

        CRS.from_epsg(4326)
        return {"ok": True, "error": None}
    except ImportError:
        return {"ok": None, "error": None}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def build_report() -> dict:
    env_info = load_backend_env()

    results = []
    for module_name, pip_name, purpose in REQUIRED:
        check = check_import(module_name)
        results.append({
            "module": module_name,
            "package": pip_name,
            "purpose": purpose,
            **check,
        })

    cuda_info = check_cuda()
    proj_info = check_proj_database()
    missing = [r for r in results if not r["installed"]]
    proj_broken = proj_info["ok"] is False

    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "results": results,
        "missing_count": len(missing),
        "total_count": len(results),
        "cuda": cuda_info,
        "proj_database": proj_info,
        "backend_env_file": env_info,
        "production_ready": len(missing) == 0 and not proj_broken,
    }
