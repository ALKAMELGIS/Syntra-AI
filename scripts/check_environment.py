#!/usr/bin/env python3
"""
Syntra AI — environment doctor (CLI).

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

The actual checking logic lives in app.diagnostics.environment, shared with
the backend's GET /api/environment endpoint (the Detection Models tab's
"Library Status" panel) so the CLI and the UI can never disagree.

Usage:
    python scripts/check_environment.py
    python scripts/check_environment.py --json     # machine-readable output
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.diagnostics.environment import build_report  # noqa: E402


def main() -> int:
    as_json = "--json" in sys.argv
    report = build_report()
    missing = [r for r in report["results"] if not r["installed"]]
    proj_broken = report["proj_database"]["ok"] is False

    if as_json:
        print(json.dumps(report, indent=2))
        return 1 if (missing or proj_broken) else 0

    print(f"Syntra AI environment check — Python {report['python']} on {report['platform']}")
    env_file = report["backend_env_file"]
    print(f"backend/.env: {'loaded (' + env_file['path'] + ')' if env_file['found'] else 'not found -- PROJ_LIB/PROJ_DATA overrides in it, if any, were NOT applied'}\n")
    for r in report["results"]:
        status = "OK   " if r["installed"] else "MISSING"
        extra = f"({r['detail']})" if r["installed"] else f"-- pip install {r['package']}"
        print(f"  [{status}] {r['module']:<28} {r['purpose']:<48} {extra}")

    print()
    cuda_info = report["cuda"]
    if cuda_info["torch_installed"]:
        if cuda_info["cuda_available"]:
            names = ", ".join(d["name"] for d in cuda_info["devices"])
            print(f"GPU: CUDA available — {len(cuda_info['devices'])} device(s): {names}")
        else:
            print("GPU: torch is installed but torch.cuda.is_available() is False "
                  "(no GPU, or a CPU-only torch build, or missing drivers).")
    else:
        print("GPU: torch is not installed, so CUDA availability cannot be checked yet.")

    proj_info = report["proj_database"]
    if proj_info["ok"] is False:
        print()
        print("PROJ database: BROKEN -- rasterio/pyproj import fine, but building a real CRS fails:")
        print(f"    {proj_info['error']}")
        print("This usually means another installed GIS tool (PostgreSQL/PostGIS, QGIS, Anaconda's")
        print("own GDAL, ArcGIS) set a system-wide PROJ_LIB/PROJ_DATA pointing at an incompatible")
        print("proj.db. Fix (this process only, no permanent system change):")
        print('    python -c "import pyproj; print(pyproj.datadir.get_data_dir())"')
        print("    -- then set PROJ_LIB and PROJ_DATA to that path in backend/.env (see .env.example)")
    elif proj_info["ok"] is True:
        print()
        print("PROJ database: OK -- rasterio can build a real CRS (EPSG:4326 round-tripped).")

    print()
    if missing or proj_broken:
        if missing:
            print(f"Result: {len(missing)}/{report['total_count']} dependencies missing. "
                  f"This machine is NOT ready to run the Syntra AI backend yet.")
            print("Install the full stack with:\n    pip install -r backend/requirements.txt")
        if proj_broken:
            print("Result: all packages installed, but the PROJ database is broken (see above) -- "
                  "any endpoint that touches a real CRS (image upload, predict, georeferencing) will fail.")
    else:
        print("Result: all dependencies present and the PROJ database is sane. "
              "This does not by itself confirm trained model weights exist — "
              "see backend/app/models/registry.py for per-model production-readiness status.")

    return 1 if (missing or proj_broken) else 0


if __name__ == "__main__":
    raise SystemExit(main())
