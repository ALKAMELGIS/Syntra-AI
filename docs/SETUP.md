# Setup

## Windows (the usual way)

1. Python 3.11 to 3.14 and Node 20 or newer.
2. Once: `cd backend && pip install -r requirements.txt` and `cd frontend && npm install`.
3. Double-click `start-local.bat`. It stops any Syntra still running (by port and by the
   engine's command line), starts the engine on port 8011 and the interface on
   http://localhost:5291.

`scripts\start-windows.ps1` does the same in a conda environment it creates on first run.

## Settings files

| File | Holds | In git |
| --- | --- | --- |
| `backend/.env` | Engine settings: data folder, import roots, Sentinel Hub configuration id. Copy from `.env.example` | no |
| `frontend/.env` | `VITE_API_PORT` (8011), `VITE_PORT` (5291) | no |
| `backend/data/secrets.json` | Keys entered in Settings (Hugging Face, RunPod, Google): never returned by the API | no |

## Installing in parts

| File | Installs |
| --- | --- |
| `requirements-core.txt` | The engine: API, GeoTIFF reading, exports, remote sensing. No model runs |
| `requirements-ml.txt` | torch, ultralytics, onnxruntime, scipy, safetensors: detection and training |
| `requirements.txt` | Both |
| `requirements-dev.txt` | The engine plus pytest |

For an NVIDIA GPU install torch from https://pytorch.org first, then `requirements-ml.txt`.
Before the first start, `python scripts/check_environment.py` reports what is missing.

## Model weights

Weights are never in git. The model list's Install button downloads a pack from its fixed URL
into `backend/weights/packs/` and checks its sha256; From file accepts a copy you already have.
Models you train are kept in `backend/data/models/`.

## Known Windows pitfalls

- Port 8000 can be reserved by Windows (`WinError 10013`): Syntra uses 8011.
- `uvicorn` may not be on `PATH` after pip: use `python -m uvicorn`.
- Another GIS tool (PostGIS, QGIS, ArcGIS) can set `PROJ_LIB`/`PROJ_DATA` to an incompatible
  `proj.db` and break every CRS call. `check_environment.py` detects it; `.env.example` shows the fix.
- An old engine left running can keep answering after its window is closed: `start-local.bat`
  stops it by command line.

## Docker

`docker compose up --build` builds the engine (Python 3.13, CPU) and the interface behind nginx.
Not yet run end to end.
