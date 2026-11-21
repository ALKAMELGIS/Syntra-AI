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

## Checking the interface

`node scripts/ui_audit.mjs` opens every toolbar panel at four screen sizes in both themes and
fails if two panels are open at once, a panel covers the toolbar or leaves the screen, or the page
throws. It needs Playwright (`npm i -D playwright && npx playwright install chromium`) and a
running Syntra; set `SYNTRA_EMAIL` and `SYNTRA_PASSWORD` to sign in. Screenshots go to `ui-audit/`.

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

## Hugging Face Space (hosted)

The Space `Alkamelgis/SYNTRA-AI` runs the engine and the built interface in one process
(`deploy/space/app.py`, Gradio SDK on the free ZeroGPU hardware: Docker Spaces and CPU Basic need PRO).
`.github/workflows/deploy-space.yml` builds and pushes it on every push to `main` once the repository
secret `HF_TOKEN` (a Hugging Face token with write access) exists; Actions > Deploy > Run workflow
deploys by hand.

From this computer instead: `deploy-space.bat` builds the interface and uploads the Space with
`scripts/deploy_space.py`. It uses the Hugging Face token saved in Syntra (Settings > Services),
or `HF_TOKEN`, or `hf auth login`; the token needs write access.

Space settings: variable `SYNTRA_PUBLIC_URL=https://alkamelgis-syntra-ai.hf.space` (set), secret
`SYNTRA_OWNER_EMAIL` (the owner's email: only it may create the first account). Open the app at that
address, not inside the Hugging Face page (the sign-in cookie needs it). The Space's disk is reset on
restart: accounts, credits and images are kept only if a storage bucket is mounted at `/data`.

## Docker

`docker compose up --build` builds the engine (Python 3.13, CPU) and the interface behind nginx.
Not yet run end to end.
