# Syntra AI — Detection Engine (Phase 1)

A satellite/drone imagery AI detection platform. Per the product spec, this
build prioritizes a **working detection engine** over dashboard polish: the
engine is fully usable from plain Python (no HTTP layer required) and is
independently tested.

## What's real vs. scaffolded — read this first

This repo was built in a sandbox with **no GPU and no package-registry
network access**, so nothing involving torch/ultralytics/rasterio could
actually be installed or executed here. To avoid claiming things that were
never verified:

| Piece | Status | How it was verified |
|---|---|---|
| Tiling (windowed reads, configurable size/overlap) | ✅ implemented | Logic tested offline with an in-memory array (`iter_tiles_from_array`); `iter_tiles` (the real rasterio path) is written but untested here — needs a real GeoTIFF + rasterio. |
| Geospatial coordinate reconstruction (pixel → geo) | ✅ implemented & tested | `backend/tests/test_pipeline_offline.py` asserts reconstructed coordinates land in the expected geo bounds. |
| Cross-tile duplicate suppression / NMS | ✅ implemented & tested — **real bug found & fixed** | Explicit unit check: 3 overlapping synthetic detections → 2 after merge. **Found during a code-correctness audit:** `tile_detections_to_global()` built `box_geo` from raw pixel-corner indices (`ring_geo[0]`/`ring_geo[2]`), which silently breaks under any standard **north-up** GeoTIFF transform (negative pixel-height coefficient flips the y-axis, turning `(xmin,ymin,xmax,ymax)` into `(xmin,ymax,xmax,ymin)` and corrupting the IoU math) — cross-tile dedup never fired for real-world imagery even though every existing test kept passing, because the one assertion on this path (`raw >= final`) is trivially true when NMS removes nothing. Fixed by deriving true min/max from the ring's actual coordinates instead of positional indices; added a regression test (`test_cross_tile_dedup_merges_overlapping_boxes_under_north_up_transform`) that fails against the old code and passes against the fix. The original end-to-end pipeline test's own numbers changed from "20 raw → 20 after NMS" to "20 raw → 12 after NMS" once fixed — direct evidence dedup previously did nothing. |
| Standardized model interface (`load_model/validate_model/predict/postprocess/georeference/export_results`) | ✅ implemented & tested | Exercised end-to-end via an offline stub detector (`app/models/stub_predictor.py` — numpy-only, deterministic, **not** a real detector). |
| Compute device manager (CPU0/CPU1/GPU/AUTO) | ✅ implemented & tested | `resolve_device()` correctly falls back to CPU when no CUDA device is present (verified here, since this sandbox has no GPU). Dedicated `test_compute_offline.py` (7 cases) locks down every option (CPU0/CPU1/GPU/AUTO), the GPU→CPU1 fallback reason string, and `list_devices()` reporting zero GPUs honestly rather than guessing. |
| GeoJSON + manifest export | ✅ implemented & tested | Written files were parsed back and checked in the pipeline test. |
| **Car / Ship detection** (`generic_detector.py`) | ⚠️ real ultralytics code, **not run** | Wraps a stock YOLOv8-COCO model as a working placeholder (`car`/`boat` classes). Needs `pip install -r requirements.txt` + a GPU/CPU box with network access to actually run. Marked `VALIDATION` status, not `READY` — it is unfine-tuned and unbenchmarked for aerial imagery. |
| **Palm / Date / Tree / Field-boundary** models | ❌ not production-ready | No trained weights or labeled datasets exist. The factory and model classes raise a clear `NotProductionReadyError` instead of fabricating detections — per the spec's final requirement. |
| React + Leaflet dashboard | 📝 written, **not built/run** | No `npm install` was possible here (no registry access). Standard Vite/React/Leaflet code — run `npm install && npm run dev` in a normal dev environment. |
| Dataset module (YOLO/COCO/GeoJSON parsing, validation, class balance, geo-separated splitting, versioning) | ✅ implemented & tested | `backend/tests/test_datasets_offline.py`: GeoJSON round-trip, class-balance math, and a split test that asserts two spatial clusters are never divided across train/val/test. |
| Benchmark metrics (precision/recall/F1/mAP50/mAP50-95/throughput/FP-rate) | ✅ implemented & tested | `backend/tests/test_benchmark_offline.py`: hand-verified precision/recall on a constructed 2-image case, a perfect-prediction AP=1.0 case, and a regression test proving matching is scoped per-image (a prediction can't score against another image's ground truth just because the raw coordinates coincide). No real model has been benchmarked yet — see the table above. |
| YOLO directory dataset parsing (images + per-image `.txt` labels + `classes.txt`/`data.yaml`) | ✅ implemented & tested | `test_yolo_dataset_dir_parsing` writes real PNG files (via Pillow, which IS installed here) and YOLO labels to a temp dir and parses them back, including a label-less "background tile" case. |
| GeoJSON → KML export | ✅ implemented & tested | `test_export_formats_offline.py` round-trips Point and Polygon features through real `xml.etree` parsing, not just string checks, and verifies an unsupported geometry type is rejected. |
| ESRI Shapefile export (.shp/.shx/.dbf) | ✅ implemented & tested | `app/engine/shapefile_export.py`, hand-written from the public spec (no pyshp/fiona/GDAL). Verified two ways in `test_shapefile_export_offline.py`: a from-scratch independent reader round-trips Point and Polygon geometries + DBF attributes, **and** the `.shp` header's self-declared file length matches the actual file size on disk — a spec-derived invariant, not just "my reader agrees with my writer". Exposed as `GET /api/results/{job_id}/{model_id}/shapefile` (zipped). |
| CI (`.github/workflows/offline-tests.yml`) | ✅ added, config-validated | Runs all four offline test scripts on every push/PR with just numpy+Pillow; a second job installs the full `requirements.txt` as an import-smoke-test. Not yet run on GitHub itself since this was written from the sandbox — check the Actions tab after pushing. |
| Docker Compose (`docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile`) | 📝 written, **not built** | Standard Python-slim + nginx images; `docker build`/`docker compose up` wasn't possible here (no image-registry network access). GPU support is commented inline (needs the NVIDIA Container Toolkit on the host). |
| Field-boundary area/perimeter math (`app/engine/measurements.py`) | ✅ implemented & tested | `test_measurements_offline.py`: a known 100m×100m synthetic square mask (via real `cv2.findContours`, not a mock — opencv IS installed here) measures within ~1-2% of the hand-computed 10,000 sqm / 1.0 ha / 400 m. Wired through `FieldBoundarySegmenter.mask_to_results()`. `predict()` still correctly refuses to run (no trained segmentation weights) — this only makes the *postprocessing math* ready for when weights exist. |
| Per-model processing statistics (`app/models/stats.py`) | ✅ implemented & tested | Spec Section 7's "Processing statistics" field — run count, error count, totals, averages, last-run info — persisted as JSON and accumulated via `record_run()` (wired into `/api/predict`'s background job) and read via `get_stats()` (wired into `/api/models`). `test_stats_offline.py` verifies accumulation across multiple runs including an error case. |
| Model Management dashboard — Phase 4 (`ModelsDashboard.jsx`) | ✅ **built and verified live** | Originally written blind (no `npm install` possible in the sandbox). **Since confirmed on the user's real Windows machine**: `npm install && npm run dev` succeeded, the dashboard renders all 6 models with live processing statistics, and the compute-device dropdown and map toolbar (zoom-to-fit, basemap toggle, legend) added later all work as designed. |
| Backend API (FastAPI + full ML stack) | ✅ **running live on real hardware** | `pip install -r requirements.txt` succeeded on the user's Windows machine (Python 3.14, CPU-only) after installing GDAL-dependent packages (`rasterio`/`pyproj`/`geopandas`) separately from the pure-Python ones. `uvicorn app.main:app` serves real responses: `/api/models` returns the full catalog with correct honest statuses, `/api/compute` correctly reports zero GPUs and falls back to CPU. Verified directly through the API's own `/docs` Swagger UI, not just by the user's say-so. |
| Known Windows deployment gotchas (found on real hardware, not hypothetical) | ⚠️ documented | (1) Port 8000 can be blocked by Windows (`WinError 10013` — Hyper-V/WSL reserve port ranges); run with `--port 8010` or any free port, and set `VITE_API_PORT` in `frontend/.env` to match. (2) `uvicorn`/other console-script entry points may not be on `PATH` after `pip install`; use `python -m uvicorn ...` instead. (3) **PROJ database conflicts**: any other installed GIS tool (PostgreSQL/PostGIS, QGIS, Anaconda's own GDAL, ArcGIS) can set a system-wide `PROJ_LIB`/`PROJ_DATA` pointing at an incompatible `proj.db`, breaking every rasterio/pyproj CRS operation with a cryptic `CRSError` even though both packages import fine. `scripts/check_environment.py` now proactively checks for this (not just import success) and prints the fix. `backend/.env.example` documents the override, and `backend/app/core/config.py` now actually calls `load_dotenv()` (previously listed as a dependency but never wired in) so the `.env` override takes effect automatically. |
| Detection Models tab redesign — per-model status cards + Library Status panel | ✅ implemented | Replaced the single flat table with per-model cards (status dot, color-coded badge honestly distinguishing Ready/Unvalidated/Not-ready — fixed a bug where "Unvalidated" rendered the same green as "Ready") with expandable details: live GSD-compatibility check (`GET /api/models/{id}/gsd-check`) and that model's registered dataset versions (`GET /api/datasets/{id}/versions`) — both backend features existed already but were never wired into the UI. Added a "Model & Dataset Manager" panel wrapping the existing `POST /api/datasets/prepare` endpoint. On top, added a "Library Status" + "Installation Options" panel (`EnvironmentPanel.jsx`) mirroring a reference screenshot the user shared of a different QGIS plugin: a per-package install-status list plus Check Dependencies / Install Libraries / Repair Installation / Update Libraries actions. The dependency-check logic was extracted from `scripts/check_environment.py` into `backend/app/diagnostics/environment.py` so the CLI and the new `GET /api/environment` endpoint can never disagree. The three install/repair/update actions run a real `pip` subprocess in a background job (`POST /api/environment/actions` + polling `GET /api/environment/actions/{job_id}`) against `sys.executable -m pip install -r backend/requirements.txt`, with a "Build" selector (Automatic/CPU/CUDA) choosing PyTorch's published CPU or CUDA wheel index — no simulated progress, no faked "installed" state; a failed pip run reports its real exit code and stderr tail. |

**Bottom line:** the orchestration engine — the hard, easy-to-get-subtly-wrong
part (tiling, georeferencing, duplicate removal, the model contract) — is
real and has passing tests. The ML models themselves need your GPU box,
your datasets, and a normal internet connection to install dependencies and
train/fine-tune.

## Project layout

```
backend/
  app/
    core/        # config, compute device manager (CPU0/CPU1/GPU/AUTO)
    engine/       # raster I/O + tiling, geo transform, NMS, postprocess/export, pipeline orchestrator, benchmark metrics
    models/       # standardized model interface, catalog/registry, factory, per-model implementations
    datasets/     # Section 9: annotation parsing (YOLO/COCO/GeoJSON), validation, geo-separated splitting, versioning
    api/          # FastAPI routes: images, models, predict, results, compute, datasets, benchmark
    main.py
  tests/
    test_pipeline_offline.py        # numpy-only, runs anywhere, no GPU/network needed
    test_datasets_offline.py        # dataset parsing/validation/splitting (incl. real YOLO dir parsing)
    test_benchmark_offline.py       # metric-computation correctness (AP/mAP/precision/recall)
    test_export_formats_offline.py  # GeoJSON -> KML round-trip
    test_measurements_offline.py    # field area/perimeter math (real cv2 contour extraction)
    test_stats_offline.py           # per-model processing-statistics accumulation
    test_shapefile_export_offline.py  # .shp/.shx/.dbf round-trip + spec-derived file-length check
    test_compute_offline.py         # CPU0/CPU1/GPU/AUTO resolution + fallback-without-GPU behavior
  requirements.txt
  Dockerfile
frontend/
  src/
    components/   # TopBar (Detect/Detection Models tabs), Sidebar (model checklist + upload + run), MapView (Leaflet), AnalysisPanel, ModelsDashboard (Phase 4)
    api/client.js
  package.json
  Dockerfile
  nginx.conf
docs/
  API.md          # full REST API reference — routes, request/response shapes, error codes
scripts/
  check_environment.py  # dependency + CUDA doctor; run this on your real machine before `uvicorn`
docker-compose.yml
.github/workflows/offline-tests.yml
```

## Running it for real

```bash
# 0. Check what your machine actually has before installing anything
python scripts/check_environment.py
# Prints an honest pass/fail per dependency (fastapi/rasterio/torch/...),
# whether CUDA is visible to torch, and an exit code (0 = fully ready).

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev   # http://localhost:5173, proxies /api -> :8000

# Offline tests (no GPU/rasterio/torch needed)
cd backend
python tests/test_pipeline_offline.py
python tests/test_datasets_offline.py
python tests/test_benchmark_offline.py
python tests/test_export_formats_offline.py
python tests/test_measurements_offline.py
python tests/test_stats_offline.py
python tests/test_shapefile_export_offline.py
python tests/test_compute_offline.py

# Or with Docker Compose (not yet built/tested here — no registry access in this sandbox):
docker compose up --build
```

Full REST endpoint reference (request/response shapes, status codes, and
which checks are real vs. still to be exercised against a live server): see
[`docs/API.md`](docs/API.md).

## The six initial detection categories

See `backend/app/models/registry.py` for the full catalog (GSD compatibility
range, supported formats, status, and honest per-model notes on what's
missing). Status values: `ready`, `validation` (running but unbenchmarked),
`not_production_ready`, `loading`, `processing`, `training`, `error`,
`disabled` — matching the spec's Model Management Interface section.

## Roadmap (spec's Section 10, Development Priorities)

- [x] **Phase 1** — detection engine architecture: tiling, model interface, NMS, georeferencing, GeoJSON export. Working placeholder for car/ship; honest stubs for palm/date/tree/field-boundary pending datasets. Also includes the Section 9 dataset module (YOLO/COCO/GeoJSON parsing, validation, geo-separated splitting, versioning) a turn ahead of where the phase list put it, since training anything in Phase 1 needs it.
- [x] **Phase 2** — CPU0 / CPU1 / GPU / AUTO SELECT compute manager.
- [ ] **Phase 3** — validated at scale against real multi-GB orthomosaics (needs a real deployment box).
- [x] **Phase 4** — Model Management dashboard: full registry fields + live processing statistics in a dedicated table view, toggled from the top bar. Dataset-version UI (listing `/api/datasets/{model_id}/versions`) is not built yet.
- [x] **Phase 5** — GeoJSON → KML export (`GET /api/results/{job_id}/{model_id}/kml`), ESRI Shapefile export (`GET /api/results/{job_id}/{model_id}/shapefile`), and field area/perimeter measurement math (`app/engine/measurements.py`), all implemented and tested. Layer styling / measurement *tools* (interactive, in the dashboard) are still open — the underlying math and export formats are done; the UI affordances for them are not.
- [x] **Phase 6 (tooling only)** — `app/engine/benchmark.py` computes real precision/recall/F1/mAP50/mAP50-95/throughput/false-positive-rate from predictions + ground truth, unit-tested against hand-verified numbers. Nothing has actually been benchmarked yet — that still requires trained weights and a labeled held-out test set, which don't exist for any model in the catalog.

## What's needed from you to move past Phase 1

1. Labeled datasets (or existing weights) for palm, date-palm, tree, and
   field-boundary-polygon classes — YOLO/COCO/GeoJSON annotation formats are
   all parsed and validated by `app/datasets/` already; point it at real
   annotation files via `POST /api/datasets/prepare` to get a geographically
   separated train/val/test split and a versioned manifest.
2. A GPU-equipped host (or at least a normal networked machine) to install
   `requirements.txt` and actually run/benchmark the models.
3. Confirmation on the "reference" UI: the dashboard here is an **original**
   layout/branding inspired by the same category of product (map + model
   checklist + run button + results panel) — it does not reuse any other
   vendor's name, logo, or visual assets.
