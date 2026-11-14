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
| **Car / Ship / Aircraft / Oil Tanks / Bridge** (`aerial_obb_detector.py`) | ⚠️ `VALIDATION`: aerial-pretrained, not benchmarked on your imagery | Backed by YOLO11-OBB pretrained on DOTA v1 (aerial and satellite scenes, rotated boxes), replacing the earlier COCO street-photo proxies. Exports rotated polygons. Cars split into `car` / `large_vehicle`, aircraft into `airplane` / `helicopter`; Oil Tanks uses DOTA's generic `storage tank` class, so it can include water or chemical tanks. Weights: `SYNTRA_AERIAL_OBB_WEIGHTS` (default `yolo11n-obb.pt`, auto-downloaded once). Verified offline by `test_aerial_obb_offline.py` (class mapping, rotated-polygon georeferencing) and **run for real**: YOLO11n-OBB on CPU over a 1920×1080 aerial marina scene georeferenced in UTM 40N found 178 ships (246 raw → 178 after cross-tile merge, 6 tiles, ~3 s), each as a tight rotated box on the boat, placed correctly on the map. |
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
| Full product redesign to match a reference screenshot ("IPSAR SAT") — Phase 6 | ✅ implemented & tested (offline) | The user shared a second reference screenshot of a different, more complete product and said "I mean this interface", then explicitly asked for everything in it: layout, all ~20 model categories "fully working", multi-GPU selection, a File menu, and a usage/credit indicator. Per the user's own choice when asked directly, the "GeoSyn $100.00" balance became a **real usage counter** (total prediction runs, derived from the same per-model stats already recorded — see `GET /api/usage`), explicitly **not** a payment/billing integration, since no payment provider is connected and fabricating a money balance would be exactly the kind of fake result this project refuses to produce elsewhere. Breakdown: **(1) Catalog expanded from 6 → 19 categories** (`app/models/registry.py`) matching the screenshot's list (Buildings, Vacant Land, Street centerline, Construction Waste, Aircraft, Unpaved Road, Walls, Oil Tanks, Oil Shadows, Tank Volume, Bridge, Change Detection, Containers) — honestly scoped: only `aircraft_detection` gets a working generic-COCO-proxy runtime (COCO has an 'airplane' class, same treatment as car/ship), every other new category is `NOT_PRODUCTION_READY` with notes on exactly what's missing (no COCO proxy exists for most of these, unlike car/ship/aircraft) rather than faked. New `backend/tests/test_registry_offline.py` locks this down: catalog size, uniqueness, and that every not-ready model's factory call raises `NotProductionReadyError` except the one documented exception (`field_boundary_segmentation`, whose refusal lives in `predict()`, not `build_model()`). **(2) Real multi-GPU selection**: `_cuda_devices()` already enumerated every physical GPU individually (this pre-existed); `resolve_device()` gained an `allowed_gpu_indices` filter (new regression tests in `test_compute_offline.py`), threaded through `DetectionEngine.run()` → `POST /api/predict`'s new `gpu_indices` field, driven by real per-GPU checkboxes in the top-bar GPU dropdown (`TopBar.jsx`) — unchecking a GPU actually excludes it from that device pool for new runs, it doesn't just change a label. **(3) TopBar redesign**: a working File menu (New Project / Open Project… / Save Project As…, all via real `localStorage` + downloaded/re-opened JSON — no account or server-side project store exists, so this is honestly scoped as local-only; Export Results — GeoJSON/KML/Shapefile wired to the existing real export endpoints), nav relabeled to "Full Map" / "Dashboard" matching the reference, a light/dark theme toggle (new `.theme-light` CSS variable override), and the usage counter described above. **(4) Sidebar redesign**: Detection Models switched from single-select radio to multi-select checkboxes (`Sidebar.jsx`) — selecting several models now starts one real prediction job per model (and per queued image in batch mode); added "Single file" / "Folder batch" tabs, the latter a real multi-file queue (`<input multiple webkitdirectory>`) that uploads and tracks each file's status independently. **(5) Right-side map toolbar** (`MapDrawTools.jsx`, new): rectangle/circle/freehand AOI drawing and a real two-point ruler (Haversine distance via Leaflet's own `distanceTo`) are genuine, working Leaflet draw tools — drawn and measured for real — but are explicitly **not yet wired into the backend as a processing-region restriction**; the AOI polygon is currently a map annotation only, and this file says so rather than implying a run is scoped to it. A "toggle detections visibility" eye button and a shortcut button that jumps to the Dataset & Model Manager are both fully wired. **(6) Multi-model/multi-image run + results**: `App.jsx` tracks one job per (image, model) pair, renders each as its own color-coded GeoJSON layer + legend entry on the map, and lists every job's own progress/result card in the Analysis panel — verified via `py_compile` (all backend files) + `pytest` (34/34 offline tests pass, up from 26) + `esbuild` (every changed/new frontend file parses clean) prior to deployment; **not yet verified live against the user's real running instance** (pending sync + browser check). |

**Bottom line:** the orchestration engine — the hard, easy-to-get-subtly-wrong
part (tiling, georeferencing, duplicate removal, the model contract) — is
real and has passing tests. The ML models themselves need your GPU box,
your datasets, and a normal internet connection to install dependencies and
train/fine-tune.

## Model packs: install open models on demand

Categories with open pretrained weights show under **Available to add** with
an **Install** button instead of "Not ready". Install downloads one ONNX file
from a fixed, reviewed URL into `SYNTRA_WEIGHTS_DIR/packs/`, checks it loads
and is the right kind of model, and the category becomes runnable. Offline
machines use **From file** with the same file. Clients only ever name a pack
id, never a URL. Chosen from the research report *Open models for satellite
detection* (Nov 2026):

| Category | Pack | Licence | Training data |
|---|---|---|---|
| Trees | DeepForest RetinaNet (GeoDeep ONNX) | MIT | NEON aerial, ~10 cm |
| Buildings | RAMP U-Net (GeoDeep ONNX) | CC BY-NC 4.0 (non-commercial) | ≤50 cm, 8 regions incl. Oman |

Tiles are read at the scale each model was trained at (from the ONNX
metadata), so a 10 cm model on 50 cm imagery sees upsampled tiles. Building
masks are merged across tiles before polygonising, so footprints cut by a
tile edge come out whole, with area in m².

The model list is grouped: **ready** (an amber dot means pretrained, not yet
checked on your imagery), **available to add**, and **needs training data**
(collapsed): palms, field boundaries, vacant land, roads, construction
waste, walls, change detection. The research found no open model for those
that is both usable at 0.3–0.7 m and licensed for commercial use.

Verified by `test_model_packs_offline.py` with synthetic ONNX models in the
same format. **Not yet run with the real weights**: Hugging Face is blocked
from the build sandbox; the first real install happens on your machine.

## Analyse a drawn area only

Draw a rectangle, circle or freehand area with the Edit tools; the sidebar
then shows **Drawn area only** and the button becomes **Run in area**. The
AOI (WGS84 GeoJSON) is transformed into the raster's CRS and clipped to it;
tiling reads only the AOI's bounding box, skips tiles outside the polygon,
and drops detections whose centre falls outside. An AOI that misses the
image is refused before the job starts. This is how a multi-GB scene is
analysed in seconds. Verified by `test_aoi_vegetation_offline.py`.

## Vegetation (spectral index, no trained model)

`vegetation_index` computes **NDVI** from red and near-infrared (VARI for
RGB-only images), outputs vegetation polygons with area (ha), mean index
and a dense/sparse class, and the total vegetation cover of the analysed
area. With the product XML uploaded beside the image, DN are calibrated to
radiance and divided by solar irradiance (ESUN) so NDVI is a ratio of
top-of-atmosphere reflectances. On the SuperView scene this moved bare sand
from −0.16 (radiance) to ~0.01, as expected physically.

Tree crowns are counted only at ≤ 1 m pixels and only as compact,
prominent peaks of the smoothed index; otherwise the count is withheld and
the manifest says why. This was tightened after the user's 2 m scene
produced ~10,000 false "trees" on one flat intertidal patch.

**Real run on the user's SuperView Neo-3 scene (2 m MUX, UTM 39N):** whole
scene 12,544 ha analysed, 201 ha vegetation (1.6%), 42 s on the user's
machine; lagoon area 1,315 ha, 160 ha vegetation (12%). Caveat recorded in
every manifest: on coasts, positive NDVI also comes from intertidal algal
mats and seagrass, which is what most of the lagoon cover is.

## Satellite products: raster check, 16-bit, band order, ONNX

Built after inspecting a real SuperView Neo-3 L2A scene (0.5 m PAN 3.6 GB,
2 m 8-band MUX, UTM 39N, 72% empty footprint):

- **Raster check on upload** (`GET /api/images/{id}/health`, shown in the
  sidebar): CRS and units, ground resolution in metres (converted for
  geographic CRSs), bit depth, band count and the RGB bands chosen, tiled vs
  stripped layout, overviews, RPC model present (not terrain-orthorectified),
  share of the frame with data. Each warning comes with the GDAL command that
  fixes it.
- **16-bit data** is stretched to 8-bit with one scene-wide 2–98% stretch
  (ignoring no-data), so every tile is treated the same. Previews use the
  same stretch and show the empty footprint as transparent.
- **Band order** follows GDAL colour interpretation, then band names, then
  the sensor (SuperView Neo-3 8-band: R,G,B = 3,2,1; WorldView 8-band: 5,3,2),
  then the band count. Override per run with `rgb_bands`.
- **Empty tiles are skipped** before inference.
- **ONNX backend**: set `SYNTRA_AERIAL_OBB_WEIGHTS` to a `.onnx` file and
  detection runs on onnxruntime with no PyTorch install. Export once with
  `yolo export model=yolo11n-obb.pt format=onnx imgsz=1024 opset=17`.
  Verified identical to ultralytics on the same tile (102/102 boxes, IoU 1.000).

Verified: `test_raster_prep_offline.py` (13 tests) and an end-to-end run on
an 8-band uint16 UTM GeoTIFF with RGB in bands 3/2/1 and an empty strip:
health ok, bands 3·2·1 picked from the sensor name, 2 empty tiles skipped,
181 ships, ONNX only.

## Train your own model, on the map

The Train icon in the map toolbar opens a five-step panel beside the map
(it replaced the separate Model setup page; the environment check is now in
Settings):

1. **Data**: upload a drone or satellite GeoTIFF (or pick one from the library) and name the classes.
2. **Label**: draw boxes or polygons on the map, over the base map and the image. Click a saved shape to delete it.
3. **Dataset**: tiles the image and labels into a YOLO dataset (boxes, rotated boxes or outlines), 20% held out.
4. **Train**: fine-tunes YOLO11 as a background job; the panel can be closed while it runs.
5. **Use**: adds the model to the sidebar list (Unvalidated) and runs it on the current image.

Verified end to end in the browser (Playwright): 3 labels → 20 tiles → 2 epochs → registered → run →
the result appears under Analysis. Training needs `ultralytics` + `torch` on the machine.

## Drone objects (trained here, non-commercial)

`syntra-drone-objects` is YOLO11n-seg fine-tuned on the TU Graz Semantic Drone Dataset (340 photos,
60 held out, about 4 cm per pixel as the model sees them). Mask mAP50 on the held-out photos:

| Class | mAP50 |
|---|---|
| Vehicles | 0.99 |
| People | 0.65 |
| Roofs | 0.55 |
| Trees | 0.44 |
| Fences, bicycles | < 0.2 (trained, not returned) |

It runs through onnxruntime (`models/onnx_seg.py`, no torch), reads tiles at its training scale, and is
refused with a 409 on imagery coarser than 22 cm. Checked on a 2 cm GeoTIFF: 6 outlines, WGS84, correct
place. One suburban Austrian site: check it on Gulf imagery before relying on counts. Dataset terms are
non-commercial.

## Works with any GeoTIFF: the 21-file matrix

`backend/tests/geotiff_matrix/variants.py` writes 21 GeoTIFFs that differ the
ways real products do. `test_geotiff_matrix_offline.py` sends each one
through upload, raster check, preview and a vegetation run via the API, and
checks the results are WGS84 lon/lat on the scene (it runs in CI).

| Variant | Result |
|---|---|
| uint8 RGB UTM, uint16 RGBN no-data 0, BGRN with an empty strip, int16 no-data −9999 | ✅ placed, analysed |
| float32 reflectance with NaN no-data | ✅ (crashed on upload before: NaN is not valid JSON) |
| RGBA alpha, internal GDAL mask over 255-filled pixels | ✅ masked pixels skipped |
| EPSG:4326, EPSG:3857, EPSG:2263 (US feet), south-up, rotated 15° | ✅ GSD converted to metres (0.50 m) |
| COG deflate, JPEG/YCbCr, 40 px tile, 1 px wide strip | ✅ |
| GCPs only, no geotransform | ✅ affine fit through the GCPs on upload (warped instead if RMS > 1 px) |
| RPC only | ✅ orthorectified onto local UTM on upload, raster check warns: no terrain model |
| Palette (classified) image | ✅ shown in its colours; vegetation refused, with the reason |
| Panchromatic single band | ✅ detection runs; vegetation refused (needs NIR or RGB) |
| No CRS at all | ✅ refused before running (409) with how to assign one, instead of exporting pixel coordinates |

The pixel rules are in one place, `raster_prep.valid_pixels` and
`read_rgb`: GDAL masks (no-data, alpha, internal mask), NaN, and an all-zero
fill when no no-data is declared. Tiling, preview, the raster check and
vegetation all use them. Georeferencing changes only the uploaded copy
(`engine/georef.py`). Ship detection (DOTA-OBB, ONNX) was also run on all 21
variants using a real marina photo as content; it found boats on every placeable
file. That run needs the model file, so it is not part of CI.

## Results are always WGS84

Detections are merged across tiles in the GeoTIFF's own projection (metres,
so overlap math is meaningful) and then reprojected to WGS84 lon/lat before
export, because GeoJSON, KML and Leaflet all expect it. Before this, results
from a projected GeoTIFF (UTM, which most orthomosaics use) did not land on
the map. Shapefile downloads now include a `.prj`. A raster with no CRS is
left in its own coordinates and the manifest says `"crs": null` rather than
guessing. Verified by `test_reprojection_offline.py` and the run above.

## Background jobs survive restarts

Prediction, dataset-export, training and pip jobs are kept in a SQLite file
(`SYNTRA_DATA_DIR/syntra.db`) instead of in memory. After a restart, finished
jobs and their results are still there, and jobs that were mid-run show as
**interrupted** with a message instead of looking live forever. The dashboard
reloads recent prediction jobs and their map layers on page load, and the
progress bar shows real tiles done out of the total. `GET /api/jobs` lists
history across all kinds. Verified by `backend/tests/test_jobs_offline.py`
(14 tests) and a live uvicorn restart.

Not yet: running jobs are not resumed (they restart from scratch), and jobs
still run in the API process. A separate worker queue (Celery/RQ + Redis) is
the next step for multi-user or multi-GPU servers.

## Security defaults

The API is safe to run on a single machine out of the box, and has switches
for when it is exposed to a network (all in `backend/.env.example`):

- Request paths (`image_path`, dataset folders, `out_dir`) must lie inside
  `SYNTRA_DATA_DIR`; add more folders with `SYNTRA_EXTRA_DATA_ROOTS`.
  `/api/predict` now takes `image_id` from the image library.
- Upload filenames are sanitised and capped by `SYNTRA_MAX_UPLOAD_MB`.
- Library Status install/repair/update (which runs `pip`) only works from
  the server itself unless `SYNTRA_ALLOW_ENV_ACTIONS=1`.
- `SYNTRA_API_KEY` turns on a shared key for every `/api/*` route except
  `/api/health`; set `VITE_API_KEY` in `frontend/.env` to match. This is a
  LAN gate, not per-user login, which is the next step.

Verified by `backend/tests/test_security_offline.py` (19 tests).

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

**Windows, one command** (creates the conda env on first run, then opens the browser):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-windows.ps1
# interface http://localhost:5291, API 8765; the next free port is used if one is taken
```

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
npm run dev   # http://localhost:5291 (VITE_PORT), proxies /api -> :8000 (VITE_API_PORT)

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
