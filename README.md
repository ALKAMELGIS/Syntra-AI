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

## Measurements and exports (phase 2, part 1)

Every result object now carries `area_m2` and `perimeter_m` (geodesic), `length_m`, `width_m` and
`orientation_deg` (smallest rotated rectangle, measured in a local equidistant projection, bearing
from north) and its centroid `lon`, `lat` (`engine/measure.py`, applied to detectors, segmentation
and vegetation). A 40 x 8 m rectangle at any bearing is measured within 5 cm and 0.5 deg.

Exports: GeoJSON, KML, Shapefile (now always with its `.prj`; it used to be left out for detector
runs, and the DBF only took the first object's fields), **GeoPackage** (written directly, no GDAL
needed; GDAL reads it back with its CRS) and **CSV** (attributes, lon, lat, WKT). Checked on the
harbour run: 175 boats with all measurements in each format.

Large uploads (>= 4096 px, stripped or without overviews) are rewritten in the background as
512 px tiles with overviews, like a COG (`engine/optimize.py`): a 162 MB 9000 x 6000 file took about
6.5 s and the image stays usable meanwhile.

## Imagery lands where it is

The map preview used to be the raster's own pixel grid stretched over its lat/lon bounding box. On a
web map that is wrong whenever the raster is not in Web Mercator: measured 7.5 m off on the 30 cm
UTM harbour scene (25 image pixels, enough to put boxes beside the boats), 165 m on a raster
rotated 15° and 460 m on a south-up raster (drawn upside down). The preview is now warped to Web
Mercator and placed by its exact corners (`preview_bounds_wgs84`). Zoomed in past the preview's
detail, full-resolution tiles are cut from the GeoTIFF (`/api/images/{id}/tiles/{z}/{x}/{y}.png`),
so a 20 000 px orthomosaic is as sharp on screen as on disk. Tests: a marker in north-up, rotated
and south-up rasters lands within 0.6 m in the preview and within 0.5 m in a zoom-20 tile.

The layout no longer scrolls the page (at 1280×720 it was 964 px tall and scrolled 244 px, hiding
the top bar), Settings is a map panel instead of a cut-off modal, the Edit menu closes when another
panel opens, and panels stay clear of the image library. Checked with an overlap audit of every
panel state at four screen sizes: no overlaps. When the satellite basemap can't load, a one-line
notice says so instead of a black map.

## Accuracy on your own images

Two area tools in the Train panel's Label step:

- **Training area**: label every object inside it; dataset tiles are cut only from inside training
  areas, so an object you did not label is never taught as background.
- **Accuracy area**: label every object inside it; it is kept out of training (labels inside it are
  dropped from the dataset and its pixels masked) and used to score models.

Model performance → *Check on this image* runs a model inside the accuracy areas and matches each
result to a label of the same class at IoU ≥ 0.5: found / wrong / missed, precision, recall, F1 per
class (`engine/accuracy.py`, `POST /api/predict/accuracy`). The latest F1 shows next to the model in
the sidebar. A one-class model is matched to one-class labels whatever the names (ship → boat).

Verified in the browser on the Dubai harbour scene: an accuracy area with 17 boats, labels for 15 of
them plus one object that is not there → exactly 15 found, 2 wrong, 1 missed (F1 0.91).

Two bugs found and fixed on the way (both changed results everywhere, not only in checks):

- Partial edge tiles were enlarged to fill the model input (a 576×432 tile went in 1.78× larger), so
  edge objects were seen at the wrong scale. Every runtime now scales a partial tile like a full one
  (`nominal_tile`); tests in `test_drone_seg_offline.py`.
- Area-limited runs used their own tile grid, so results inside an area differed from a whole-image
  run (F1 0.79 between the two). The area window is now aligned to the whole image's grid: identical
  results inside the area (17 of 17), and fewer tiles read. Test in `test_accuracy_offline.py`.

The Analysis panel has a *Min score* slider that filters results on the map and in the counts
without rerunning. The model list puts models suited to the image's resolution first; the others
are muted with the reason in their tooltip.

## Model performance

The chart icon in the map toolbar opens the Model performance panel: mAP50 on held-out images (the
best epoch), loss per epoch (training against validation, with a one-line reading of the gap),
validation precision / recall / mAP50 per epoch, and mAP50 per class (classes under 0.2 are dimmed
and not returned). Data: `results.csv` beside a model trained here, or
`app/models/pack_metrics/<pack_id>.json` for packs (`GET /api/models/performance`,
`GET /api/models/{id}/performance`). Tested in `test_performance_offline.py`; screenshots at
1440 px and 390 px.

## Drone objects (trained here, non-commercial)

`syntra-drone-objects` is YOLO11n-seg fine-tuned on the TU Graz Semantic Drone Dataset (340 photos,
60 held out, about 4 cm per pixel as the model sees them). The run was planned for 30 epochs and
stopped after 14; the weights are the best of those (epoch 12). Mask mAP50 on the held-out photos:

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
- **Data folder by class (plan stage 4b).** `SYNTRA_DATA_DIR` is now
  `images/<key>/<name>.tif` (one folder per image), `classes/<class>/labels/
  <key>.geojson` (the owner's labels, GeoJSON with real coordinates; the
  source of truth), `classes/_areas/<key>.geojson` (training and accuracy
  areas), `datasets/<class>/vNNN/` (YOLO layout: `images/` and `labels/`
  with train, val and test, plus `data.yaml` and `manifest.json`),
  `models/<class>/vNNN/` and `models/<class>/approved.json`
  (`app/core/layout.py`). An existing folder is moved once at start-up
  (`app/core/migrate.py`): every move is listed in `migration.json`, old label
  files are kept as `*.annotations.json.moved`. The split is now by area:
  whole blocks of tiles go to val, train tiles touching them are dropped, and
  accuracy areas are the test split with their own labels; a trained model is
  scored on test (`test_metrics.json`). On an image too small for a one-tile
  gap the gap is dropped and the manifest says so. Library (toolbar) shows
  each class: samples against its target, areas, dataset versions with
  train/val/test counts, models with their test or val score and
  Approve / Roll back; Build vNNN tiles every image labelled with the class;
  each version shows three steps, saved here -> backed up -> trained, and the
  next step is its only button: Upload to Hugging Face (tiles packed into tar
  shards of 2,000 files, plus the original GeoJSON labels), then Train here or
  on a RunPod GPU; a click on the version opens its split, notes and folders.
  Only the owner (the first, admin account) can build, back up, train and
  approve; other accounts see the library read-only (`is_owner`,
  `test_only_the_owner_builds_and_approves`). Verified: `test_layout_offline.py` (4,
  including the move of an old folder and a class built from two images with
  the same file name), `test_spatial_split_offline.py` (3: no train tile
  touches a val or test tile; accuracy areas become a labelled test split;
  small-image fallback), shard staging and the pod unpacking shards; live on
  this machine's data folder: the harbour image and its 14 labels moved, Build
  v001 for boat gave 10 train / 1 val / 5 test tiles. Not run: Upload and
  RunPod from the panel (no keys here), and a test score from a real training.
- **Approved model in detection.** Approving a version in the Library adds
  one detection model per class, `class_<class>` (e.g. "Boat", version
  "v001 approved"), to Detection models; Roll back removes it, and approving
  a newer version keeps the same id, so a saved selection runs what is
  approved now. It is "Unvalidated" until that version has a test score
  from your accuracy areas, then Ready with its F1 in the notes. Each run
  records the version it used, and a run already started keeps its version
  if another is approved meanwhile (`app/models/approved.py`). Verified:
  `test_approved_model_offline.py` (4), and live: an approved stand-in
  version shows as "Boat" in the list. Not run: detection with real trained
  class weights (none trained yet).
- **Label tools and click-to-segment (plan stage 4).** Train > Label has
  six tools, one key each: Click (A, AI), Polygon (P), Rectangle (R: drag,
  Shift for a square, or three clicks for a rotated rectangle), Square (S,
  Alt from the centre), Circle (C, saved as 64 points) and Freehand (F).
  Classes switch with 1-9; lengths show in metres while drawing; an exact
  size in metres places a rectangle, square or circle with one click.
  Shapes outside the image are refused. Click runs SAM 2.1 tiny (Apache 2.0)
  as two ONNX graphs on this computer: the visible view is encoded once and
  cached, each click only runs the decoder; Alt+click or right-click
  removes, Enter saves, Esc discards, Backspace undoes a click. The mask is
  then fitted to the class's shape (`app/engine/shapes.py`): buildings get
  right angles, panels and vehicles a rotated rectangle, tanks a circle,
  trees a smooth edge, with a quality gate that keeps the traced outline
  when the fit strays (IoU floor 0.85-0.90, area ratio 0.7-1.43). Verified:
  `test_shapes_offline.py` (9 tests; a rotated L-shaped roof comes back with
  6 corners of 90° at IoU > 0.96), `test_click_segment_offline.py` (6 tests,
  including two tiny ONNX graphs with the SAM 2.1 export's tensor names, a
  zip-slip check on the model download and view caching), live in the
  browser: all six tools saved labels on the Dubai harbour image, a 10 m
  exact square measured 33 px at 30 cm. On the real roofs (colour mask as a
  stand-in) single roofs went 479 -> 6 corners at IoU 0.97, merged roofs
  were refused by the gate. **Not yet run with the real SAM weights**:
  huggingface.co is blocked in the build environment; on your machine
  Click > Install click tool downloads about 120 MB once.
- **Labels from a file (Syntra labels v1).** Train > Label > Import adds
  polygons drawn in ArcGIS or QGIS as labels, without drawing. The file:
  Shapefile as a .zip (.shp .shx .dbf .prj), GeoPackage, GeoJSON, KML or
  FlatGeobuf, any CRS declared in the file. Fields: `class` (text,
  required; also read from class_name, classname, label, category, type
  or name, or one class for the whole file), `role` (optional: `object`,
  `complete` = training area, `accuracy` = accuracy area). Geometry is
  Polygon or MultiPolygon, one object each. Before anything is added the
  person sees the fields, picks the class field, renames or leaves out each
  value, and picks the role field. On import labels are clipped to the
  image, holes dropped, parts under 2 px skipped, and a label already on
  the image (same class, IoU >= 0.9) skipped, so the same file twice adds
  nothing; the report says how many of each were skipped. "Standard" shows
  the fields and downloads a template GeoJSON placed on the open image.
  (`app/annotations/importer.py`, `POST /api/images/{id}/annotations/import`
  and `/import/preview`). Verified: `test_label_import_offline.py` (2: a
  UTM Shapefile with Type and role fields, clipped, outside, no class,
  re-import duplicates; one class for a GeoJSON, a class map, MultiPolygon
  parts, a point refused), live: a 4-feature Shapefile zip, two values
  mapped to ship, 4 labels added.
- **Model performance follows training live.** A training run (on this
  computer or a RunPod GPU) appears in Model performance from its first
  epoch, marked as training, with epoch N of the planned total, a progress
  bar, and its loss and validation lines growing every 4 s on an axis that
  already spans the planned epochs. Hovering a chart reads out that epoch's
  values. Finished runs read `results.csv`; stopped or failed runs say why.
  The worker now reports each epoch after validation (`on_fit_epoch_end`),
  with summed train and val loss and mask metrics for segmentation, so an
  epoch's numbers are its own and not the previous epoch's. Verified:
  `test_training_live_perf_offline.py` (2), live with a simulated run whose
  epochs were written every 2.5 s. Not run with a real ultralytics training
  here (no GPU or dataset in the build environment).
- **Results on the map belong to the image shown.** After a reload the
  Layers list restores only the newest finished run per model on the image
  that is open; runs on other images or copies of it stay off the map.
  Remove on a results layer also hides the run on the server
  (`POST /api/jobs/{id}/hide`, files kept), so it does not come back after
  a reload. `GET /api/jobs` takes `image_id` and leaves hidden runs out
  unless `include_hidden=true`. Verified: `test_jobs_hide_offline.py`, live:
  two runs from another image did not appear, a removed layer stayed gone
  after a reload.
- **Refine outlines and measure accuracy.** A result layer's ⋯ menu has
  Refine outlines (SAM) and Measure accuracy. Refine sends every detected
  object to SAM 2.1 as a box prompt (one encode per 1024 px cell, the click
  tool's model), fits the class's shape (`shapes.py`), re-measures it and
  adds the result as a new layer "<name> · refined" beside the original
  (`app/engine/refine_results.py`, background job `refine`). An object keeps
  the detector's outline, marked `outline: detector` with the reason, when
  SAM's score is under 0.70, the trace overlaps the detection under IoU 0.30
  or its area is outside 0.15-1.6× the detection: refinement never invents
  a shape. Measure accuracy matches the layer one to one with the labels
  inside the image's accuracy areas (IoU ≥ 0.5, greedy) and shows F1,
  precision, recall and the mean outline IoU in the layer's Properties; it
  refuses without an accuracy area. Verified: `test_refine_results_offline.py`
  (3 tests: a stand-in SAM tightens a box and the layer is re-measured, an
  unsure SAM keeps the detector outline, 409 without SAM, path refusal,
  scoring 1 of 2 matched with an object outside the area ignored), live in
  the browser with the stand-in engine: refined layer added, Properties
  showed F1 80 %, precision 66.7 %, recall 100 %. **Not run with the real
  SAM weights or a trained model**: the 95-98 % target needs a trained
  class model and labelled accuracy areas, and this page is how to check it.
- **Train on a RunPod GPU (plan stage 3).** Settings > GPU training: paste
  a RunPod API key (checked, kept in `secrets.json`). Train > Where: RunPod
  GPU (RTX 4090 / 5090, L40S, A40; Community or Secure; a stop-after limit in
  hours). Needs the samples saved to Hugging Face first: the pod reads them
  from there. The pod runs the same `train_worker.py` as local training
  (plus `pod_main.py`), uploads progress every minute and the weights, ONNX
  and results.csv at the end, then leaves `done.json`. Syntra shows epochs,
  pod status and cost, downloads the model next to a local run's files (so
  Use / register / performance charts work unchanged) and terminates the pod.
  Three guards against a pod left running: the pod's `timeout`, its own
  terminate call, and Syntra's watcher with your key (resumed when the
  engine restarts; GPU runs are not marked interrupted). A restarted pod does
  not train twice. Built against the RunPod REST API v2 as documented on
  2026-11-15 (`POST /v2/pods` with `gpu`, `cloud`, `disk`, `env`, `args`;
  `POST /v2/pods/{id}/action` terminate). Verified by
  `test_cloud_train_offline.py` (8 tests: the real pod script run here
  against stand-in Hugging Face and Ultralytics packages; the watcher against
  a recording RunPod). Not yet run on a real pod: needs a funded RunPod
  account; whether the pod's own key may terminate it is not documented,
  which is why the watcher also terminates it.
- **Fixed: training always started from yolov8 detect weights,** whatever the
  dataset (OBB ships, segmentation). Base weights now follow the dataset's
  task: `yolo11{n,s,m}`, `-obb`, `-seg` (`test_base_weights_follow_the_dataset_task`).
- **Fixed: reviewed objects never reached the dataset.** Review areas cover
  under 1% of a tile, so the 35%-of-tile area rule and the 80%-black rule
  (which counted the masked part as black) dropped every tile: 0 tiles from 3
  accepted ships. Tiles with labels now pass the area rule, black is measured
  inside the area only, and rejected objects are written as background tiles
  on purpose. Live: 20 training tiles + 1 check tile from the same review
  (`test_review_reaches_the_dataset`, fails on the old builder).
- **Samples saved to Hugging Face (plan stage 2).** Settings > Training
  samples: paste a Write token once; the engine checks it, keeps it in
  `SYNTRA_DATA_DIR/secrets.json` and never sends it back. After Train >
  Build dataset, Save to Hugging Face uploads the YOLO export as one folder
  per push (`<date>_<image>_<task>/`, one git commit each) in a private
  dataset repo (default `<user>/syntra-samples`), with
  `syntra_source.json` (image, CRS, bounds, GSD, classes, counts). A new
  repo is created private; a push into a public repo is refused. Folders at
  the Hub's 10,000-file limit are refused before upload. Errors say whether
  the token or the network is at fault. Verified by
  `test_samples_hf_offline.py` (6 tests, Hugging Face replaced by a
  recorder). Not run against huggingface.co: this build machine's proxy
  blocks it, which the live check showed as "a proxy or firewall blocks
  huggingface.co".
- **Review results, then train on the review (plan stage 2).** Analysis >
  Review on a finished run lists its objects least sure first. Accept (A) or
  reject (R) each one; the map flies to it, outlined, clear of the panel; the
  list moves on by itself. Decisions are saved beside the result
  (`<model>.review.json`) and survive a refresh. Add to training turns them
  into the image's training data: an accepted object becomes a label inside a
  small training area, a rejected one an empty small area (a hard negative,
  "not a ship"). Training tiles are masked to their areas, so the unreviewed
  objects around them are never taught as background. Applying again replaces
  what the last apply wrote. Verified by `test_review_offline.py` (4 tests)
  and live on 175 ships detected in Dubai harbour (3 accepted, 1 rejected by
  keyboard -> 3 labels + 4 areas). Missed objects are still drawn in Train.
- **Faster Run, with honest timing (plan stage 1).**
  - Tiles are read on a background thread while the model runs
    (`prefetch`); reading was 28% of a run when done in line.
  - The GPU is found through onnxruntime, not torch: an NVIDIA card with
    `onnxruntime-gpu`, or any DirectX 12 card with DirectML, is now offered
    and used (TensorRT FP16 with an engine cache, then CUDA, then DirectML,
    then CPU). A card without GPU onnxruntime shows the one-line fix in
    `/api/compute` (`gpu_status`). Before, GPUs were only found when torch had
    CUDA, so most machines ran on the CPU whatever they had.
  - Above the Run button: area in km2 and, once the model has run on this
    computer, the expected time (median of its last five measured runs;
    never an invented figure). During a run: tiles done and time left from
    the run's own pace. Live check: "27 s left" at second 4 of a run that
    ended at second 29.
  - INT8 measured, not shipped as default: on Dubai harbour (0.3 m) a
    calibrated INT8 YOLO11n-OBB with the head kept in float ran 1.9x faster
    (191 -> 100 ms/tile on 2 cores) and found 96% of what the full model
    found. Fully quantising the head returned 0 detections.
  - `npm run lint` (part of `npm run build`) fails on undefined names: a
    missing `useEffect` import had blanked the page with a green build. Error
    boundaries keep a failing part from blanking the whole interface.
- **Open from this computer (This PC tab).** The engine runs on the same
  machine, so a multi-GB scene is not uploaded: the file is hard-linked into
  the library (instant, no extra disk space) or, across disks, copied by the
  file system. The original is never modified: anything that writes into a
  library file (overviews, a GCP transform) first breaks the link
  (`detach_if_linked`). Opening the same file again reuses its entry. Only
  requests from this computer can use it, inside `SYNTRA_IMPORT_ROOTS`
  (default: the home folder). Verified by `test_local_import_offline.py`
  (8 tests; the original-unchanged test fails when the detach is removed) and
  live: a 486 MB scene on the map 2.8 s after the click.
- **A refresh keeps your work.** The session (image, selected models, AOI,
  score filter), the map view per image and the image layer's opacity are
  kept in the browser; results and running jobs come back from the server.
  The preview is cached on disk with an ETag, so after a refresh the image is
  back in 0.4 s instead of being read again. File > Save project now saves
  this record (it used to save an empty `{}`).
- **Large scenes on the map.** Measured on a real SuperView Neo-3 MUX scene
  (10,806 x 10,395 px, 8 bands, uint16, 1.8 GB, stripped, no overviews) on a
  2-core VM: raster check 1.5 s; preview 7.6 s (was 118 s: it is now read
  decimated in the file's CRS and warped in memory, not warped from full
  resolution); background optimise 114 s; after it, tiles 0.2-0.7 s at
  z13-z16. The map stays on the preview while the optimise runs, then
  switches to full-resolution tiles. The scene stretch is computed once per
  file even when a view asks for 30 tiles at once (was 30 x 0.35 s).
  The Legend has the image as a layer: show/hide, opacity, ground resolution.
- **Masks uploaded instead of imagery.** A scene's cloud or quality mask
  (for example SuperView Neo `*_Cloud.tiff`) is named in the raster check
  as "No image data: this is a mask", with the file to upload instead;
  predictions on it are refused, and the map draws the file's footprint in
  red at its location. Verified by `test_mask_upload_offline.py` (4 tests)
  and a live upload of an all-zero EPSG:4326 mask.
- **Accounts.** The app opens on Sign in / Create account. The first account
  is the admin and can only be made from the computer running Syntra; after
  that anyone with the address can create an account unless
  `SYNTRA_ALLOW_SIGNUP=0`. Passwords are scrypt-hashed with a per-user salt;
  the session is an HttpOnly cookie whose token is stored only as a SHA-256
  hash in `syntra.db`. Five wrong passwords lock an email or address for ten
  minutes. `SYNTRA_AUTH=off` restores open access. Verified by
  `test_auth_offline.py` (9 tests) and a live browser run at four screen sizes.
- **Editing saved labels.** In the Label step with no tool on, clicking a
  saved shape selects it (it used to delete it): drag a corner, drag a
  midpoint to add a corner, Alt+click or right-click a corner to remove it,
  drag the shape to move it, the round handle to rotate it (Shift in 15°
  steps), 1–9 or a class row to change its class, Delete to remove it, Esc to
  finish. Each finished change is saved at once through
  `PATCH /api/images/{id}/annotations/{annotation_id}` (outline and / or
  class; id and creation time kept, pixel grid recomputed, shape marked
  "edited"); a refused change puts the old outline back. Verified:
  `test_annotations_store_offline.py` (2 new: update keeps id and moves
  class; bad or outside shapes refused and nothing changed) and live: select,
  drag a corner, one PATCH, saved as edited.
- **Layers, attribute table, pop-ups (like ArcGIS Online).** The Layers
  button (stacked sheets) replaces Legend and the hide-all eye. Each run is a
  layer with its own eye, colour, count and options: zoom to, show table,
  information (model and version, image, run time, CRS, extent, every field
  with its type and unit), colour, transparency, labels, pop-ups, filter by
  score, rename, move up / down, export GeoJSON, remove from map. The image
  layer has opacity and display contrast / brightness (the file is never
  changed; detection reads the original pixels); basemap Satellite, Streets
  or None. Clicking an object opens its pop-up with its real attributes and
  selects it; the attribute table docks under the map with a tab per layer,
  sort, search, map extent only, selected only, Ctrl / Shift selection
  outlined on the map, zoom to selection, export, and sum / mean of the
  sorted column. Verified live with sample result layers (panel, options,
  labels, table, pop-up, information) and `vite build`.
- **Clear image display, like ArcGIS.** The map, the preview and the
  gallery draw every image with a per-band percent clip (0.5–99.5 %,
  `raster_prep.DISPLAY_STRETCH`), as ArcGIS shows imagery by default; an
  8-bit photo whose values sat between 40 and 235 used to look grey and
  hazy. Display only: the detector still reads the stored values. Stretch
  choices in Properties: percent clip 0.5 / 2 / 5 %, min–max, none (as
  stored). Cached previews are redrawn once. Verified: a test that the
  display plan spreads 40..235 while the detector's plan stays 0..255, the
  full suite (317), and a before / after on Haven.tif.
- **Steady GeoTIFF display while zooming.** The image's preview now stays
  underneath at every zoom and the full-resolution tiles draw over it, so a
  tile still loading shows the preview instead of a hole and switching
  between the two never blanks the image. Tiles are not requested mid-zoom,
  a wider ring is kept around the view, and the server keeps rendered tiles
  in memory (96 MB, keyed by the file's size and change time) so returning
  to a place is instant. Verified: `test_tile_cache_offline.py` (2) and a
  live zoom-in / zoom-out burst over a 6000 px GeoTIFF, 56 screenshots
  mid-animation with the image complete in every one.
- **Transparency in the layer list.** Each layer's ⋯ list has a
  Transparency slider (image, results, services) that changes the map as it
  moves; the image row shows its transparency when it is not 0 %. A newly
  opened image starts fully opaque with no display changes; settings belong
  to the image they were made on. Verified live (a saved 30 % opacity from
  another image reset to 100 %, slider set 40 % with the list open).
- **Delete an image.** A small trash icon at the top left of each image in
  the gallery (on hover; always on touch screens) asks in the card, then
  deletes every copy of that image from `backend/data/images`
  (`DELETE /api/images/library/{folder}`). An image opened from your
  computer is a hard link there, so the original file stays; an image with
  labels is refused so samples are never lost. Verified:
  `test_image_delete_offline.py` (7: copy gone and original kept, labelled
  image refused, path tricks refused) and live (3 copies deleted).
- **Layer lists like ArcGIS Online.** Each layer's ⋯ opens a small floating
  list under the button (above it when there is no room): results layers get
  Zoom to layer, Show properties, Show table, Rename (in place), Duplicate,
  Export GeoJSON, Remove, Move up / down; the image gets Zoom to layer, Show
  properties and Remove (takes it off the map; the file stays in the
  library); map services get Show properties and Remove. Appearance, filter
  and display settings live on the layer's Properties page with its source
  and fields. Verified live: both lists, rename, duplicate, image removal,
  raster properties.
- **More layer controls.** Visible range per layer (from / to zoom). Image
  bands (Auto, or any R G B) and stretch (2–98 %, 0.5–99.5 %, 5–95 %,
  min–max) drawn by the tile server (`tiles/{z}/{x}/{y}.png?bands=4,3,2&stretch=0.5-99.5`)
  for the map only: detection keeps its own bands and 2–98 % stretch; a
  choice belongs to the image it was made for. Add > From a file: a
  Shapefile (.zip), GeoPackage, KML, GeoJSON or FlatGeobuf is read by the
  server (`POST /api/vector/read`, geopandas), reprojected to WGS84 and shown
  as a layer with the same options and attribute table; nothing is kept on
  the server. Add > Map service: XYZ tiles or WMS, for viewing. Verified:
  `test_vector_and_display_offline.py` (13: wrong type, zip pointing outside
  itself, empty zip, nothing left behind, UTM GeoJSON to WGS84, display
  parameters), a real 4-band GeoTIFF rendered with bands 4,4,4 and min–max,
  and live in the browser: a UTM Shapefile zip placed correctly with its
  attribute table, an XYZ service row, the range and band controls.
- **Image gallery.** The Images dock shows each image's own preview with a
  short name read from the file name (sensor and product, e.g. SuperView
  Neo-3 PAN), the capture date when the name carries one, resolution and
  area; copies of the same file show once with their count (×4); the full
  file name is only the card's native tooltip. Search by name, sensor or
  date; All / In view (footprint in the map view); sort; Gallery / List.
  Verified live against a copy of the user's 11 library entries (4 cards).
- **Sign-in globe.** The sign-in page shows a real Earth turning on the left
  (`components/EarthGlobe.jsx`): NASA Blue Marble on an orthographic sphere,
  lit from the upper left with NASA city lights on the night side, a satellite
  on its orbit, and the imaged place nearest the viewer named with its real
  coordinates. Textures are public domain, taken from the MIT `three-globe`
  package (`src/assets/earth/SOURCE.md`), and ship with the app, so it works
  offline. Still when the system asks for reduced motion; on a phone the globe
  sits above the form. Verified live: first-run and sign-in pages at
  1280 x 760 and 390 x 844, no page errors.
- `SYNTRA_API_KEY` turns on a shared key for every `/api/*` route except
  `/api/health`; set `VITE_API_KEY` in `frontend/.env` to match. This is a
  shared key for scripts and QGIS; people sign in with accounts (above).

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
