# Syntra AI — Backend API Reference

Base URL (local dev): `http://localhost:8000`
All endpoints are prefixed `/api/...`. Request/response bodies are JSON
unless noted. This document describes the API as implemented in
`backend/app/api/*.py` — it is kept in sync with the code, not aspirational.

> This API cannot run in the build sandbox (no network to install
> `fastapi`/`rasterio`/`torch`/etc. — see `scripts/check_environment.py`).
> Everything below reflects the real route definitions; it has not been
> exercised against a live server.

---

## Models — `/api/models`

### `GET /api/models`
Returns the full six-model catalog (`app/models/registry.py`), each entry
merged with its live processing statistics (`app/models/stats.py`).

Response: array of
```json
{
  "model_id": "palm_tree_detection",
  "name": "...", "model_type": "detection|segmentation",
  "version": "0.0.0-unweighted",
  "supported_gsd_m": [0.25, 1.0],
  "supported_image_sources": [...],
  "supported_formats": [...],
  "status": "not_production_ready|validation|production",
  "hardware_compatibility": [...],
  "default_confidence_threshold": 0.5,
  "notes": "...",
  "processing_statistics": { "run_count": 0, "error_count": 0, "...": "..." }
}
```
`status` is the field the UI and `predict` endpoint both trust: a model
reporting `not_production_ready` has no trained weights, and the API will
refuse to run it rather than fabricate detections.

### `GET /api/models/{model_id}`
Same shape as one array entry above. `404` if `model_id` is unknown.

### `GET /api/models/{model_id}/gsd-check?gsd=<meters>`
Checks a requested ground-sample-distance against the model's supported
range (`supported_gsd_m`) and returns a compatibility report (degraded /
incompatible warnings included). `404` if `model_id` is unknown.

---

## Images — `/api/images`

### `POST /api/images` (multipart upload, field name `file`)
Accepts `.tif`/`.tiff` only (`400` otherwise). Stores the file under
`UPLOAD_DIR`, then — if `rasterio` is installed — reads real raster metadata
(width/height/bands/dtype/CRS/GSD). If `rasterio` is missing, returns a
`note` saying metadata is unavailable rather than guessing values.

Response:
```json
{ "image_id": "...", "filename": "...", "path": "...",
  "width": 0, "height": 0, "bands": 0, "dtype": "...",
  "crs": "...", "gsd_x_m": 0.0, "gsd_y_m": 0.0 }
```

### `GET /api/images`
Lists previously uploaded `.tif`/`.tiff` files with path and size.

---

## Predict — `/api/predict`

### `POST /api/predict`
Starts an asynchronous detection run (`BackgroundTasks`) for one model
against one already-uploaded image.

Request:
```json
{
  "image_path": "/path/to/uploaded.tif",
  "model_id": "car_detection",
  "device": "cpu0|cpu1|gpu|auto",
  "confidence_threshold": 0.5,
  "tile_size": 1024,
  "overlap": 0.2
}
```
- `409` if the model's status is `not_production_ready` — the response body
  explains why and includes the model's `notes`. This is intentional: the
  spec requires refusing fake detections rather than running an unweighted
  model.
- `404` if `image_path` does not exist on disk.
- On success: `{"job_id": "...", "status": "queued"}`.

The job runs the full pipeline (tile → predict → postprocess → georeference
→ merge/dedupe → export), writes `{model_id}.geojson` and
`{model_id}.manifest.json` under `RESULTS_DIR/{job_id}/`, and records a row
via `app/models/stats.record_run`.

### `GET /api/predict/{job_id}`
Polls job status: `queued` → `processing` (with `tiles_processed` count) →
`completed` (with the full `RunResult`) or `error` (with the exception
string). `404` for an unknown `job_id`.

> Job state is held in an in-memory dict (`_JOBS`) guarded by a `threading.Lock`.
> This is fine for a single-worker dev server; a multi-worker/production
> deployment needs to swap this for Redis or a database, since in-memory
> state is not shared across worker processes.

---

## Results — `/api/results`

All four endpoints read the GeoJSON a completed job already wrote to disk;
none of them re-run inference.

- `GET /api/results/{job_id}/{model_id}` → raw GeoJSON FeatureCollection.
- `GET /api/results/{job_id}/{model_id}/kml` → same detections as KML
  (`application/vnd.google-earth.kml+xml`), built with stdlib XML only.
- `GET /api/results/{job_id}/{model_id}/shapefile` → a zip containing
  `.shp`/`.shx`/`.dbf` (`application/zip`), built from scratch with
  `struct` (no `pyshp`/`fiona` dependency). `422` if the GeoJSON contains an
  unsupported geometry type (only Point and Polygon are implemented).
- `GET /api/results/{job_id}/{model_id}/manifest` → the run manifest
  (counts, timing, device used, model version).

All four return `404` if the job/model combination has no results on disk.

---

## Compute — `/api/compute`

### `GET /api/compute`
Device inventory backing the dashboard's GPU/CPU badge:
```json
{ "cpu0": {"threads": 4}, "cpu1": {"threads": 8}, "gpu": [], "gpu_count": 0 }
```
`gpu` is genuinely empty (not faked) on any host without a CUDA-visible
`torch` install — see `app/core/compute.py::_cuda_devices`.

---

## Datasets — `/api/datasets`

### `POST /api/datasets/prepare`
Parses an already-on-disk annotation set (YOLO directory, COCO JSON, or
GeoJSON), validates it, computes class balance, computes a **geographically
separated** train/val/test split (grid-cell binning, not random — avoids
spatial leakage), and registers a versioned snapshot.

Request:
```json
{
  "model_id": "palm_tree_detection",
  "annotation_format": "yolo|coco|geojson",
  "annotation_path": "/path/to/images_or_annotations",
  "images_dir": null,
  "labels_dir": null,
  "classes_file": null,
  "train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15,
  "cell_size_m": 1000.0
}
```
- `400` if COCO format is requested without `images_dir`, or if parsing
  fails (`AnnotationError`).
- `422` if `validate_dataset` finds structural problems (body includes the
  list of problems).
- Success:
```json
{
  "version_id": "...", "sample_count": 0,
  "class_balance": {...}, "underrepresented_classes": [...],
  "split": {"train": 0, "val": 0, "test": 0,
            "geographically_separated": true, "notes": "..."}
}
```

### `GET /api/datasets/{model_id}/versions`
Lists all registered dataset versions for a model.

### `GET /api/datasets/{model_id}/versions/{version_id}`
Full detail for one version. `404` if not found.

---

## Benchmark — `/api/benchmark`

### `POST /api/benchmark`
Computes real metrics (precision, recall, F1, mAP50, mAP50-95,
false-positive rate, per-class AP50, throughput) from predictions +
ground truth you submit — it does not invent numbers.

Request:
```json
{
  "model_id": "car_detection",
  "predictions": [{"image_id": "img1", "box": [x1,y1,x2,y2], "class_id": 0, "score": 0.9}],
  "ground_truth": [{"image_id": "img1", "box": [x1,y1,x2,y2], "class_id": 0}],
  "num_images": 1,
  "inference_times_ms": [42.0],
  "gpu_memory_mb": null
}
```
Matching is scoped per `image_id` (`app/engine/benchmark.py::_match_per_image`)
so a prediction cannot score a match against ground truth from a different
image. Response is the full `BenchmarkReport` dataclass as JSON.

---

## Error-handling conventions

- `404` — referenced entity (model/job/image/version) does not exist.
- `400` — malformed request (bad file type, missing required field, unparsable annotations).
- `409` — request is well-formed but refused on principle (predicting with a non-production-ready model).
- `422` — request parsed but failed validation (dataset problems, unsupported geometry for shapefile export).

None of these are placeholders — each is wired to a real check in the route
handler, listed above.
