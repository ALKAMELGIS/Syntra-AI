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
| Cross-tile duplicate suppression / NMS | ✅ implemented & tested | Explicit unit check: 3 overlapping synthetic detections → 2 after merge. Also exercised inside the full pipeline test. |
| Standardized model interface (`load_model/validate_model/predict/postprocess/georeference/export_results`) | ✅ implemented & tested | Exercised end-to-end via an offline stub detector (`app/models/stub_predictor.py` — numpy-only, deterministic, **not** a real detector). |
| Compute device manager (CPU0/CPU1/GPU/AUTO) | ✅ implemented & tested | `resolve_device()` correctly falls back to CPU when no CUDA device is present (verified here, since this sandbox has no GPU). |
| GeoJSON + manifest export | ✅ implemented & tested | Written files were parsed back and checked in the pipeline test. |
| **Car / Ship detection** (`generic_detector.py`) | ⚠️ real ultralytics code, **not run** | Wraps a stock YOLOv8-COCO model as a working placeholder (`car`/`boat` classes). Needs `pip install -r requirements.txt` + a GPU/CPU box with network access to actually run. Marked `VALIDATION` status, not `READY` — it is unfine-tuned and unbenchmarked for aerial imagery. |
| **Palm / Date / Tree / Field-boundary** models | ❌ not production-ready | No trained weights or labeled datasets exist. The factory and model classes raise a clear `NotProductionReadyError` instead of fabricating detections — per the spec's final requirement. |
| React + Leaflet dashboard | 📝 written, **not built/run** | No `npm install` was possible here (no registry access). Standard Vite/React/Leaflet code — run `npm install && npm run dev` in a normal dev environment. |
| Dataset module (YOLO/COCO/GeoJSON parsing, validation, class balance, geo-separated splitting, versioning) | ✅ implemented & tested | `backend/tests/test_datasets_offline.py`: GeoJSON round-trip, class-balance math, and a split test that asserts two spatial clusters are never divided across train/val/test. |
| Benchmark metrics (precision/recall/F1/mAP50/mAP50-95/throughput/FP-rate) | ✅ implemented & tested | `backend/tests/test_benchmark_offline.py`: hand-verified precision/recall on a constructed 2-image case, a perfect-prediction AP=1.0 case, and a regression test proving matching is scoped per-image (a prediction can't score against another image's ground truth just because the raw coordinates coincide). No real model has been benchmarked yet — see the table above. |

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
    test_pipeline_offline.py    # numpy-only, runs anywhere, no GPU/network needed
    test_datasets_offline.py    # dataset parsing/validation/splitting
    test_benchmark_offline.py   # metric-computation correctness (AP/mAP/precision/recall)
  requirements.txt
frontend/
  src/
    components/   # TopBar, Sidebar (model checklist + upload + run), MapView (Leaflet), AnalysisPanel
    api/client.js
  package.json
```

## Running it for real

```bash
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
```

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
- [ ] **Phase 4** — Model Management dashboard beyond the current checklist (model versioning UI, training status, hardware compatibility matrix).
- [ ] **Phase 5** — fuller GIS visualization/export (shapefile/KML export, layer styling, measurement tools).
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
