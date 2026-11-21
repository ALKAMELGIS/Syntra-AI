# Architecture

Syntra is two programs on one computer: an **engine** (Python, FastAPI, port 8011) that reads
GeoTIFFs, runs models and keeps all data, and an **interface** (React + Leaflet, port 5291) that
calls the engine's `/api` and draws the map. Every source file is listed in [TREE.md](TREE.md).

```
browser ── interface (frontend/src) ──/api──▶ engine (backend/app)
                                                │
                     api/ ──▶ engine/, models/, training/, annotations/, datasets/
                                                │
                                core/ (config, security, jobs, layout, secrets)
                                                │
                    backend/data (images, labels, datasets, models, results)
                    backend/weights/packs (downloaded model packs)
```

## Engine: `backend/app/`

| Folder | Files | Responsibility | Depends on |
| --- | --- | --- | --- |
| `api/` | 20 | HTTP routes only: check the request, start a job, return JSON. No GIS or model logic | everything below |
| `engine/` | 30 | The work: tiling, georeferencing, NMS, exports, accuracy, refinement, remote sensing, RunPod, Hugging Face | `core`, `models` |
| `models/` | 16 | One interface for every detector (`base.py`), the catalog (`registry.py`), which runtime backs which id (`factory.py`), model packs | `core` |
| `training/` | 8 | Dataset building (tiles, splits, augmentation), the training worker, automatic parameters | `core`, `engine` |
| `annotations/` | 2 | Labels per image: storage, lon/lat ↔ pixel, importing labels from GIS files | `core` |
| `datasets/` | 4 | Parsing YOLO / COCO / GeoJSON, validation, geographic splits, versions | — |
| `core/` | 11 | Settings, security (paths, API key), accounts, the job store, the data layout, secrets | — |
| `diagnostics/` | 1 | Environment check used by Settings > System | — |
| `main.py` | | Creates the app and registers every router (protected by default) | |

Rule of direction: `api` calls down and nothing calls `api`; `core` imports nothing above it
(the one exception, `core/migrate.py`, is a one-time data move).
Heavy libraries (rasterio, torch, ultralytics, onnxruntime) are imported inside functions, so the
engine and its tests start without them.

## Interface: `frontend/src/`

| Folder | Holds |
| --- | --- |
| `main.jsx`, `App.jsx` | Start-up, and the project state shared by everything (image, models, AOI, results) |
| `api/client.js` | Every call to the engine |
| `lib/prefs.js` | Per-browser preferences (theme, accent, units, detection options) |
| `styles/tokens.css` | Design tokens: colours, type, radii, both themes. Use these, never raw values |
| `styles/app.css` | All component styles |
| `components/shell/` | Top bar and sidebar |
| `components/map/` | The map, its toolbar, drawing tools, layers panel, attribute table, basemaps |
| `components/images/` | Image library, This PC browser, raster check |
| `components/detection/` | Model list, results panel, review |
| `components/labels/` | Annotation tools, label import |
| `components/training/` | Train, model performance, sample library, Hugging Face |
| `components/remote-sensing/` | Sentinel-2 index panel |
| `components/settings/` | Settings window, Python environment |
| `components/auth/` | Sign-in screen and its globe |
| `components/ui/` | Icons, notes, error boundary |

## A detection run, end to end

1. Sidebar → `api.startPrediction` → `POST /api/predict` (`api/predict.py`).
2. The route resolves the image by id (never a path from the request), creates a job in
   `core/jobs.py` and runs it in the background.
3. `models/factory.py` builds the detector; tiles are read by `engine/raster_io.py` at the model's
   training scale when known, run, merged across tiles (`engine/nms.py`) and reprojected to WGS84.
4. The result is written to `data/results/<job>/<model>.geojson`; the interface polls the job and
   draws the layer.

## Where data lives: `backend/data/`

| Path | Holds |
| --- | --- |
| `images/<key>/` | Each GeoTIFF once, with its metadata and preview |
| `classes/<class>/labels/<key>.geojson` | The owner's labels: the source of truth for training |
| `classes/_areas/<key>.geojson` | Training and accuracy areas |
| `datasets/<class>/vNNN/` | Built datasets (`data.yaml`, `manifest.json`, tiles) |
| `models/<class>/vNNN/` | Training runs; `approved.json` picks the version detection uses |
| `results/` | Detection results |
| `syntra.db`, `secrets.json` | Jobs and accounts (SQLite), keys |
| `cache/sentinel/` | Sentinel Hub answers, so a scene is fetched once |

None of it is in git. `<key>` is the first 8 hex digits of the image id (short Windows paths).

## Rules every change keeps

- Results are never faked; a model without measured weights says so (README status table).
- Paths from requests go through `core/security.py`; new routers are registered protected.
- Long work is a job in `core/jobs.py`, never state in a module variable.
- Each fix has a test in `backend/tests/test_*_offline.py`; the suite runs without torch.
- New files start with one line saying what they do, then `python scripts/make_tree.py`.

Full house rules: [.claude/skills/syntra-dev/SKILL.md](../.claude/skills/syntra-dev/SKILL.md).
