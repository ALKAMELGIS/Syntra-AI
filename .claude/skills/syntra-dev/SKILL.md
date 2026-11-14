---
name: syntra-dev
description: House rules for changing the Syntra AI codebase (FastAPI detection engine in backend/, React + Leaflet dashboard in frontend/). Use this whenever you add or change an API route, a model in the catalog, a training or dataset step, a UI panel or style in this repo, or review a pull request here, even for a small fix, so the change keeps the project's honesty, security and design conventions.
---

# Working on Syntra AI

Syntra AI turns satellite and drone GeoTIFFs into GIS layers: tile the raster,
run a detector per tile, georeference, merge duplicates across tiles, export
GeoJSON / KML / Shapefile. GIS analysts rely on what it outputs, so three
things matter more than speed of delivery: results are never faked, requests
can't reach outside the data directory, and the interface stays a quiet
instrument around the imagery.

## 1. Honesty about what works

The README status table is the contract with the user. Every row says what is
real, what is written but unrun, and how it was verified.

- A model without trained, benchmarked weights stays `NOT_PRODUCTION_READY`
  and its `build_model()` / `predict()` raises `NotProductionReadyError`. Never
  return placeholder detections to make a demo look complete; the user has
  explicitly rejected that.
- A model running on pretrained but unbenchmarked weights (the DOTA-OBB
  models for cars, ships, aircraft, oil tanks, bridges) is `VALIDATION`,
  shown as "Unvalidated" in the UI. Map new categories by DOTA class *name*
  in `DOTA_CLASS_MAP`, never by index.
- Never feed raw raster values to a model. Tiles go through
  `plan_rgb()` + `iter_tiles(rgb_plan=...)`: chosen RGB bands, scene-wide
  8-bit stretch, empty tiles skipped. New sensors with a known band layout
  go in `SENSOR_RGB` with a note on how the order was established.
- Anything new a GeoTIFF can get wrong belongs in `raster_health.py` as a
  finding with a level, a plain explanation and the fix.
- Exported geometry is WGS84 lon/lat. Do cross-tile merging in the raster
  CRS, then reproject (`reproject_detections_to_wgs84`). Never write
  projected coordinates into GeoJSON or KML.
- When you add or change a capability, update its README row: what changed,
  and how you verified it (test name, live check, or "not run, needs X").
- Never claim something ran if it did not. "Written, not built" is a fine
  status.

## 2. Security rules for the API

All of these live in `backend/app/core/security.py`. Use them; don't
reimplement them.

- **Never trust a path from a request body.** Prefer ids: images are
  addressed by `image_id`, resolved server-side with
  `find_entry_by_image_id(settings.UPLOAD_DIR, ...)`. If a field must be a
  path (dataset folders, `out_dir`, `data_yaml`), pass it through
  `confine_path(value, field="name", must_exist=...)`, which returns 403
  outside `SYNTRA_DATA_DIR` + `SYNTRA_EXTRA_DATA_ROOTS`.
- **Path parameters** that build file paths (`job_id`, `model_id`) must be
  checked to stay inside their root after `.resolve()`; see `_result_path`
  in `api/results.py`.
- **Uploads** go through `safe_filename()` and the streamed
  `SYNTRA_MAX_UPLOAD_MB` check in `api/images.py`.
- **New routers** are registered in `app/main.py` with
  `dependencies=_protected`, so the optional `SYNTRA_API_KEY` applies.
  Only `/api/health` stays open.
- **Anything that runs a subprocess on the server** (pip, training) must be
  gated like `environment_actions_allowed()`: loopback by default, explicit
  env flag otherwise.
- Long work runs as a background job stored with `get_store()` from
  `app/core/jobs.py` (create, update, append, get by kind), with a pollable
  status endpoint. Never keep job state in a module-level dict: it vanishes
  on restart. Give each job kind its own name, add it to `_KINDS` in
  `api/jobs.py`, and treat `interrupted` as a terminal state in the UI.
- Blocking I/O inside `async def` goes through `run_in_threadpool`.

## 3. Tests

- Tests live in `backend/tests/test_*_offline.py` and must pass with only
  `numpy`, `pillow`, `opencv-python-headless`, `fastapi`, `httpx`, `pytest`,
  `pyyaml`. Heavy deps (`rasterio`, `torch`, `ultralytics`) are lazy-imported
  inside functions so the app and tests import without them.
- Every bug fix gets a regression test that fails on the old code. Every new
  route that touches the filesystem gets a test that a path outside the data
  dir is refused.
- Run `python -m pytest -q backend/tests` before committing, and say how many
  passed.
- For the UI, `npx vite build` in `frontend/` must succeed. When you can,
  start the backend and `npx vite`, screenshot the changed screen, and look
  at it before calling the work done.

## 4. Interface conventions

The tokens are at the top of `frontend/src/styles.css`, with the design
direction in a comment. Work inside them:

- **Colour.** Graphite ramp (`--bg`, `--panel`, `--hover-bg`), three text
  roles (`--text`, `--text-dim`, `--text-muted`), one accent (`--accent`) for
  selection and the single number that matters. Primary actions use
  `--primary-bg` / `--primary-fg`. Every new colour must also work under
  `.theme-light`. No hue gradients, glows or coloured shadows. The colour
  belongs to the imagery and its detection layers, not the chrome.
- **Type.** IBM Plex Sans for UI, IBM Plex Mono with tabular figures for
  readouts (GSD, versions, counts, coordinates, file sizes). That mono
  readout is the product's signature, so use the `.mono` class for new
  figures. Weights 400-600, sentence case, no uppercase tracked labels, no
  exclamation marks.
- **Spacing.** 4px grid; list and nav rows 32px (40px for denser tables);
  panel padding 16px; controls `--radius-control` (8px), cards `--radius`
  (12px).
- **Status.** "Not ready" is quiet (outlined, muted) because most catalog
  entries are in that state; "Unvalidated" is amber; errors and missing
  dependencies are red.
- **Icons** come from `components/Icons.jsx` (inline SVG, one stroke). No
  emoji in the UI.
- **Map panels.** Map tools live in the one toolbar on the right. A tool
  with content (Edit, Legend, Analysis) opens a floating panel just left of
  the toolbar, aligned with its icon, never a fixed corner box or a layout
  column; one panel at a time; Esc closes it. Rows are 32 px: swatch, name,
  one mono value. Explanations go in tooltips, not on the panel.
- **Motion.** Name transitions (`background-color`, `color`, `transform`)
  with `var(--ease)`; no looping animation except a live spinner.

## 5. Shipping a change

1. Branch from `main`.
2. Make the change with its test, update the README status row and
   `.env.example` for any new setting.
3. Run backend tests and the frontend build.
4. Commit with a message that says what changed and how it was verified.
5. Open a pull request whose description lists what is verified and what
   still needs checking on the user's machine (GPU, real GeoTIFFs, Windows).
