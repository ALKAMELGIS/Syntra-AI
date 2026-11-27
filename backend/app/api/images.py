"""Source-image management: upload, list, and metadata inspection."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import is_loopback, safe_filename
from app.core.image_library import build_entry, find_entry_by_image_id, list_library, write_entry

router = APIRouter(prefix="/api/images", tags=["images"])

_RASTER_EXT = (".tif", ".tiff")


# ---------------------------------------------------------------------------
# Open from this computer: the engine runs on the user's own machine, so a
# multi-GB scene does not need to travel through the browser at all. The file
# is hard-linked into the library (instant, no extra space) or, across disks,
# copied by the file system. The original is never modified: anything that
# writes into a library file detaches the link first (detach_if_linked).
# ---------------------------------------------------------------------------
def import_roots() -> list[Path]:
    import os

    extra = [p for p in os.getenv("SYNTRA_IMPORT_ROOTS", "").split(os.pathsep) if p.strip()]
    roots = [Path(p).expanduser() for p in extra] or [Path.home()]
    return [r.resolve() for r in roots if r.exists()]


def _local_only(request: Request) -> None:
    if not is_loopback(request):
        raise HTTPException(403, detail="Opening files from disk works only on the computer running Syntra. Upload instead")


def _inside_roots(raw: str) -> Path:
    try:
        p = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError):
        raise HTTPException(400, detail="Not a valid path")
    for root in import_roots():
        if p == root or root in p.parents:
            return p
    raise HTTPException(403, detail="Outside the folders Syntra may open. Add the folder to SYNTRA_IMPORT_ROOTS")


@router.get("/local/browse")
def browse_local(request: Request, path: str | None = None):
    """Folders and GeoTIFFs in one folder under the import roots."""
    _local_only(request)
    roots = import_roots()
    if not path:
        if len(roots) == 1:
            path = str(roots[0])
        else:
            return {"path": None, "parent": None, "roots": [str(r) for r in roots],
                    "entries": [{"name": str(r), "path": str(r), "kind": "dir"} for r in roots]}
    folder = _inside_roots(path)
    if not folder.is_dir():
        raise HTTPException(404, detail="Folder not found")
    entries = []
    try:
        items = sorted(folder.iterdir(), key=lambda q: q.name.lower())
    except PermissionError:
        raise HTTPException(403, detail="Windows did not allow reading this folder")
    for q in items:
        if q.name.startswith((".", "$")) or q.name in ("AppData", "node_modules"):
            continue
        try:
            if q.is_dir():
                entries.append({"name": q.name, "path": str(q), "kind": "dir"})
            elif q.suffix.lower() in _RASTER_EXT:
                st = q.stat()
                entries.append({"name": q.name, "path": str(q), "kind": "file", "size_bytes": st.st_size,
                                "has_xml": q.with_suffix(".xml").exists()})
        except OSError:
            continue
        if len(entries) >= 1000:
            break
    entries.sort(key=lambda e: (e["kind"] != "dir", e["name"].lower()))
    parent = folder.parent if any(folder != r and r in folder.parents for r in roots) else None
    return {"path": str(folder), "parent": str(parent) if parent else None,
            "roots": [str(r) for r in roots], "entries": entries}


class LocalImport(BaseModel):
    path: str


@router.post("/local")
async def import_local(body: LocalImport, request: Request, background_tasks: BackgroundTasks):
    """Add a GeoTIFF from this computer to the library without uploading it."""
    import os
    import shutil

    _local_only(request)
    settings.ensure_dirs()
    src = _inside_roots(body.path)
    if not src.is_file() or src.suffix.lower() not in _RASTER_EXT:
        raise HTTPException(400, detail="Choose a GeoTIFF (.tif or .tiff)")
    st = src.stat()
    if st.st_size > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit (SYNTRA_MAX_UPLOAD_MB)")

    # the same file opened again (a refresh, a second click): reuse its entry
    for e in list_library(settings.UPLOAD_DIR):
        if e.get("source_path") == str(src) and e.get("source_mtime") == st.st_mtime and e.get("size_bytes") == st.st_size:
            return {**_fresh(_claim(e, request)), "reused": True}

    filename = safe_filename(src.name)
    image_id = str(uuid.uuid4())
    dest = _library_path(image_id, filename)

    def _bring(a: Path, b: Path) -> str:
        try:
            os.link(a, b)
            return "linked"
        except OSError:                       # another disk, or a file system without hard links
            shutil.copyfile(a, b)
            return "copied"

    how = await run_in_threadpool(_bring, src, dest)
    xml = src.with_suffix(".xml")
    if xml.is_file() and xml.stat().st_size <= 2 * 1024 * 1024:
        await run_in_threadpool(_bring, xml, dest.with_suffix(".xml"))
    entry = await _register(dest, filename, image_id, background_tasks)
    entry.update(source_path=str(src), source_mtime=st.st_mtime, imported=how)
    _save_dict(dest, entry)
    return entry


class FromBasemap(BaseModel):
    source: str = "usgs"                  # a key of basemap_capture.SOURCES, or "custom"
    url: str | None = None                # custom: an XYZ template with {z} {x} {y}
    bbox: list[float]                     # [west, south, east, north]
    zoom: int
    name: str | None = None


@router.get("/basemap-sources")
def basemap_sources():
    from app.engine.basemap_capture import MAX_TILES, SOURCES
    return {"sources": [{"id": k, **{x: v[x] for x in ("label", "max_zoom", "licence")}} for k, v in SOURCES.items()],
            "max_tiles": MAX_TILES}


@router.post("/from-basemap")
async def from_basemap(body: FromBasemap, request: Request, background_tasks: BackgroundTasks):
    """A basemap area saved as a GeoTIFF in the library, to label and train on."""
    import datetime

    from app.engine import basemap_capture as bc

    if len(body.bbox) != 4:
        raise HTTPException(422, detail="Give the area as [west, south, east, north]")
    try:
        if body.source == "custom":
            template = await run_in_threadpool(bc.check_url, body.url or "")
        elif body.source in bc.SOURCES:
            template = bc.SOURCES[body.source]["url"]
            if body.zoom > bc.SOURCES[body.source]["max_zoom"]:
                raise bc.CaptureError(f"This source goes to zoom {bc.SOURCES[body.source]['max_zoom']}")
        else:
            raise bc.CaptureError("Unknown source")
        bc.plan(tuple(body.bbox), body.zoom)
    except bc.CaptureError as exc:
        raise HTTPException(422, detail=str(exc))
    settings.ensure_dirs()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    filename = safe_filename(f"{(body.name or ('basemap-' + body.source)).strip()}-z{body.zoom}-{stamp}.tif")
    image_id = str(uuid.uuid4())
    dest = _library_path(image_id, filename)
    try:
        info = await run_in_threadpool(bc.capture, template, body.bbox, body.zoom, dest)
    except bc.CaptureError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, detail=str(exc))
    entry = await _register(dest, filename, image_id, background_tasks)
    entry.update(source=f"basemap:{body.source}", capture=info)
    _save_dict(dest, entry)
    return _fresh(_claim(entry, request))


def _claim(entry: dict, request: Request) -> dict:
    """Record who added the image: a user sees only their own images (the owner sees all)."""
    from app.core.auth import acting_user_id

    uid = acting_user_id(request)
    if uid is not None and uid not in (entry.get("owners") or []):
        entry["owners"] = [*(entry.get("owners") or []), uid]
        _save(entry)
    return entry


def _visible(entry: dict | None, request: Request) -> dict:
    from app.core.auth import can_see_any

    if entry is None or not can_see_any(request, entry.get("owners")):
        raise HTTPException(status_code=404, detail="No library image with this id.")
    return entry


@router.post("")
async def upload_image(file: UploadFile, background_tasks: BackgroundTasks, request: Request, metadata: UploadFile | None = None):
    """Upload a GeoTIFF, optionally with the product's metadata XML (e.g. the
    SuperView/WorldView sidecar). The XML carries calibration gains and the
    sensor id, which vegetation indices and band detection use."""
    settings.ensure_dirs()
    filename = safe_filename(file.filename)
    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only GeoTIFF (.tif/.tiff) uploads are supported.")

    image_id = str(uuid.uuid4())
    dest = _library_path(image_id, filename)
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    # Both the disk copy and the rasterio metadata reads below are blocking
    # (synchronous file I/O / C-extension calls). On a multi-GB GeoTIFF the
    # copy alone can run for a long time; running it directly in this `async
    # def` handler would stall FastAPI's single event loop for that whole
    # duration, freezing *every other* request the frontend is making in the
    # background (job/usage/environment polling) — which is what made the
    # whole app, not just the upload, look hung on a large upload. Running it
    # in a worker thread keeps the server responsive while the copy proceeds.
    def _copy() -> bool:
        written = 0
        with dest.open("wb") as f:
            while chunk := file.file.read(8 * 1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    return False
                f.write(chunk)
        return True

    if metadata is not None and metadata.filename:
        if not safe_filename(metadata.filename).lower().endswith(".xml"):
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Product metadata must be an .xml file.")
        xml_bytes = await metadata.read(2 * 1024 * 1024 + 1)
        if len(xml_bytes) > 2 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Product metadata XML is larger than 2 MB.")

    if not await run_in_threadpool(_copy):
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit (SYNTRA_MAX_UPLOAD_MB).",
        )

    if metadata is not None and metadata.filename:
        # stored next to the image under the same stem, where detect_sensor()
        # and read_calibration_gains() look for it
        dest.with_suffix(".xml").write_bytes(xml_bytes)

    return _claim(await _register(dest, filename, image_id, background_tasks), request)


# ---------------------------------------------------------------------------
# Resumable upload: a large GeoTIFF sent over the internet (as opposed to the
# This PC tab, which needs no network at all) can be interrupted by a dropped
# connection, a browser crash, or an accidental tab close. Re-sending the
# whole file from byte 0 every time is what actually made a slow link feel
# unusable, not anything about how the bytes are processed once they arrive.
#
# The key is derived from the file's own identity (name, size, last-modified)
# plus the acting user, not handed out and remembered by the browser: a File
# object cannot survive a reload anyway, so what "resuming" means in practice
# is that re-selecting the same file in the picker after a crash is recognised
# as the same upload and continues from the bytes already on disk, instead of
# starting over.
#
# Chunks land at whatever byte offset the client declares (X-Upload-Offset),
# not strictly in order: the part file is preallocated to its final size up
# front (a sparse file -- no disk cost for the holes) so several chunks can be
# written concurrently, each to its own non-overlapping range, which is what
# lets the frontend run multiple chunk uploads in parallel instead of one at
# a time. What bytes have actually arrived is tracked separately as a list of
# [start, end) ranges in the metadata file, merged as they come in; a chunk's
# own write and the ranges update that follows it are serialised per upload
# key (_resumable_lock_for) so two chunks finishing at the same instant can't
# corrupt each other's bookkeeping -- the file writes themselves, to disjoint
# ranges, need no lock.
# ---------------------------------------------------------------------------
_RESUMABLE_DIR_NAME = "_resumable"
_MAX_CHUNK_BYTES = 64 * 1024 * 1024       # one PUT; the frontend sends chunks well under this
_RESUMABLE_MAX_AGE_DAYS = 14               # an abandoned upload is swept away after this long

import threading

_resumable_locks: dict[str, threading.Lock] = {}
_resumable_locks_guard = threading.Lock()


def _resumable_lock_for(key: str) -> threading.Lock:
    with _resumable_locks_guard:
        lock = _resumable_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _resumable_locks[key] = lock
        return lock


def _merge_ranges(ranges: list[list[int]], start: int, end: int) -> list[list[int]]:
    """Insert [start, end) and coalesce with whatever already overlaps or touches it."""
    merged: list[list[int]] = []
    placed = False
    for s, e in sorted(ranges + [[start, end]]):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _received_bytes(ranges: list[list[int]]) -> int:
    return sum(e - s for s, e in ranges)


def _resumable_dir() -> Path:
    d = settings.UPLOAD_DIR / _RESUMABLE_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resumable_key(owner: int | None, name: str, size: int, modified: str) -> str:
    import hashlib

    raw = f"{owner if owner is not None else ''}:{name}:{size}:{modified}"
    return hashlib.sha256(raw.encode()).hexdigest()[:40]


def _resumable_paths(key: str) -> tuple[Path, Path]:
    import re

    if not re.fullmatch(r"[0-9a-f]{40}", key or ""):
        raise HTTPException(status_code=400, detail="Invalid upload key.")
    d = _resumable_dir()
    return d / f"{key}.part", d / f"{key}.json"


def _sweep_stale_resumables() -> None:
    import time

    cutoff = time.time() - _RESUMABLE_MAX_AGE_DAYS * 86400
    for meta in _resumable_dir().glob("*.json"):
        try:
            if meta.stat().st_mtime < cutoff:
                meta.unlink(missing_ok=True)
                meta.with_suffix(".part").unlink(missing_ok=True)
                with _resumable_locks_guard:
                    _resumable_locks.pop(meta.stem, None)
        except OSError:
            pass


class ResumableStart(BaseModel):
    name: str
    size: int
    modified: str = "0"


@router.post("/resumable")
async def resumable_start(body: ResumableStart, request: Request):
    """Begin a chunked upload, or report how far an already-started one got
    (as the list of byte ranges already on disk, so the frontend can skip
    them and only upload what is still missing)."""
    import json

    from app.core.auth import acting_user_id

    filename = safe_filename(body.name)
    if not filename.lower().endswith(_RASTER_EXT):
        raise HTTPException(status_code=400, detail="Only GeoTIFF (.tif/.tiff) uploads are supported.")
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if body.size <= 0 or body.size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit (SYNTRA_MAX_UPLOAD_MB).",
        )

    await run_in_threadpool(_sweep_stale_resumables)
    uid = acting_user_id(request)
    key = _resumable_key(uid, filename, body.size, body.modified)
    part, meta = _resumable_paths(key)
    if part.exists() and meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        ranges = info.get("ranges", [])
    else:
        def _create() -> None:
            with part.open("wb") as f:
                f.truncate(body.size)   # sparse file: the holes cost no disk space

        await run_in_threadpool(_create)
        ranges = []
        meta.write_text(
            json.dumps({"name": filename, "size": body.size, "owner": uid, "ranges": ranges}),
            encoding="utf-8",
        )
    return {"key": key, "size": body.size, "ranges": ranges}


@router.put("/resumable/{key}")
async def resumable_chunk(key: str, request: Request):
    """Write one chunk at the byte offset the client declares. Several chunks
    may be in flight at once (parallel PUTs to disjoint ranges); each is
    written independently and only the bookkeeping of which ranges have
    arrived is serialised, so this does not force uploads to go one at a
    time the way a strict append would."""
    import json

    part, meta = _resumable_paths(key)
    if not part.exists() or not meta.exists():
        raise HTTPException(status_code=404, detail="No upload in progress with this key. Start again.")

    try:
        offset = int(request.headers.get("x-upload-offset", "-1"))
    except ValueError:
        offset = -1
    if offset < 0:
        raise HTTPException(status_code=400, detail="Missing or invalid X-Upload-Offset header.")

    content_length = int(request.headers.get("content-length") or 0)
    if content_length <= 0 or content_length > _MAX_CHUNK_BYTES:
        raise HTTPException(status_code=413, detail=f"Each chunk must be under {_MAX_CHUNK_BYTES // (1024 * 1024)} MB.")

    chunk = await request.body()
    if len(chunk) != content_length:
        raise HTTPException(status_code=400, detail="Chunk body did not match its declared length.")

    lock = _resumable_lock_for(key)

    def _write_and_record() -> dict:
        with lock:
            info = json.loads(meta.read_text(encoding="utf-8"))
            if offset + content_length > info["size"]:
                raise HTTPException(status_code=400, detail="Chunk would exceed the file's declared size.")
            with part.open("r+b") as f:
                f.seek(offset)
                f.write(chunk)
            ranges = _merge_ranges(info.get("ranges", []), offset, offset + content_length)
            info["ranges"] = ranges
            meta.write_text(json.dumps(info), encoding="utf-8")
            received = _received_bytes(ranges)
            return {"received": received, "complete": received >= info["size"]}

    return await run_in_threadpool(_write_and_record)


@router.post("/resumable/{key}/finish")
async def resumable_finish(key: str, background_tasks: BackgroundTasks, request: Request, metadata: UploadFile | None = None):
    """All bytes are in: move the partial file into the library (same disk,
    so this is instant, not a second copy) and run the normal import."""
    import json

    part, meta = _resumable_paths(key)
    if not part.exists() or not meta.exists():
        raise HTTPException(status_code=404, detail="No upload in progress with this key.")
    info = json.loads(meta.read_text(encoding="utf-8"))
    ranges = info.get("ranges", [])
    received = _received_bytes(ranges)
    if received != info["size"] or (ranges and (len(ranges) != 1 or ranges[0] != [0, info["size"]])):
        raise HTTPException(status_code=409, detail=f"Upload is not complete yet ({received} of {info['size']} bytes).")

    with _resumable_locks_guard:
        _resumable_locks.pop(key, None)

    image_id = str(uuid.uuid4())
    dest = _library_path(image_id, info["name"])
    await run_in_threadpool(lambda: part.replace(dest))
    meta.unlink(missing_ok=True)

    if metadata is not None and metadata.filename:
        if not safe_filename(metadata.filename).lower().endswith(".xml"):
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Product metadata must be an .xml file.")
        xml_bytes = await metadata.read(2 * 1024 * 1024 + 1)
        if len(xml_bytes) > 2 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Product metadata XML is larger than 2 MB.")
        dest.with_suffix(".xml").write_bytes(xml_bytes)

    return _claim(await _register(dest, info["name"], image_id, background_tasks), request)


@router.delete("/resumable/{key}")
async def resumable_cancel(key: str):
    """The user discarded the upload: free the partial bytes right away
    instead of waiting for the stale-upload sweep."""
    part, meta = _resumable_paths(key)
    part.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)
    with _resumable_locks_guard:
        _resumable_locks.pop(key, None)
    return Response(status_code=204)


def _library_path(image_id: str, filename: str) -> Path:
    """images/<key>/<name>.tif: one folder per image (app.core.layout)."""
    from app.core.layout import image_key

    folder = settings.UPLOAD_DIR / image_key(image_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


async def _register(dest: Path, filename: str, image_id: str, background_tasks: BackgroundTasks) -> dict:
    """Everything after the file is in the library folder: georeference, read
    metadata, check, save the entry, start the background optimise."""
    raster_meta = None
    raster_meta_note = None
    preview_bounds = None
    bounds_wgs84 = None
    health = None
    georef = None
    try:
        from app.engine.georef import ensure_georeferenced
        georef = await run_in_threadpool(ensure_georeferenced, dest)
    except ImportError:
        pass
    except Exception as exc:
        georef = {"method": "failed", "error": str(exc)}

    try:
        from app.engine.raster_io import compute_wgs84_bounds, read_raster_info
        info = await run_in_threadpool(read_raster_info, dest)
        raster_meta = {
            "width": info.width, "height": info.height, "bands": info.count,
            "dtype": info.dtype, "crs": info.crs, "gsd_x_m": info.gsd_x, "gsd_y_m": info.gsd_y,
        }
        bounds_wgs84 = await run_in_threadpool(compute_wgs84_bounds, dest)
        from app.engine.raster_io import preview_bounds_wgs84
        preview_bounds = await run_in_threadpool(preview_bounds_wgs84, dest)
        from app.engine.raster_health import check_raster
        health = (await run_in_threadpool(check_raster, dest)).to_dict()
        from app.engine.georef import georef_finding
        if georef and georef.get("method") != "failed" and (finding := georef_finding(georef)):
            health["findings"].insert(0, finding)
            if finding["level"] == "warning" and health["status"] == "ok":
                health["status"] = "warning"
        if health.get("gsd_m") and not health.get("projected"):
            # geographic CRS: pixel size is in degrees; store the ground size in metres
            raster_meta["gsd_x_m"] = raster_meta["gsd_y_m"] = health["gsd_m"]
    except ImportError:
        raster_meta_note = "rasterio not installed in this environment — metadata unavailable until deployed with full requirements.txt"
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read raster: {exc}")

    entry = build_entry(
        image_id=image_id,
        filename=filename,
        path=dest,
        size_bytes=dest.stat().st_size,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        raster_meta=raster_meta,
        raster_meta_note=raster_meta_note,
        bounds_wgs84=bounds_wgs84,
        health=health,
        preview_bounds_wgs84=preview_bounds,
    )
    out = entry.to_dict()
    # Large stripped files or files without overviews are made fast to read in the
    # background (tiles + overviews); the image is usable meanwhile.
    # A file with no image data (a mask) is not worth retiling: nothing will run on it.
    empty = any(f.get("code") == "no_data" for f in (health or {}).get("findings", []))
    try:
        from app.engine.optimize import needs_optimizing
        plan = None if empty else await run_in_threadpool(needs_optimizing, dest)
    except Exception:
        plan = None
    if plan:
        from app.core.jobs import get_store
        job_id = get_store().create("optimize", image_id=image_id, progress=0.0, plan=plan)
        out["optimize_job_id"] = job_id
        background_tasks.add_task(_run_optimize, job_id, str(dest))
    entry_dict = out
    _save_dict(dest, entry_dict)
    return entry_dict


def _save_dict(dest: Path, entry: dict) -> None:
    import json
    from app.core.image_library import sidecar_path
    sidecar_path(dest).write_text(json.dumps(entry, indent=2), encoding="utf-8")


def _run_optimize(job_id: str, path: str) -> None:
    from app.core.jobs import get_store
    from app.engine.optimize import optimize

    jobs = get_store()
    jobs.update(job_id, status="processing")
    try:
        res = optimize(path, progress=lambda pct: jobs.update(job_id, progress=round(pct, 1)))
        # the raster check (layout, overviews) changed: redo it on next listing
        import json
        from app.core.image_library import sidecar_path
        sc = sidecar_path(Path(path))
        entry = json.loads(sc.read_text(encoding="utf-8"))
        if entry.get("health"):
            entry["health"]["version"] = 0
        entry.pop("optimize_job_id", None)
        sc.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        jobs.update(job_id, status="completed", result=res, progress=100.0)
    except Exception as exc:
        jobs.update(job_id, status="error", error=str(exc))


def _fresh(entry: dict) -> dict:
    """Bring a saved library entry up to date once, and save it: the Web Mercator
    preview bounds (missing on images uploaded before the preview was reprojected),
    and a raster check saved by an older version of the rules."""
    if entry.get("bounds_wgs84") and not entry.get("preview_bounds_wgs84"):
        try:
            from app.engine.raster_io import preview_bounds_wgs84
            entry = {**entry, "preview_bounds_wgs84": preview_bounds_wgs84(entry["path"])}
            _save(entry)
        except Exception:
            pass
    health = entry.get("health")
    if not health:
        return entry
    try:
        from app.engine.raster_health import HEALTH_VERSION, check_raster
    except ImportError:
        return entry
    if health.get("version") == HEALTH_VERSION:
        return entry
    try:
        new = check_raster(entry["path"]).to_dict()
    except Exception:
        return entry
    kept = [f for f in health.get("findings", []) if str(f.get("code", "")).startswith("georef")]
    new["findings"] = kept + new["findings"]
    if any(f["level"] == "warning" for f in kept) and new["status"] == "ok":
        new["status"] = "warning"
    entry = {**entry, "health": new}
    _save(entry)
    return entry


def _save(entry: dict) -> None:
    try:
        import json
        from app.core.image_library import sidecar_path
        sidecar_path(Path(entry["path"])).write_text(json.dumps(entry, indent=2), encoding="utf-8")
    except OSError:
        pass


@router.get("")
def list_images(request: Request):
    from app.core.auth import can_see_any

    settings.ensure_dirs()
    return [_fresh(e) for e in list_library(settings.UPLOAD_DIR) if can_see_any(request, e.get("owners"))]


@router.get("/{image_id}")
def get_image(image_id: str, request: Request):
    settings.ensure_dirs()
    return _fresh(_visible(find_entry_by_image_id(settings.UPLOAD_DIR, image_id), request))


@router.get("/{image_id}/health")
def get_image_health(image_id: str, request: Request):
    """Re-run the raster health check (CRS, resolution, bit depth, layout...)."""
    settings.ensure_dirs()
    entry = _visible(find_entry_by_image_id(settings.UPLOAD_DIR, image_id), request)
    try:
        from app.engine.raster_health import check_raster
        return check_raster(entry["path"]).to_dict()
    except ImportError:
        raise HTTPException(status_code=503, detail="rasterio not installed — health check unavailable in this environment.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read raster: {exc}")


_EMPTY_TILE = None


def _empty_png() -> bytes:
    global _EMPTY_TILE
    if _EMPTY_TILE is None:
        import io
        from PIL import Image
        b = io.BytesIO()
        Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(b, format="PNG")
        _EMPTY_TILE = b.getvalue()
    return _EMPTY_TILE


@router.delete("/library/{folder}")
def delete_library_copy(folder: str, request: Request):
    """Delete one copy of an image from the library (backend/data/images/<folder>).

    Only Syntra's own copy goes: an image opened from your computer is a hard
    link there, so your original file stays where it is. An image that has
    labels is refused, so samples are never lost by accident."""
    import re
    import shutil

    from app.annotations import store

    if not re.fullmatch(r"[0-9a-f]{8}|image", folder or ""):
        raise HTTPException(status_code=400, detail="Not a library folder.")
    root = settings.UPLOAD_DIR.resolve()
    target = (root / folder).resolve()
    if target.parent != root or not target.is_dir():
        raise HTTPException(status_code=404, detail="No such image in the library.")
    rasters = [p for p in target.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")]
    from app.core.auth import is_owner
    if not is_owner(request):              # a user deletes only an image no one else holds
        uid = __import__("app.core.auth", fromlist=["acting_user_id"]).acting_user_id(request)
        entries = [e for e in list_library(settings.UPLOAD_DIR) if Path(e["path"]).resolve().parent == target]
        if not entries or any((e.get("owners") or []) != [uid] for e in entries):
            raise HTTPException(status_code=404, detail="No such image in the library.")
    labelled = sum(len(store.load_annotations(p)) for p in rasters)
    if labelled:
        raise HTTPException(status_code=409, detail=f"This image has {labelled} label(s). Delete or move them first; the image was kept.")
    freed = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    shutil.rmtree(target)
    return {"deleted": folder, "freed_bytes": freed}


@router.get("/{image_id}/tiles/{z}/{x}/{y}.png")
def get_image_tile(image_id: str, z: int, x: int, y: int, request: Request, bands: str | None = None, stretch: str | None = None):
    """Full-resolution XYZ tile (Web Mercator, 256 px) cut from the GeoTIFF.

    Optional display choice for the map only: `bands=4,3,2` and
    `stretch=0.5-99.5` (percentiles). Detection never reads these."""
    if not (0 <= z <= 26 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise HTTPException(status_code=400, detail="Tile out of range.")
    try:
        from app.engine.tiles import parse_display
        disp_bands, disp_stretch = parse_display(bands, stretch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    entry = _visible(find_entry_by_image_id(settings.UPLOAD_DIR, image_id), request)
    try:
        from app.engine.tiles import cached_tile
        png = cached_tile(entry["path"], z, x, y, disp_bands, disp_stretch)
    except ImportError:
        raise HTTPException(status_code=503, detail="rasterio not installed.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    headers = {"Cache-Control": "private, max-age=3600"}
    return Response(content=png or _empty_png(), media_type="image/png", headers=headers)


@router.get("/{image_id}/preview.png")
def get_image_preview(image_id: str, request: Request):
    """A real, decimated true-color PNG of the raster for the map overlay —
    not a placeholder box. See app.engine.raster_io.read_preview_png.

    Cached on disk beside the library file and keyed by the file's size and
    modification time, so a page refresh shows the image at once instead of
    reading the scene again (7.6 s on a 1.8 GB scene), and a file that changed
    (after the background optimise) gets a fresh one."""
    import os

    settings.ensure_dirs()
    entry = _visible(find_entry_by_image_id(settings.UPLOAD_DIR, image_id), request)
    path = Path(entry["path"])
    try:
        st = path.stat()
    except OSError:
        raise HTTPException(status_code=404, detail="The image file is missing from the library folder.")
    etag = f'"{st.st_size:x}-{st.st_mtime_ns:x}-d2"'      # d2: display stretch; older previews are redrawn
    headers = {"ETag": etag, "Cache-Control": "private, no-cache"}      # revalidate: 304 when unchanged
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    cache = path.with_name(path.name + ".preview.png")
    tag = cache.with_name(cache.name + ".etag")
    try:
        if cache.exists() and tag.exists() and tag.read_text(encoding="utf-8") == etag:
            return Response(content=cache.read_bytes(), media_type="image/png", headers=headers)
    except OSError:
        pass
    try:
        from app.engine.raster_io import read_preview_png
        png_bytes = read_preview_png(path)
    except ImportError:
        raise HTTPException(status_code=503, detail="rasterio/Pillow not installed — preview unavailable in this environment.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not render preview: {exc}")
    try:
        tmp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
        tmp.write_bytes(png_bytes)
        os.replace(tmp, cache)
        tag.write_text(etag, encoding="utf-8")
    except OSError:
        pass                                    # read-only data folder: serve without caching
    return Response(content=png_bytes, media_type="image/png", headers=headers)
