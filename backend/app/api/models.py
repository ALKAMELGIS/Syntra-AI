"""GET endpoints for the Detection Models dashboard (Phase 4)."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.models.custom_registry import delete_custom_model, get_custom_model
from app.models.registry import get_metadata, list_catalog
from app.core.jobs import get_store
from app.models.packs import PACKS, is_installed, list_packs, pack_for_model, pack_path, verify_onnx
from app.models.stats import get_stats

router = APIRouter(prefix="/api/models", tags=["models"])


def _serialize(meta) -> dict:
    stats = get_stats(settings.DATA_DIR, meta.model_id)
    return {
        "model_id": meta.model_id,
        "name": meta.name,
        "model_type": meta.model_type,
        "version": meta.version,
        "supported_gsd_m": [meta.supported_gsd.min_m, meta.supported_gsd.max_m],
        "supported_image_sources": meta.supported_image_sources,
        "supported_formats": meta.supported_formats,
        "status": meta.status.value,
        "hardware_compatibility": meta.hardware_compatibility,
        "default_confidence_threshold": meta.default_confidence_threshold,
        "notes": meta.notes,
        "processing_statistics": stats.to_dict(),
        "pack": _pack_summary(meta.model_id),
    }


def _pack_summary(model_id: str) -> dict | None:
    pack = pack_for_model(model_id)
    if pack is None:
        return None
    return {"pack_id": pack.pack_id, "title": pack.title, "licence": pack.licence, "url": pack.url,
            "commercial_use": pack.commercial_use, "installed": is_installed(pack)}


# ---------------------------------------------------------------- model packs
INSTALL_KIND = "model_install"
MAX_PACK_MB = 600


def _get_pack(pack_id: str):
    pack = PACKS.get(pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail=f"Unknown model pack '{pack_id}'.")
    return pack


def _download(job_id: str, pack) -> None:
    import urllib.request

    jobs = get_store()
    dest = pack_path(pack)
    part = dest.with_suffix(dest.suffix + ".part")
    jobs.update(job_id, status="running")
    try:
        req = urllib.request.Request(pack.url, headers={"User-Agent": "Syntra-AI"})
        with urllib.request.urlopen(req, timeout=60) as r, part.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got, last = 0, 0
            while chunk := r.read(1 << 20):
                got += len(chunk)
                if got > MAX_PACK_MB * 1e6:
                    raise ValueError(f"Download exceeded {MAX_PACK_MB} MB.")
                f.write(chunk)
                if got - last > 2_000_000:
                    jobs.update(job_id, bytes=got, total_bytes=total or None)
                    last = got
        info = verify_onnx(part, pack.runtime)
        part.replace(dest)
        jobs.update(job_id, status="completed", bytes=got, total_bytes=total or got, model=info)
    except Exception as exc:
        part.unlink(missing_ok=True)
        jobs.update(job_id, status="error", url=pack.url, detail=str(exc),
                    error="Couldn't download. Get the file in a browser, then use From file.")


@router.get("/packs")
def get_packs():
    return list_packs()


@router.post("/packs/{pack_id}/install")
def install_pack(pack_id: str, background_tasks: BackgroundTasks):
    pack = _get_pack(pack_id)
    if is_installed(pack):
        return {"pack_id": pack_id, "status": "completed", "installed": True}
    job_id = get_store().create(INSTALL_KIND, pack_id=pack_id, url=pack.url, bytes=0)
    background_tasks.add_task(_download, job_id, pack)
    return {"job_id": job_id, "status": "queued"}


@router.get("/packs/install/{job_id}")
def install_status(job_id: str):
    job = get_store().get(job_id, INSTALL_KIND)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job


@router.post("/packs/{pack_id}/file")
async def install_pack_from_file(pack_id: str, file: UploadFile):
    """Offline install: the same .onnx the pack would download."""
    pack = _get_pack(pack_id)
    if not (file.filename or "").lower().endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Choose the model's .onnx file.")
    dest = pack_path(pack)
    part = dest.with_suffix(dest.suffix + ".part")

    def _save() -> int:
        n = 0
        with part.open("wb") as f:
            while chunk := file.file.read(1 << 20):
                n += len(chunk)
                if n > MAX_PACK_MB * 1e6:
                    raise ValueError("too big")
                f.write(chunk)
        return n

    try:
        size = await run_in_threadpool(_save)
        info = await run_in_threadpool(verify_onnx, part, pack.runtime)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Not a usable model file for {pack.title}: {exc}")
    part.replace(dest)
    return {"pack_id": pack_id, "installed": True, "bytes": size, "model": info}


@router.delete("/packs/{pack_id}")
def uninstall_pack(pack_id: str):
    pack = _get_pack(pack_id)
    path = pack_path(pack)
    from app.models.packs import packs_dir
    if path.exists() and path.parent.resolve() != packs_dir().resolve():
        # found in one of the user's own model folders: never delete their file
        raise HTTPException(status_code=409, detail=f"This model is loaded from {path.parent}. Remove the file there to uninstall it.")
    path.unlink(missing_ok=True)
    return {"pack_id": pack_id, "installed": False}


@router.get("")
def list_models():
    return [_serialize(m) for m in list_catalog()]


@router.get("/{model_id}")
def get_model(model_id: str):
    try:
        return _serialize(get_metadata(model_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/{model_id}")
def delete_model(model_id: str):
    """Only a model registered from a training run (app.models.
    custom_registry) can be removed here — the static catalog above is a
    fixed product category list, not user data."""
    if get_custom_model(settings.DATA_DIR, model_id) is None:
        raise HTTPException(
            status_code=404 if model_id not in {m.model_id for m in list_catalog()} else 409,
            detail=f"'{model_id}' is not a trained model you registered — nothing to delete here.",
        )
    delete_custom_model(settings.DATA_DIR, model_id)
    return {"deleted": model_id}


@router.get("/{model_id}/gsd-check")
def gsd_check(model_id: str, gsd: float):
    try:
        meta = get_metadata(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    from app.models.base import DetectionModel

    # gsd_compatibility_report is an instance method; build a throwaway shim
    class _Shim(DetectionModel):
        def load_model(self, device): ...
        def validate_model(self): return True, "shim"
        def predict(self, *a, **k): return []
        def export_results(self, *a, **k): return {}

    shim = _Shim()
    shim.metadata = meta
    return shim.gsd_compatibility_report(gsd)
