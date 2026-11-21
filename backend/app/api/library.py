"""The sample library, by class: what is labelled, which dataset versions and
models exist, which model is approved.

    GET  /api/library                         every class with samples, areas, datasets, models
    POST /api/library/{cls}/build             a new dataset version from every image labelled with it
    POST /api/library/{group}/approve         {"version": "v003"} or {"version": null}

Everything here is read from the folders (app.core.layout): what the panel
shows is what is on disk.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from app.core import layout
from app.core.config import settings
from app.core.jobs import get_store

router = APIRouter(prefix="/api/library", tags=["library"])


def _ring_km2(ring) -> float:
    if len(ring) < 3:
        return 0.0
    lat0 = math.radians(sum(p[1] for p in ring) / len(ring))
    k = 111.32 * math.cos(lat0)
    pts = [(p[0] * k, p[1] * 111.32) for p in ring]
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2


def _features(f: Path) -> list[dict]:
    return (layout.read_json(f, {}) or {}).get("features", [])


def _metrics(run: Path) -> dict:
    out: dict = {}
    csv_path = next(iter(sorted(run.rglob("results.csv"))), None)
    if csv_path:
        try:
            rows = list(csv.DictReader(open(csv_path, newline="")))
            keys = [k for k in (rows[0] if rows else {}) if k.strip().startswith("metrics/mAP50(")]
            if keys:
                out["val_map50"] = round(max(float(r[keys[0]]) for r in rows), 4)
                out["epochs"] = len(rows)
        except (OSError, ValueError, IndexError):
            pass
    test = next(iter(sorted(run.rglob("test_metrics.json"))), None)
    if test:
        out["test"] = layout.read_json(test, None)
    onnx = next(iter(sorted(run.rglob("best.onnx"))), None)
    out["weights"] = bool(onnx or next(iter(sorted(run.rglob("best.pt"))), None))
    return out


def _export_jobs() -> dict:
    return {(j.get("dataset_group"), j.get("dataset_version")): j["job_id"]
            for j in get_store().list("dataset_export", limit=500) if j.get("dataset_group")}


def library() -> dict:
    classes_root, ds_root, m_root = layout.classes_dir(), layout.datasets_dir(), layout.models_dir()
    area_files = {f.stem: _features(f) for f in sorted((classes_root / layout.AREAS).glob("*.geojson"))}
    jobs = _export_jobs()
    groups = sorted({p.name for root in (ds_root, m_root) if root.is_dir() for p in root.iterdir() if p.is_dir()})
    out = []
    for d in sorted(p for p in classes_root.iterdir() if p.is_dir() and p.name != layout.AREAS) if classes_root.is_dir() else []:
        meta = layout.read_json(d / "class.json", {}) or {}
        files = sorted((d / "labels").glob("*.geojson"))
        n = sum(len(_features(f)) for f in files)
        keys = [f.stem for f in files]
        roles = {"complete": [0, 0.0], "accuracy": [0, 0.0]}
        for k in keys:
            for feat in area_files.get(k, []):
                role = (feat.get("properties") or {}).get("role")
                if role in roles:
                    roles[role][0] += 1
                    roles[role][1] += _ring_km2(feat["geometry"]["coordinates"][0])
        mine = [g for g in groups if d.name in g.split("+")]
        datasets, models = [], []
        for g in mine:
            for v in layout.versions(ds_root / g):
                m = layout.read_json(v / "manifest.json", {}) or {}
                pushed = layout.read_json(v / "pushed.json", None)
                datasets.append({"group": g, "version": v.name, "created": m.get("created"), "counts": m.get("counts"),
                                 "split": m.get("split"), "samples": m.get("samples"), "images": len(m.get("sources") or []),
                                 "export_job_id": jobs.get((g, v.name)),
                                 "pushed": {"folder": pushed.get("folder"), "url": pushed.get("url"), "at": pushed.get("pushed_at")} if pushed else None})
            appr = layout.approved(g)
            for v in layout.versions(m_root / g):
                info = layout.read_json(v / "model.json", {}) or {}
                pushed_m = layout.read_json(v / "pushed.json", None)
                models.append({"group": g, "version": v.name, "dataset_version": info.get("dataset_version"),
                               "approved": v.name == appr, "pushed": pushed_m, **_metrics(v)})
        last = datasets[-1]["samples"] if datasets else None
        built_with = (last or {}).get(meta.get("name") or d.name) if isinstance(last, dict) else None
        out.append({
            "name": meta.get("name") or d.name, "slug": d.name, "shape": meta.get("shape"), "target": meta.get("target", 1500),
            "samples": n, "images": len(files),
            "areas": {r: {"count": c, "km2": round(a, 3)} for r, (c, a) in roles.items()},
            "new_since_last": None if built_with is None else max(0, n - int(built_with)),
            "datasets": datasets, "models": models,
        })
    return {"classes": out, "data_dir": str(settings.DATA_DIR)}


@router.get("")
def get_library(request: Request):
    from app.core.auth import is_owner
    return {**library(), "owner": is_owner(request)}


class BuildRequest(BaseModel):
    task: str = "detect"
    tile: int = 640
    overlap: int = 64
    val_split: float = 0.2
    neg_ratio: float = 0.1
    augment: bool = True


def _image_for_key(key: str) -> dict | None:
    from app.core.image_library import list_library

    for e in list_library(settings.UPLOAD_DIR):
        if Path(e["path"]).parent.name == key and e.get("image_id"):
            return e
    return None


def _parts_for_class(slug: str) -> tuple[str, list[dict]]:
    """One build request per image that has labels of this class."""
    from app.annotations import store

    d = layout.classes_dir() / slug
    meta = layout.read_json(d / "class.json", {}) or {}
    name = meta.get("name") or slug
    parts = []
    for f in sorted((d / "labels").glob("*.geojson")):
        entry = _image_for_key(f.stem)
        if not entry:
            continue
        anns = [a for a in store.load_annotations(entry["path"]) if a["class_name"] == name or store.is_area(a)]
        objs = [a for a in store.objects_for_training(anns)]
        if not objs:
            continue
        parts.append({
            "image_id": entry["image_id"], "image_path": entry["path"], "key": f.stem,
            "annotations": [{"class_id": 0, "points_px": a["points_px"]} for a in objs],
            "train_areas_px": [a["points_px"] for a in store.areas(anns, store.TRAIN_AREA)] or None,
            "exclude_areas_px": [a["points_px"] for a in store.areas(anns, store.CHECK_AREA)] or None,
            "test_areas_px": [a["points_px"] for a in store.areas(anns, store.CHECK_AREA)] or None,
            "test_annotations": store.objects_for_test(anns, [name]) or None,
            "negative_areas_px": [a["points_px"] for a in store.areas(anns, store.TRAIN_AREA)
                                  if (a.get("source") or {}).get("decision") == "reject"] or None,
            "samples": len([a for a in anns if a["class_name"] == name]),
        })
    return name, parts


def _merge(part_dir: Path, out: Path) -> None:
    for sub in ("images", "labels"):
        for split_dir in (part_dir / sub).glob("*"):
            dest = out / sub / split_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_dir.iterdir():
                shutil.move(str(f), str(dest / f.name))


def _run_class_build(job_id: str, slug: str, name: str, parts: list[dict], req: BuildRequest) -> None:
    import numpy as np

    from app.api.training import area_filter, write_manifest
    from app.training.dataset_builder import DatasetBuilder, write_data_yaml
    from app.training.raster_window import RasterReader

    jobs = get_store()
    jobs.update(job_id, status="processing")
    group = slug
    version, out = layout.next_version(layout.datasets_dir() / group)
    jobs.update(job_id, dataset_group=group, dataset_version=version, out_dir=str(out))
    try:
        total: dict = {}
        index_rows, splits = [], []
        for k, part in enumerate(parts):
            if (jobs.get(job_id) or {}).get("cancel_requested"):
                raise RuntimeError("Canceled")
            reader = RasterReader(part["image_path"])
            tf, bounds = area_filter(part["train_areas_px"], part["exclude_areas_px"], req.tile, reader.width, reader.height)

            def progress(pct, message=None, k=k):
                jobs.update(job_id, progress=round((k + pct / 100) / len(parts) * 95, 1),
                            **({"message": f"Image {k + 1} of {len(parts)}: {message}"} if message else {}))

            b = DatasetBuilder(
                reader, [{"cls": 0, "pts": np.asarray(a["points_px"], float)} for a in part["annotations"]], [name],
                str(out / "_parts" / part["key"]), tile=req.tile, overlap=req.overlap, val_split=req.val_split,
                neg_ratio=req.neg_ratio, task=req.task, augment=req.augment, tile_filter=tf, area_bounds=bounds,
                negative_areas=part["negative_areas_px"], test_areas=part["test_areas_px"],
                test_annotations=[{"cls": 0, "pts": np.asarray(a["points_px"], float)} for a in (part["test_annotations"] or [])],
                progress=progress, log=lambda m: jobs.append(job_id, "log", m, max_len=500))
            b.prefix = part["key"]                       # tile names stay unique across images
            _, info = b.run()
            reader.close()
            _merge(out / "_parts" / part["key"], out)
            idx = out / "_parts" / part["key"] / "tiles_index.csv"
            if idx.exists():
                index_rows += list(csv.reader(open(idx, newline="")))[1:]
            for key, val in info["stats"].items():
                if isinstance(val, (int, float)):
                    total[key] = total.get(key, 0) + val
            splits.append(info.get("split"))
        shutil.rmtree(out / "_parts", ignore_errors=True)
        total["class_counts"] = {name: total.get("train_objects", 0) + total.get("val_objects", 0)}
        has_test = any((out / "images" / "test").glob("*")) if (out / "images" / "test").is_dir() else False
        for split in ("train", "val") + (("test",) if has_test else ()):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        write_data_yaml(str(out / "data.yaml"), str(out), [name], req.task, test=has_test)
        with open(out / "tiles_index.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "split", "augmentation", "x_offset_px", "y_offset_px", "objects", "black_ratio"])
            w.writerows(index_rows)
        info = {"created": __import__("time").strftime("%Y-%m-%d %H:%M:%S"), "classes": [name], "task": req.task,
                "tile_size": req.tile, "overlap": req.overlap, "stats": total,
                "split": splits[0] if len(splits) == 1 else {"per_image": splits}}
        (out / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
        req_like = type("R", (), {"image_path": parts[0]["image_path"]})()
        manifest = write_manifest(out, group, version, req_like, info)
        manifest["sources"] = [{"image": p["key"], "file": Path(p["image_path"]).name} for p in parts]
        manifest["labels"] = [lf for p in parts for lf in _label_hashes(p["key"])]
        manifest["samples"] = {name: sum(p["samples"] for p in parts)}
        gsds = []
        for p in parts:                                         # the scale the model will learn: detection matches it
            try:
                from app.engine.segmentation_run import ground_gsd_m
                g = ground_gsd_m(p["image_path"])
                if g:
                    gsds.append(g)
            except Exception:  # noqa: BLE001 - no CRS or no rasterio: the scale stays unknown
                pass
        if gsds:
            manifest["gsd_m"] = round(sorted(gsds)[len(gsds) // 2], 4)
        layout.write_json(out / "manifest.json", manifest)
        jobs.update(job_id, status="completed", progress=100.0, data_yaml=str(out / "data.yaml"), info=info,
                    class_names=[name], task=req.task)
        try:                                                   # Hugging Face backup, when turned on
            from app.api.samples import auto_backup
            push_id = auto_backup(job_id)
            if push_id:
                jobs.update(job_id, push_job_id=push_id)
        except Exception as exc:  # noqa: BLE001 - the dataset is built; the backup can be retried by hand
            jobs.update(job_id, push_error=str(getattr(exc, "detail", exc)))
    except Exception as exc:  # noqa: BLE001 - shown to the owner as it is
        shutil.rmtree(out, ignore_errors=True)
        jobs.update(job_id, status="error", error=str(exc))


def _label_hashes(key: str) -> list[dict]:
    import hashlib

    root = layout.classes_dir()
    files = sorted(root.glob(f"*/labels/{key}.geojson")) + sorted(root.glob(f"{layout.AREAS}/{key}.geojson"))
    return [{"file": str(f.relative_to(root)).replace("\\", "/"), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]


def _owner(request: Request):
    from app.core.auth import require_owner
    require_owner(request)


@router.post("/{slug}/build", dependencies=[Depends(_owner)])
def build(slug: str, req: BuildRequest, background_tasks: BackgroundTasks):
    if slug != layout.class_slug(slug) or not (layout.classes_dir() / slug).is_dir():
        raise HTTPException(status_code=404, detail="No such class in the library")
    if req.overlap >= req.tile or req.task not in ("detect", "obb", "segment"):
        raise HTTPException(status_code=400, detail="Check the tile, overlap and task")
    name, parts = _parts_for_class(slug)
    if not parts:
        raise HTTPException(status_code=409, detail="No labelled objects of this class on any image yet")
    job_id = get_store().create("dataset_export", progress=0.0, task=req.task, class_names=[name],
                                image_id=parts[0]["image_id"], image_name=f"{len(parts)} image(s)")
    background_tasks.add_task(_run_class_build, job_id, slug, name, parts, req)
    return {"job_id": job_id, "status": "queued", "images": len(parts)}


class ApproveRequest(BaseModel):
    version: str | None


@router.post("/{group}/approve", dependencies=[Depends(_owner)])
def approve(group: str, body: ApproveRequest):
    if not all(part == layout.class_slug(part) for part in group.split("+")):
        raise HTTPException(status_code=404, detail="No such model group")
    if body.version is not None:
        if body.version not in {p.name for p in layout.versions(layout.models_dir() / group)}:
            raise HTTPException(status_code=404, detail="No such model version")
        if not _metrics(layout.models_dir() / group / body.version).get("weights"):
            raise HTTPException(status_code=409, detail="This version has no trained weights yet")
    layout.set_approved(group, body.version)
    return {"group": group, "approved": layout.approved(group)}


# --- a model version to Hugging Face (weights, curves, and the data it learned from) ---
MODEL_PUSH_KIND = "model_push"


class ModelPushRequest(BaseModel):
    version: str
    repo: str | None = None          # <user>/<name>; default <user>/syntra-<group>
    public: bool = False
    include_data: bool = True


def _model_card(group: str, version: str, run: Path, ds_dir: Path | None) -> str:
    m = _metrics(run)
    man = layout.read_json(ds_dir / "manifest.json", {}) if ds_dir else {}
    names = []
    if ds_dir and (ds_dir / "data.yaml").exists():
        from app.training.dataset_builder import read_data_yaml_names
        names = read_data_yaml_names(ds_dir / "data.yaml")
    lines = [f"# {group} {version}", "", "YOLO detector trained with Syntra AI.", "",
             "| | |", "|---|---|", f"| Classes | {', '.join(names) or group} |",
             f"| Best validation mAP50 | {m.get('val_map50', '—')} |", f"| Epochs | {m.get('epochs', '—')} |"]
    if man.get("source"):
        lines.append(f"| Training data | {man['source']} |")
    if man.get("gsd_note"):
        lines.append(f"| Imagery | {man['gsd_note']} |")
    lines += ["", "Files: `best.pt` (Ultralytics), `best.onnx` when exported, `results.csv` (per epoch)"
              + (", `data/` (the YOLO dataset it was trained on)." if ds_dir else ".")]
    return "\n".join(lines) + "\n"


def _run_model_push(job_id: str, group: str, version: str, repo: str, token: str, public: bool, include_data: bool) -> None:
    from app.engine.hf_push import PushError, push_model

    jobs = get_store()
    jobs.update(job_id, status="processing")
    run = layout.models_dir() / group / version
    info = layout.read_json(run / "model.json", {}) or {}
    ds = layout.datasets_dir() / group / (info.get("dataset_version") or "") if info.get("dataset_version") else None
    ds = ds if ds and ds.is_dir() else None
    try:
        res = push_model(run, ds if include_data else None, repo, token, _model_card(group, version, run, ds),
                         public=public, progress=lambda p, msg=None: jobs.update(job_id, progress=p, **({"message": msg} if msg else {})))
        layout.write_json(run / "pushed.json", res)
        jobs.update(job_id, status="completed", result=res)
    except PushError as exc:
        jobs.update(job_id, status="error", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        jobs.update(job_id, status="error", error=f"Upload failed: {type(exc).__name__}: {exc}")


@router.post("/{group}/push", dependencies=[Depends(_owner)])
def push_model_version(group: str, body: ModelPushRequest, background_tasks: BackgroundTasks):
    from app.core.secrets import get_secret

    if not all(part == layout.class_slug(part) for part in group.split("+")):
        raise HTTPException(status_code=404, detail="No such model group")
    if body.version not in {p.name for p in layout.versions(layout.models_dir() / group)}:
        raise HTTPException(status_code=404, detail="No such model version")
    if not _metrics(layout.models_dir() / group / body.version).get("weights"):
        raise HTTPException(status_code=409, detail="This version has no trained weights yet")
    token, user = get_secret("hf_token"), get_secret("hf_user")
    if not token:
        raise HTTPException(status_code=409, detail="Add your Hugging Face token in Settings first")
    repo = (body.repo or f"{user}/syntra-{group.replace('_', '-').replace('+', '-')}").strip()
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise HTTPException(status_code=400, detail="Use the form name/repo")
    job_id = get_store().create(MODEL_PUSH_KIND, progress=0.0, group=group, version=body.version, repo=repo, public=body.public)
    background_tasks.add_task(_run_model_push, job_id, group, body.version, repo, token, body.public, body.include_data)
    return {"job_id": job_id, "status": "queued", "repo": repo}
