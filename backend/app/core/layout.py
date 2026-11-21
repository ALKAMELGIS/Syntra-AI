"""
Where everything lives under SYNTRA_DATA_DIR, by class.

    images/<key>/<name>.tif          each image once, with its sidecars
                                     (<name>.tif.meta.json, .xml, .preview.png)
    classes/<class>/class.json       colour, outline shape, target
    classes/<class>/labels/<key>.geojson   the owner's labels: the source of truth
    classes/_areas/<key>.geojson     training ("complete") and accuracy areas of an image
    datasets/<class>/v001/           data.yaml, manifest.json,
                                     images/{train,val,test}, labels/{train,val,test}
    models/<class>/v001/             the training run (weights/best.onnx, results.csv, metrics)
    models/<class>/approved.json     which version detection uses
    engines/                         tools Syntra downloads (the click-to-segment model)
    results/                         detection results

<key> is the first 8 hex digits of the image id: short paths, because
Windows refuses paths over 260 characters unless long paths are enabled.
A dataset or model made from several classes is filed under their names
joined by "+" (building+ship).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.config import settings

AREAS = "_areas"
_SLUG = re.compile(r"[^a-z0-9_-]+")


def image_key(image_id: str) -> str:
    return re.sub(r"[^0-9a-f]", "", (image_id or "").lower())[:8] or "image"


def class_slug(name: str) -> str:
    s = _SLUG.sub("_", (name or "").strip().lower()).strip("_")
    return s[:40] or "object"


def group_slug(class_names: list[str]) -> str:
    """Folder for a dataset or model of one or more classes."""
    names = sorted({class_slug(n) for n in class_names if n})
    return "+".join(names)[:80] or "object"


def images_dir() -> Path:
    return settings.UPLOAD_DIR


def classes_dir() -> Path:
    return settings.DATA_DIR / "classes"


def datasets_dir() -> Path:
    return settings.DATA_DIR / "datasets"


def models_dir() -> Path:
    return settings.DATA_DIR / "models"


def class_dir(name: str) -> Path:
    return classes_dir() / (AREAS if name == AREAS else class_slug(name))


def next_version(parent: Path) -> tuple[str, Path]:
    """('v003', parent/'v003') — the first free version folder, created empty."""
    parent.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name[1:]) for p in parent.iterdir() if p.is_dir() and re.fullmatch(r"v\d{3,}", p.name)]
    n = max(nums, default=0) + 1
    while True:
        name = f"v{n:03d}"
        path = parent / name
        try:
            path.mkdir()
            return name, path
        except FileExistsError:            # two builds at once
            n += 1


def versions(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted((p for p in parent.iterdir() if p.is_dir() and re.fullmatch(r"v\d{3,}", p.name)),
                  key=lambda p: int(p.name[1:]))


def approved(group: str) -> str | None:
    f = models_dir() / group / "approved.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError):
        return None


def set_approved(group: str, version: str | None) -> None:
    d = models_dir() / group
    d.mkdir(parents=True, exist_ok=True)
    f = d / "approved.json"
    if version is None:
        f.unlink(missing_ok=True)
        return
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": version}, indent=2), encoding="utf-8")
    tmp.replace(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def dataset_of(data_yaml: str | Path) -> tuple[str, str] | None:
    """(group, version) when data.yaml is in datasets/<group>/vNNN, else None."""
    try:
        rel = Path(data_yaml).resolve().parent.relative_to(datasets_dir().resolve())
    except (ValueError, OSError):
        return None
    parts = rel.parts
    return (parts[0], parts[1]) if len(parts) == 2 and re.fullmatch(r"v\d{3,}", parts[1]) else None


def new_model_dir(data_yaml: str | Path) -> tuple[str | None, str | None, Path | None]:
    """models/<group>/vNNN for a dataset of the library; (None, None, None) for any other data.yaml."""
    ds = dataset_of(data_yaml)
    if not ds:
        return None, None, None
    version, path = next_version(models_dir() / ds[0])
    write_json(path / "model.json", {"dataset": ds[0], "dataset_version": ds[1], "version": version})
    return ds[0], version, path
