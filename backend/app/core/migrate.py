"""
One-time move of the data folder into the per-class layout (app.core.layout).

    uploads/<uuid>_<name>.tif (+ sidecars)  ->  images/<key>/<name>.tif (+ sidecars)
    <image>.annotations.json                ->  classes/<class>/labels/<key>.geojson,
                                                classes/_areas/<key>.geojson

Files are moved, not copied (images can be gigabytes). Every move is
written to migration.json first, so it can be traced and undone by hand;
the old label files are kept, renamed *.annotations.json.moved. Datasets
and training runs made before the layout stay where they are: the jobs that
point at them keep working, and new ones go into datasets/<class>/vNNN and
models/<class>/vNNN.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path

from app.core.config import settings

log = logging.getLogger(__name__)
_LEGACY = re.compile(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_(.+\.tiff?)$", re.I)


def migrate_layout(data_dir: Path | None = None) -> dict:
    from app.annotations import store
    from app.core.layout import image_key

    data_dir = Path(data_dir or settings.DATA_DIR)
    legacy = data_dir / "uploads"
    images = settings.UPLOAD_DIR
    report = {"moved_images": 0, "moved_files": 0, "labels": 0, "errors": []}
    if not legacy.is_dir() or legacy.resolve() == images.resolve():
        return report
    record = data_dir / "migration.json"
    done = json.loads(record.read_text()) if record.exists() else {"moves": []}

    for tif in sorted(legacy.iterdir()):
        m = _LEGACY.match(tif.name)
        if not m or not tif.is_file():
            continue
        image_id, name = m.group(1), m.group(2)
        folder = images / image_key(image_id)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            moves = [(tif, folder / name)]
            for side in legacy.iterdir():                     # <file>.meta.json, .preview.png, .annotations.json…
                if side != tif and side.name.startswith(tif.name):
                    moves.append((side, folder / (name + side.name[len(tif.name):])))
            for xml in (tif.with_suffix(".xml"), tif.with_suffix(".XML")):
                if xml.exists():
                    moves.append((xml, folder / (Path(name).stem + xml.suffix)))
            done["moves"].extend({"from": str(a), "to": str(b), "at": time.strftime("%Y-%m-%dT%H:%M:%S")} for a, b in moves)
            record.write_text(json.dumps(done, indent=2))
            for a, b in moves:
                shutil.move(str(a), str(b))
                report["moved_files"] += 1
            report["moved_images"] += 1
            meta = folder / (name + ".meta.json")
            if meta.exists():
                entry = json.loads(meta.read_text())
                entry["path"] = str(folder / name)
                meta.write_text(json.dumps(entry, indent=2))
            report["labels"] += len(store.load_annotations(folder / name))   # moves the old label file
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
            report["errors"].append(f"{tif.name}: {exc}")
    try:
        if not any(legacy.iterdir()):
            legacy.rmdir()
    except OSError:
        pass
    if report["moved_images"]:
        log.info("Data folder: moved %d images into images/<key>/, %d labels into classes/",
                 report["moved_images"], report["labels"])
    return report
