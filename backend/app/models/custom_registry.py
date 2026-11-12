"""
Trained-model registry — closes the Train -> Detect loop (gap ③ from the
"ما تبقى ليعمل النظام كامل" review).

The static CATALOG in app/models/registry.py is Phase 1/5's fixed category
list (palm/date/tree/car/...), each hand-written with an honest
READY/NOT_PRODUCTION_READY status. A model trained via POST
/api/training/train produces real best.pt weights that have no entry
there at all — without this module, a freshly trained model had nowhere
to go and the Detect step could never run it.

This is the dynamic half: one sidecar JSON file on disk (DATA_DIR/
custom_models.json), same plain-JSON-next-to-nothing-special convention as
app.core.image_library and app.annotations.store. app.models.registry
merges these entries into list_catalog()/get_metadata(), and
app.models.factory dispatches them to app.models.custom_detector —
so once registered, a trained model is a real, selectable, runnable
catalog entry, not a special case bolted on elsewhere.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

_LOCK = threading.Lock()


def _registry_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "custom_models.json"


@dataclass
class CustomModelEntry:
    model_id: str
    name: str
    weights_path: str
    class_names: dict[int, str] = field(default_factory=dict)
    task: str = "detect"
    created_at: str = ""
    source_job_id: str | None = None
    data_yaml: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(raw: dict) -> CustomModelEntry:
    raw = dict(raw)
    raw["class_names"] = {int(k): v for k, v in (raw.get("class_names") or {}).items()}
    return CustomModelEntry(**raw)


def _load_all(data_dir: str | Path) -> list[dict]:
    path = _registry_path(data_dir)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_all(data_dir: str | Path, entries: list[dict]) -> None:
    path = _registry_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def list_custom_models(data_dir: str | Path) -> list[CustomModelEntry]:
    return [_coerce(e) for e in _load_all(data_dir)]


def get_custom_model(data_dir: str | Path, model_id: str) -> CustomModelEntry | None:
    for e in list_custom_models(data_dir):
        if e.model_id == model_id:
            return e
    return None


def register_custom_model(
    data_dir: str | Path,
    *,
    name: str,
    weights_path: str,
    class_names: dict[int, str],
    task: str = "detect",
    source_job_id: str | None = None,
    data_yaml: str | None = None,
    notes: str = "",
) -> CustomModelEntry:
    weights_path = str(weights_path)
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not class_names:
        raise ValueError("class_names cannot be empty — nothing to detect.")
    with _LOCK:
        entries = _load_all(data_dir)
        entry = CustomModelEntry(
            model_id=f"custom_{uuid.uuid4().hex[:8]}",
            name=name,
            weights_path=weights_path,
            class_names=dict(class_names),
            task=task,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source_job_id=source_job_id,
            data_yaml=data_yaml,
            notes=notes,
        )
        entries.append(entry.to_dict())
        _save_all(data_dir, entries)
        return entry


def delete_custom_model(data_dir: str | Path, model_id: str) -> bool:
    with _LOCK:
        entries = _load_all(data_dir)
        kept = [e for e in entries if e.get("model_id") != model_id]
        if len(kept) == len(entries):
            return False
        _save_all(data_dir, kept)
        return True
