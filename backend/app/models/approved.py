"""
The models the owner approved in the Library, as detection models.

For every class (or group of classes) with models/<group>/approved.json, one
catalog entry `class_<group>` points at the approved version's weights. The
id stays the same when the owner approves a newer version or rolls back, so
a user's selection keeps working and always runs what is approved now; the
version is shown in the entry and saved with each run.

Status is honest about what was measured: READY only when the version has a
test score from the owner's accuracy areas (test_metrics.json), otherwise
VALIDATION ("Unvalidated").
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PREFIX = "class_"


@dataclass
class ApprovedModel:
    model_id: str
    group: str
    version: str
    weights: str
    class_names: dict[int, str]
    task: str
    test: dict | None
    dataset_version: str | None


def _weights(run: Path) -> Path | None:
    for name in ("best.pt", "best.onnx"):
        found = sorted(run.rglob(name))
        if found:
            return found[0]
    return None


def list_approved() -> list[ApprovedModel]:
    from app.core import layout
    from app.training.dataset_builder import read_data_yaml_names

    root = layout.models_dir()
    out: list[ApprovedModel] = []
    if not root.is_dir():
        return out
    for g in sorted(p for p in root.iterdir() if p.is_dir()):
        version = layout.approved(g.name)
        if not version:
            continue
        run = g / version
        w = _weights(run)
        if w is None:
            continue
        info = layout.read_json(run / "model.json", {}) or {}
        ds = layout.datasets_dir() / g.name / (info.get("dataset_version") or "")
        names, task = [], "detect"
        yaml = ds / "data.yaml"
        if yaml.is_file():
            names = read_data_yaml_names(str(yaml))
            for line in yaml.read_text(encoding="utf-8").splitlines():
                if line.startswith("task:"):
                    task = line.split(":", 1)[1].strip() or "detect"
        if not names:
            names = g.name.split("+")
        test = next(iter(sorted(run.rglob("test_metrics.json"))), None)
        out.append(ApprovedModel(
            model_id=PREFIX + g.name, group=g.name, version=version, weights=str(w),
            class_names=dict(enumerate(names)), task=task,
            test=layout.read_json(test, None) if test else None, dataset_version=info.get("dataset_version"),
        ))
    return out


def get_approved(model_id: str) -> ApprovedModel | None:
    if not model_id.startswith(PREFIX):
        return None
    return next((m for m in list_approved() if m.model_id == model_id), None)


def weights_of(group: str, version: str) -> str | None:
    from app.core import layout

    w = _weights(layout.models_dir() / group / version)
    return str(w) if w else None
