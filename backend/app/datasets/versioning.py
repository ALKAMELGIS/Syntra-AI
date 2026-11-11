"""
Simple, dependency-free dataset version control: each `register_version()`
call snapshots a manifest (sample list, class balance, split, source paths)
to `<data_dir>/dataset_versions/<model_id>/<version_id>.json`. This is
intentionally plain JSON-on-disk rather than a database or DVC/LakeFS
integration, so it works with nothing beyond the stdlib; swap in a real
versioning backend later without changing the DatasetVersion schema.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from app.datasets.schema import AnnotationFormat, ClassBalanceReport, DatasetSplit, DatasetVersion


def register_version(
    data_dir: str | Path,
    model_id: str,
    sample_ids: list[str],
    class_balance_report: ClassBalanceReport,
    split: DatasetSplit,
    annotation_format: AnnotationFormat,
    source_paths: list[str],
) -> DatasetVersion:
    version = DatasetVersion(
        version_id=str(uuid.uuid4())[:8],
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_id=model_id,
        sample_count=len(sample_ids),
        class_balance=class_balance_report,
        split=split,
        annotation_format=annotation_format,
        source_paths=source_paths,
    )

    out_dir = Path(data_dir) / "dataset_versions" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{version.version_id}.json"
    out_path.write_text(json.dumps(asdict(version), indent=2, default=str))
    return version


def list_versions(data_dir: str | Path, model_id: str) -> list[dict]:
    out_dir = Path(data_dir) / "dataset_versions" / model_id
    if not out_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]


def load_version(data_dir: str | Path, model_id: str, version_id: str) -> dict:
    path = Path(data_dir) / "dataset_versions" / model_id / f"{version_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No dataset version '{version_id}' for model '{model_id}'.")
    return json.loads(path.read_text())
