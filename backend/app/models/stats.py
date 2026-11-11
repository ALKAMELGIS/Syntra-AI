"""
Per-model processing statistics — spec Section 7 ("Processing statistics"
is one of the required Model Management Interface fields).

Plain JSON-on-disk, read-modify-write under a module-level lock — adequate
for a single-process deployment; a multi-worker/multi-process deployment
should swap this for a real datastore (Redis/Postgres) without changing the
public functions below (`record_run` / `get_stats` / `get_all_stats`).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_LOCK = threading.Lock()


@dataclass
class ModelRunStats:
    model_id: str
    run_count: int = 0
    error_count: int = 0
    total_tiles_processed: int = 0
    total_detections: int = 0
    total_duration_seconds: float = 0.0
    last_run_at: str | None = None
    last_status: str | None = None
    last_device: str | None = None
    last_duration_seconds: float | None = None
    last_detection_count: int | None = None

    @property
    def avg_duration_seconds(self) -> float:
        return round(self.total_duration_seconds / self.run_count, 3) if self.run_count else 0.0

    @property
    def avg_detections_per_run(self) -> float:
        return round(self.total_detections / self.run_count, 2) if self.run_count else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["avg_duration_seconds"] = self.avg_duration_seconds
        d["avg_detections_per_run"] = self.avg_detections_per_run
        return d


def _stats_path(data_dir: str | Path, model_id: str) -> Path:
    return Path(data_dir) / "model_stats" / f"{model_id}.json"


def get_stats(data_dir: str | Path, model_id: str) -> ModelRunStats:
    path = _stats_path(data_dir, model_id)
    if not path.exists():
        return ModelRunStats(model_id=model_id)
    raw = json.loads(path.read_text())
    raw.pop("avg_duration_seconds", None)
    raw.pop("avg_detections_per_run", None)
    return ModelRunStats(**raw)


def get_all_stats(data_dir: str | Path) -> dict[str, ModelRunStats]:
    stats_dir = Path(data_dir) / "model_stats"
    if not stats_dir.exists():
        return {}
    out = {}
    for p in stats_dir.glob("*.json"):
        out[p.stem] = get_stats(data_dir, p.stem)
    return out


def record_run(
    data_dir: str | Path,
    model_id: str,
    *,
    status: str,                       # "completed" | "error"
    tiles_processed: int = 0,
    detection_count: int = 0,
    duration_seconds: float = 0.0,
    device: str | None = None,
) -> ModelRunStats:
    """Atomically (within this process) accumulate one run's outcome into the model's persisted stats."""
    path = _stats_path(data_dir, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        stats = get_stats(data_dir, model_id)
        stats.run_count += 1
        if status == "error":
            stats.error_count += 1
        stats.total_tiles_processed += tiles_processed
        stats.total_detections += detection_count
        stats.total_duration_seconds += duration_seconds
        stats.last_run_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stats.last_status = status
        stats.last_device = device
        stats.last_duration_seconds = duration_seconds
        stats.last_detection_count = detection_count

        path.write_text(json.dumps(asdict(stats), indent=2))
        return stats
