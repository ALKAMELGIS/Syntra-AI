"""
Measured run speed on this machine, for an honest time estimate before Run.

Every completed run records seconds per model tile (empty tiles excluded: they
cost almost nothing) under its model, device kind and tile size. The estimate
is the median of the last five runs. No measurement yet = no estimate: the UI
shows the area only, never an invented duration.
"""
from __future__ import annotations

import json
import os
import statistics
import threading
from pathlib import Path

_LOCK = threading.Lock()
_KEEP = 5


def _file(data_dir: Path) -> Path:
    return Path(data_dir) / "speed.json"


def _key(model_id: str, device: str, tile_size: int) -> str:
    kind = "cpu" if not device or device == "cpu" else "gpu"
    return f"{model_id}|{kind}|{int(tile_size)}"


def _load(data_dir: Path) -> dict:
    try:
        return json.loads(_file(data_dir).read_text())
    except (OSError, ValueError):
        return {}


def record(data_dir: Path, model_id: str, device: str, tile_size: int, model_tiles: int, seconds: float) -> None:
    if model_tiles < 4 or seconds <= 0:          # too few tiles to say anything about speed
        return
    with _LOCK:
        d = _load(data_dir)
        k = _key(model_id, device, tile_size)
        d[k] = (d.get(k, []) + [round(seconds / model_tiles, 4)])[-_KEEP:]
        tmp = _file(data_dir).with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.replace(tmp, _file(data_dir))


def seconds_per_tile(data_dir: Path, model_id: str, device: str, tile_size: int) -> float | None:
    vals = _load(data_dir).get(_key(model_id, device, tile_size))
    return statistics.median(vals) if vals else None


class Eta:
    """Remaining time from this run's own pace (all tiles, empty ones included)."""

    def __init__(self, clock=None):
        import time

        self._clock = clock or time.monotonic
        self._start = None

    def update(self, done: int, total: int | None) -> int | None:
        now = self._clock()
        if self._start is None:
            self._start = now
            return None
        if not total or done <= 0 or done >= total:
            return 0 if total and done >= total else None
        elapsed = now - self._start
        if elapsed < 2:                            # first seconds: the pace is not known yet
            return None
        return int(round(elapsed / done * (total - done)))
