"""
Persistent job store for every background job the API runs.

Before this module each router kept its jobs in a module-level dict, so a
server restart (or a crash, or `--reload` picking up an edit) silently lost
every queued, running and finished job: the dashboard kept polling ids the
server no longer knew and got 404s.

Jobs now live in a small SQLite database at `SYNTRA_DATA_DIR/syntra.db`:

* SQLite ships with Python, so this works on the user's Windows machine
  with nothing extra to install, and needs no separate server process.
* One row per job: id, kind ("predict", "dataset_export", "train",
  "environment"), status, timestamps, and a JSON document with everything
  else (progress, result, log tail...). The routers keep their own field
  names; this module never interprets them beyond `status`.
* On startup, jobs left in an active state by a previous process are
  marked `interrupted` with a clear message. Their worker thread is gone, so
  reporting them as still running would be a lie.

The interface (create/get/update/append/list) is deliberately small so it
can be backed by PostgreSQL later without touching the routers.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings

ACTIVE_STATUSES = ("queued", "processing", "running")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_kind_created ON jobs (kind, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _dumps(data: dict[str, Any]) -> str:
    # default=str keeps Path objects and other simple values serialisable
    return json.dumps(data, default=str)


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)

    # ----------------------------------------------------------------- core
    def create(self, kind: str, *, job_id: str | None = None, status: str = "queued", **fields: Any) -> str:
        job_id = job_id or str(uuid.uuid4())
        now = _now()
        data = {**fields, "status": status}
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, status, created_at, updated_at, data) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, kind, status, now, now, _dumps(data)),
            )
        return job_id

    def get(self, job_id: str, kind: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or (kind is not None and row["kind"] != kind):
            return None
        return self._row_to_job(row)

    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        """Merge `fields` into the job document. Returns the updated job."""
        with self._lock:
            row = self._conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            data = json.loads(row["data"])
            data.update(fields)
            status = data.get("status", "queued")
            self._conn.execute(
                "UPDATE jobs SET data = ?, status = ?, updated_at = ? WHERE id = ?",
                (_dumps(data), status, _now(), job_id),
            )
        return data

    def append(self, job_id: str, key: str, item: Any, *, max_len: int | None = None) -> None:
        """Append to a list field, keeping only the last `max_len` items."""
        with self._lock:
            row = self._conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return
            data = json.loads(row["data"])
            items = list(data.get(key) or [])
            items.append(item)
            if max_len is not None and len(items) > max_len:
                items = items[-max_len:]
            data[key] = items
            self._conn.execute(
                "UPDATE jobs SET data = ?, updated_at = ? WHERE id = ?",
                (_dumps(data), _now(), job_id),
            )

    def list(self, kind: str | None = None, *, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM jobs", []
        where = []
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if status:
            where.append("status = ?")
            args.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_job(r) for r in rows]

    def mark_interrupted(self) -> int:
        """Called once at startup: no job from a previous process can still
        be running, because its worker thread died with that process."""
        message = "The server restarted while this job was running. Start it again."
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, data FROM jobs WHERE status IN ({','.join('?' * len(ACTIVE_STATUSES))})",
                ACTIVE_STATUSES,
            ).fetchall()
            now = _now()
            for r in rows:
                data = json.loads(r["data"])
                data.update(status="interrupted", error=message, previous_status=data.get("status"))
                self._conn.execute(
                    "UPDATE jobs SET data = ?, status = 'interrupted', updated_at = ? WHERE id = ?",
                    (_dumps(data), now, r["id"]),
                )
        return len(rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
        data = json.loads(row["data"])
        data.update(
            job_id=row["id"], kind=row["kind"], status=row["status"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
        return data


# One store per database path. Tests point SYNTRA_DATA_DIR at a temp folder,
# so the store follows settings.DATA_DIR rather than being fixed at import.
_STORES: dict[Path, JobStore] = {}
_STORES_LOCK = threading.Lock()


def get_store() -> JobStore:
    path = (settings.DATA_DIR / "syntra.db").resolve()
    with _STORES_LOCK:
        store = _STORES.get(path)
        if store is None:
            store = _STORES[path] = JobStore(path)
        return store
