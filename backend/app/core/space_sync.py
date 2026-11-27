"""Keep a Hugging Face Space's data across restarts.

A Space's disk is wiped when it restarts or is rebuilt, and on ZeroGPU hardware a
mounted storage bucket can't be written to. So the data folder (accounts, the
image library, labels, results, trained models) is copied to a private dataset
repo on the Hub:

- at start, before anything opens the folder, the repo's files are downloaded
  into it (restore);
- then every SYNTRA_SYNC_EVERY seconds (default 120) the files that changed are
  sent in one commit, and the ones removed are deleted (save). Databases are sent
  as consistent copies made with SQLite's backup API, never as live files.
- every SQUASH_EVERY commits the repo's history is squashed, so old copies of
  the databases don't pile up in its storage.

It runs only when both are set (Space > Settings > Variables and secrets):
  SYNTRA_SYNC_REPO  the dataset repo, e.g. "Alkamelgis/syntra-data" (made private if missing)
  HF_TOKEN          a token with write access to it (secret)
The click tool's model files (engines/) and temporary files are not copied: they
are fetched or made again.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger("syntra.sync")

SKIP = ["engines/*", "tmp/*", "*.db-wal", "*.db-shm", "*.db-journal", ".cache/*", ".sync/*", "*/__pycache__/*",
        "*.part", "*.lock", "uploads_tmp/*", "chunks/*", "*/chunks/*"]
MAX_FILE_MB = 2000
SQUASH_EVERY = 20


def settings_from_env() -> tuple[str, str] | None:
    repo, token = os.getenv("SYNTRA_SYNC_REPO", "").strip(), os.getenv("HF_TOKEN", "").strip()
    return (repo, token) if repo and token else None


def _skipped(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in SKIP)


def scan(data_dir: Path) -> dict[str, tuple[int, int]]:
    """{relative path: (size, mtime_ns)} of the files to keep."""
    out = {}
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(data_dir).as_posix()
        if _skipped(rel):
            continue
        st = p.stat()
        if st.st_size > MAX_FILE_MB * 1024 * 1024:
            continue
        out[rel] = (st.st_size, st.st_mtime_ns)
    return out


def _db_copy(src: Path, dst: Path) -> None:
    """A consistent copy of a live SQLite database (WAL included)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()


class Sync:
    def __init__(self, data_dir: Path, repo: str, token: str, api=None):
        self.data_dir = Path(data_dir)
        self.repo = repo
        self.token = token
        self._api = api
        self.manifest_path = self.data_dir / ".sync" / "manifest.json"
        self.commits = 0
        self.last: dict = {"saved_at": None, "error": None, "files": 0}
        self._lock = threading.Lock()

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api

    def _manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_manifest(self, m: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(m), encoding="utf-8")

    def restore(self) -> int:
        """Download the repo's files into the data folder. -> how many files."""
        self.api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)
        files = [f for f in self.api.list_repo_files(self.repo, repo_type="dataset") if not f.startswith(".") and f != "README.md"]
        if not files:
            return 0
        self._download(files)
        self._write_manifest({k: list(v) for k, v in scan(self.data_dir).items()})
        return len(files)

    def _download(self, files: list[str]) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=self.repo, repo_type="dataset", local_dir=str(self.data_dir),
                          allow_patterns=files, token=self.token)

    def save(self) -> dict:
        """Send what changed since the last save in one commit. -> {added, deleted}."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        with self._lock:
            now = scan(self.data_dir)
            before = self._manifest()
            changed = [k for k, v in now.items() if before.get(k) != list(v)]
            gone = [k for k in before if k not in now]
            if not changed and not gone:
                return {"added": 0, "deleted": 0}
            ops = []
            with tempfile.TemporaryDirectory() as tmp:
                for rel in changed:
                    src = self.data_dir / rel
                    if rel.endswith(".db") or rel.endswith(".sqlite"):
                        copy = Path(tmp) / rel
                        try:
                            _db_copy(src, copy)
                        except sqlite3.Error:
                            continue
                        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=copy.read_bytes()))
                    else:
                        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(src)))
                ops += [CommitOperationDelete(path_in_repo=rel) for rel in gone]
                self.api.create_commit(repo_id=self.repo, repo_type="dataset", operations=ops,
                                       commit_message=f"Syntra data: {len(changed)} changed, {len(gone)} removed")
            self._write_manifest({k: list(v) for k, v in now.items()})
            self.commits += 1
            if self.commits % SQUASH_EVERY == 0:
                try:
                    self.api.super_squash_history(repo_id=self.repo, repo_type="dataset")
                except Exception as exc:  # noqa: BLE001 - keeping history is harmless
                    log.warning("Could not squash the data repo's history: %s", exc)
            self.last.update(saved_at=time.time(), error=None, files=len(now))
            return {"added": len(changed), "deleted": len(gone)}

    def loop(self, every: float) -> None:
        while True:
            time.sleep(every)
            try:
                self.save()
            except Exception as exc:  # noqa: BLE001 - try again next round
                self.last["error"] = str(exc)
                log.warning("Saving the data to %s failed: %s", self.repo, exc)


_sync: Sync | None = None


def restore_at_start(data_dir: Path) -> str:
    """Called before the app opens its data folder. -> a line for the log."""
    global _sync
    cfg = settings_from_env()
    if not cfg:
        return "Data sync off (set SYNTRA_SYNC_REPO and HF_TOKEN to keep data across restarts)"
    _sync = Sync(Path(data_dir), *cfg)
    try:
        n = _sync.restore()
        return f"Data restored from {cfg[0]}: {n} files"
    except Exception as exc:  # noqa: BLE001 - start empty rather than not at all
        _sync.last["error"] = str(exc)
        return f"Could not restore data from {cfg[0]}: {exc}"


def start_saving() -> bool:
    if _sync is None:
        return False
    every = float(os.getenv("SYNTRA_SYNC_EVERY", "120"))
    threading.Thread(target=_sync.loop, args=(max(30.0, every),), daemon=True, name="syntra-sync").start()
    return True


def status() -> dict:
    if _sync is None:
        return {"on": False}
    return {"on": True, "repo": _sync.repo, **_sync.last}
