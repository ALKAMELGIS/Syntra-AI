"""Space data sync: restore before start, then only changed files are sent, live
SQLite databases as consistent copies, removed files deleted, nothing synced when
the settings are missing."""
import sqlite3

import pytest

pytest.importorskip("huggingface_hub")

from app.core import space_sync  # noqa: E402


class FakeHub:
    def __init__(self):
        self.files = {}        # path -> bytes
        self.commits = []
        self.squashed = 0

    def create_repo(self, *a, **k):
        pass

    def list_repo_files(self, *a, **k):
        return list(self.files)

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        names = []
        for op in operations:
            if type(op).__name__ == "CommitOperationDelete":
                self.files.pop(op.path_in_repo, None)
                names.append("-" + op.path_in_repo)
            else:
                src = op.path_or_fileobj
                self.files[op.path_in_repo] = src if isinstance(src, bytes) else open(src, "rb").read()
                names.append("+" + op.path_in_repo)
        self.commits.append(sorted(names))

    def super_squash_history(self, **k):
        self.squashed += 1


def make_sync(tmp_path, hub):
    s = space_sync.Sync(tmp_path, "me/data", "tok", api=hub)

    def download(files):
        for f in files:
            p = tmp_path / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(hub.files[f])
    s._download = download
    return s


def test_sends_only_what_changed_and_deletes_what_went(tmp_path):
    hub = FakeHub()
    (tmp_path / "images" / "a").mkdir(parents=True)
    (tmp_path / "images" / "a" / "x.tif").write_bytes(b"tif")
    (tmp_path / "engines" / "segment").mkdir(parents=True)
    (tmp_path / "engines" / "segment" / "m.onnx").write_bytes(b"big")         # fetched again: not kept
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "up.part").write_bytes(b"half")
    s = make_sync(tmp_path, hub)
    assert s.save() == {"added": 1, "deleted": 0}
    assert hub.commits[-1] == ["+images/a/x.tif"]
    assert s.save() == {"added": 0, "deleted": 0}                              # nothing new: no commit
    assert len(hub.commits) == 1
    (tmp_path / "images" / "a" / "x.tif").unlink()
    (tmp_path / "results.json").write_text("{}")
    s.save()
    assert hub.commits[-1] == ["+results.json", "-images/a/x.tif"]


def test_a_live_database_is_sent_as_a_consistent_copy(tmp_path):
    hub = FakeHub()
    live = sqlite3.connect(tmp_path / "auth.db")
    live.execute("PRAGMA journal_mode=WAL")
    live.execute("create table u (email text)")
    live.execute("insert into u values ('owner@example.test')")
    live.commit()                                       # still in the WAL file, not the main file
    s = make_sync(tmp_path, hub)
    s.save()
    assert "auth.db" in hub.files and not any(k.endswith("-wal") for k in hub.files)
    copy = tmp_path / "check.db"
    copy.write_bytes(hub.files["auth.db"])
    assert sqlite3.connect(copy).execute("select email from u").fetchall() == [("owner@example.test",)]
    live.close()


def test_restore_brings_the_files_back(tmp_path):
    hub = FakeHub()
    hub.files = {"images/a/x.tif": b"tif", "auth.db": b"db"}
    s = make_sync(tmp_path, hub)
    assert s.restore() == 2
    assert (tmp_path / "images" / "a" / "x.tif").read_bytes() == b"tif"
    assert s.save() == {"added": 0, "deleted": 0}       # what came back isn't sent again


def test_history_is_squashed_now_and_then(tmp_path):
    hub = FakeHub()
    s = make_sync(tmp_path, hub)
    for i in range(space_sync.SQUASH_EVERY):
        (tmp_path / f"f{i}.txt").write_text(str(i))
        s.save()
    assert hub.squashed == 1


def test_off_without_the_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTRA_SYNC_REPO", raising=False)
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setattr(space_sync, "_sync", None)
    assert "off" in space_sync.restore_at_start(tmp_path)
    assert space_sync.start_saving() is False
    assert space_sync.status() == {"on": False}
