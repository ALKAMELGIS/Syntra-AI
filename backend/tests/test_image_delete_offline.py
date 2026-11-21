"""Deleting an image from the library: only Syntra's copy, never one with labels."""
import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SYNTRA_AUTH", "off")
    from app.main import app
    return TestClient(app)


def _copy(folder, name="scene.tif", original=None):
    from app.core.config import settings
    d = settings.UPLOAD_DIR / folder
    d.mkdir(parents=True)
    p = d / name
    if original:
        os.link(original, p)                       # opened from the user's computer: a hard link
    else:
        p.write_bytes(b"II*\0" * 100)
    (d / f"{name}.meta.json").write_text(json.dumps({"image_id": folder + "-0000", "filename": name, "path": str(p)}))
    return p


def test_deletes_the_copy_and_keeps_the_users_original(client, tmp_path):
    original = tmp_path / "mine.tif"
    original.write_bytes(b"II*\0" * 10)
    p = _copy("abcd1234", original=original)
    r = client.delete("/api/images/library/abcd1234")
    assert r.status_code == 200 and r.json()["freed_bytes"] > 0
    assert not p.parent.exists() and original.read_bytes() == b"II*\0" * 10


def test_refuses_an_image_with_labels(client):
    from app.annotations import store
    p = _copy("beef0001")
    store.add_annotation(p, "ship", [[0.1, 0.9], [0.2, 0.9], [0.2, 0.8]], {"west": 0, "south": 0, "east": 1, "north": 1}, 100, 100)
    r = client.delete("/api/images/library/beef0001")
    assert r.status_code == 409 and "label" in r.json()["detail"]
    assert p.exists()


@pytest.mark.parametrize("folder", ["..", "../x", "ABCDEF12", "abc", "nothere1"])
def test_refuses_anything_that_is_not_a_library_folder(client, folder):
    assert client.delete(f"/api/images/library/{folder}").status_code in (400, 404, 405)
