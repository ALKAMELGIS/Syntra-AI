"""Offline tests for app.core.security and the API surfaces it protects.

The unit tests need only FastAPI; the HTTP tests use FastAPI's TestClient
against the real app (no rasterio/torch needed: those are lazy-imported).
"""
from __future__ import annotations

import io

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core import security
from app.core.config import settings


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "data" / "results")
    monkeypatch.delenv("SYNTRA_EXTRA_DATA_ROOTS", raising=False)
    monkeypatch.delenv("SYNTRA_API_KEY", raising=False)
    monkeypatch.delenv("SYNTRA_ALLOW_ENV_ACTIONS", raising=False)
    settings.ensure_dirs()
    yield tmp_path


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


# --- confine_path ---------------------------------------------------------
def test_confine_path_accepts_file_inside_data_dir():
    f = settings.UPLOAD_DIR / "a.tif"
    f.write_bytes(b"x")
    assert security.confine_path(str(f), must_exist=True) == f.resolve()


def test_confine_path_rejects_traversal_and_absolute_outside(tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    for raw in (str(outside), "../secret.txt", "uploads/../../secret.txt"):
        with pytest.raises(HTTPException) as exc:
            security.confine_path(raw)
        assert exc.value.status_code == 403


def test_confine_path_honours_extra_roots(tmp_path, monkeypatch):
    extra = tmp_path / "labelled"
    extra.mkdir()
    monkeypatch.setenv("SYNTRA_EXTRA_DATA_ROOTS", str(extra))
    assert security.confine_path(str(extra / "x.json")) == (extra / "x.json").resolve()


def test_confine_path_missing_file_is_404():
    with pytest.raises(HTTPException) as exc:
        security.confine_path(str(settings.UPLOAD_DIR / "nope.tif"), must_exist=True)
    assert exc.value.status_code == 404


# --- safe_filename ----------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("../../evil.tif", "evil.tif"),
    ("..\\..\\win.tif", "win.tif"),
    ("ok name-1.tif", "ok name-1.tif"),
    ("مدينة.tif", "_.tif"),
    ("", "upload"),
    ("..", "upload"),
])
def test_safe_filename(raw, expected):
    assert security.safe_filename(raw) == expected


# --- HTTP surfaces ----------------------------------------------------------
def test_upload_cannot_escape_upload_dir(client):
    r = client.post("/api/images", files={"file": ("../../escape.tif", io.BytesIO(b"II*\x00"), "image/tiff")})
    # rasterio may be missing (200 with a note) or reject the fake bytes (400);
    # either way nothing may be written outside UPLOAD_DIR.
    assert r.status_code in (200, 400)
    assert not (settings.DATA_DIR / "escape.tif").exists()
    assert not (settings.DATA_DIR.parent / "escape.tif").exists()
    if r.status_code == 200:
        assert r.json()["filename"] == "escape.tif"


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0)
    r = client.post("/api/images", files={"file": ("big.tif", io.BytesIO(b"x" * 10), "image/tiff")})
    assert r.status_code == 413
    assert not any(settings.UPLOAD_DIR.glob("*big.tif"))


def test_predict_rejects_path_outside_data_dir(client, tmp_path):
    outside = tmp_path / "system.tif"
    outside.write_bytes(b"x")
    r = client.post("/api/predict", json={"model_id": "car_detection", "image_path": str(outside)})
    assert r.status_code == 403


def test_predict_unknown_image_id_is_404(client):
    r = client.post("/api/predict", json={"model_id": "car_detection", "image_id": "does-not-exist"})
    assert r.status_code == 404


def test_predict_requires_an_image(client):
    r = client.post("/api/predict", json={"model_id": "car_detection"})
    assert r.status_code == 400


def test_results_job_id_traversal_is_404(client):
    (settings.DATA_DIR / "car_detection.geojson").write_text("{}")
    r = client.get("/api/results/../car_detection")
    assert r.status_code == 404
    r = client.get("/api/results/%2E%2E/car_detection")
    assert r.status_code == 404


def test_env_actions_blocked_when_disabled(client, monkeypatch):
    monkeypatch.setenv("SYNTRA_ALLOW_ENV_ACTIONS", "0")
    r = client.post("/api/environment/actions", json={"action": "install"})
    assert r.status_code == 403


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("SYNTRA_API_KEY", "s3cret")
    assert client.get("/api/health").status_code == 200  # health stays open for probes
    assert client.get("/api/images").status_code == 401
    assert client.get("/api/images", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/images", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/api/images?api_key=s3cret").status_code == 200


def test_api_open_when_no_key_configured(client):
    assert client.get("/api/images").status_code == 200


def test_unexpected_errors_are_explained_as_json(monkeypatch, tmp_path):
    import errno

    from fastapi.testclient import TestClient

    import app.api.images as images
    from app.core.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.delenv("SYNTRA_API_KEY", raising=False)
    c = TestClient(app, raise_server_exceptions=False)

    def full(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(images, "list_library", full)
    r = c.get("/api/images")
    assert r.status_code == 507 and "disk is full" in r.json()["detail"]

    def boom(*a, **k):
        raise ValueError("bad band count")

    monkeypatch.setattr(images, "list_library", boom)
    r = c.get("/api/images")
    assert r.status_code == 500 and "ValueError: bad band count" in r.json()["detail"]
