"""Offline tests for the persistent job store (app.core.jobs) and /api/jobs."""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app.core import jobs as jobs_mod
from app.core.config import settings
from app.core.jobs import JobStore


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "data" / "results")
    monkeypatch.delenv("SYNTRA_API_KEY", raising=False)
    settings.ensure_dirs()
    yield tmp_path


# --- JobStore ---------------------------------------------------------------
def test_create_get_update_roundtrip(tmp_path):
    store = JobStore(tmp_path / "j.db")
    job_id = store.create("predict", model_id="car_detection", tiles_processed=0)
    job = store.get(job_id)
    assert job["status"] == "queued" and job["kind"] == "predict"
    assert job["model_id"] == "car_detection" and job["job_id"] == job_id
    store.update(job_id, status="processing", tiles_processed=7)
    job = store.get(job_id)
    assert job["status"] == "processing" and job["tiles_processed"] == 7
    assert job["model_id"] == "car_detection"  # untouched fields survive a merge


def test_get_with_wrong_kind_is_none(tmp_path):
    store = JobStore(tmp_path / "j.db")
    job_id = store.create("train")
    assert store.get(job_id, "predict") is None
    assert store.get(job_id, "train") is not None
    assert store.get("nope") is None


def test_append_caps_list_length(tmp_path):
    store = JobStore(tmp_path / "j.db")
    job_id = store.create("dataset_export")
    for i in range(10):
        store.append(job_id, "log", f"line {i}", max_len=3)
    assert store.get(job_id)["log"] == ["line 7", "line 8", "line 9"]


def test_jobs_survive_a_new_store_on_the_same_file(tmp_path):
    path = tmp_path / "j.db"
    first = JobStore(path)
    job_id = first.create("predict", model_id="ship_detection")
    first.update(job_id, status="completed", result={"final_detection_count": 12})
    first.close()

    second = JobStore(path)  # what a server restart does
    job = second.get(job_id)
    assert job["status"] == "completed"
    assert job["result"]["final_detection_count"] == 12


def test_mark_interrupted_only_touches_active_jobs(tmp_path):
    store = JobStore(tmp_path / "j.db")
    running = store.create("train", status="processing")
    queued = store.create("predict")
    pip = store.create("environment", status="running")
    done = store.create("predict", status="completed")
    failed = store.create("predict", status="error")

    assert store.mark_interrupted() == 3
    for jid in (running, queued, pip):
        job = store.get(jid)
        assert job["status"] == "interrupted"
        assert "restarted" in job["error"]
    assert store.get(running)["previous_status"] == "processing"
    assert store.get(done)["status"] == "completed"
    assert store.get(failed)["status"] == "error"


def test_list_filters_and_orders_newest_first(tmp_path):
    store = JobStore(tmp_path / "j.db")
    a = store.create("predict")
    b = store.create("train")
    c = store.create("predict", status="completed")
    assert [j["job_id"] for j in store.list("predict")] == [c, a]
    assert [j["job_id"] for j in store.list(status="completed")] == [c]
    assert len(store.list()) == 3 and store.list()[-1]["job_id"] == a
    assert b in {j["job_id"] for j in store.list()}


def test_concurrent_updates_do_not_lose_writes(tmp_path):
    store = JobStore(tmp_path / "j.db")
    job_id = store.create("dataset_export")

    def worker(n):
        for i in range(50):
            store.append(job_id, "log", f"{n}-{i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.get(job_id)["log"]) == 200


def test_get_store_follows_data_dir():
    store = jobs_mod.get_store()
    assert store.db_path == (settings.DATA_DIR / "syntra.db").resolve()


# --- HTTP -------------------------------------------------------------------
def _client():
    from app.main import app
    return TestClient(app)


def test_restart_marks_running_jobs_interrupted_over_http():
    store = jobs_mod.get_store()
    job_id = store.create("predict", status="processing", model_id="car_detection")
    with _client() as client:  # entering the context runs the lifespan startup
        r = client.get(f"/api/predict/{job_id}")
        assert r.status_code == 200
        assert r.json()["status"] == "interrupted"


def test_prediction_job_is_persisted_and_listed():
    image = settings.UPLOAD_DIR / "scene.tif"
    image.write_bytes(b"not really a tiff")
    with _client() as client:
        r = client.post("/api/predict", json={"model_id": "car_detection", "image_path": str(image)})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        # The background task ran (TestClient runs it after the response). It
        # fails here because rasterio/ultralytics aren't installed or the file
        # isn't a raster, and that failure must be recorded, not lost.
        job = client.get(f"/api/predict/{job_id}").json()
        assert job["status"] in ("error", "completed")
        assert job["image_name"] == "scene.tif"

        listed = client.get("/api/jobs?kind=predict").json()
        assert [j["job_id"] for j in listed] == [job_id]
        assert client.get(f"/api/jobs/{job_id}").json()["kind"] == "predict"


def test_jobs_api_rejects_unknown_kind():
    with _client() as client:
        assert client.get("/api/jobs?kind=bogus").status_code == 400
        assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_job_ids_are_scoped_by_kind_over_http():
    train_id = jobs_mod.get_store().create("train", status="completed")
    with _client() as client:
        assert client.get(f"/api/predict/{train_id}").status_code == 404
        assert client.get(f"/api/training/train/{train_id}").status_code == 200


def test_cancel_finished_training_is_409():
    train_id = jobs_mod.get_store().create("train", status="done")
    with _client() as client:
        assert client.post(f"/api/training/train/{train_id}/cancel").status_code == 409


def test_export_cancel_is_persisted():
    export_id = jobs_mod.get_store().create("dataset_export", status="processing")
    with _client() as client:
        assert client.post(f"/api/training/export-dataset/{export_id}/cancel").status_code == 200
    assert jobs_mod.get_store().get(export_id)["cancel_requested"] is True
