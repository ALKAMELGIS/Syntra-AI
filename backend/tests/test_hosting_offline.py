"""Train > Setup asks the engine where it runs: a Space on ZeroGPU can't train."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.delenv("SYNTRA_AUTH", raising=False)
    from app.main import app
    return TestClient(app)


def test_local_engine(client, monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    r = client.get("/api/compute/hosting").json()
    assert r["space_id"] is None and r["zero_gpu"] is False


def test_zero_gpu_space_is_not_a_training_gpu(client, monkeypatch):
    from app.api import compute
    monkeypatch.setenv("SPACE_ID", "Alkamelgis/SYNTRA-AI")
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    monkeypatch.setattr(compute, "list_devices", lambda: {"gpu": [{"index": 0, "name": "RTX PRO 6000"}]})
    r = client.get("/api/compute/hosting").json()
    assert r["space_id"] == "Alkamelgis/SYNTRA-AI" and r["zero_gpu"] is True and r["train_gpu"] is False
    monkeypatch.delenv("SPACES_ZERO_GPU")
    assert client.get("/api/compute/hosting").json()["train_gpu"] is True      # upgraded to a T4/L4
