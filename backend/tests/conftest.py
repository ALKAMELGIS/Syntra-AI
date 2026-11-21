"""Every test gets its own data folder, so nothing is written to backend/data."""
import pytest


@pytest.fixture(autouse=True)
def _own_data_dir(tmp_path_factory, monkeypatch):
    from app.core.config import settings

    root = tmp_path_factory.mktemp("data")
    monkeypatch.setattr(settings, "DATA_DIR", root)
    monkeypatch.setattr(settings, "UPLOAD_DIR", root / "images")
    monkeypatch.setattr(settings, "RESULTS_DIR", root / "results")
