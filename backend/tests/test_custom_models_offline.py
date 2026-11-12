"""
Offline tests for closing the Train -> Detect loop (gap ③):
  - app.models.custom_registry: sidecar-JSON persistence of trained models.
  - app.models.registry: static CATALOG + dynamic custom models merge
    transparently through list_catalog()/get_metadata().
  - app.models.factory: build_model() dispatches a custom model_id to
    CustomYoloDetector with the registered weights path, without touching
    any static-catalog dispatch logic.
  - app.training.dataset_builder.read_data_yaml_names: reads back exactly
    the class-name order write_data_yaml wrote, with no PyYAML dependency.

Run directly: python tests/test_custom_models_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.models import custom_registry  # noqa: E402
from app.models.custom_detector import CustomYoloDetector  # noqa: E402
from app.models.factory import build_model  # noqa: E402
from app.models.registry import CATALOG, get_metadata, list_catalog  # noqa: E402
from app.training.dataset_builder import read_data_yaml_names, write_data_yaml  # noqa: E402


def _register_one(tmp_path, *, names=("tree", "building")):
    yaml_path = tmp_path / "data.yaml"
    write_data_yaml(str(yaml_path), str(tmp_path), list(names), "detect")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    read_names = read_data_yaml_names(str(yaml_path))
    return custom_registry.register_custom_model(
        tmp_path, name="Test Trained Model", weights_path=str(weights),
        class_names=dict(enumerate(read_names)), source_job_id="job123", data_yaml=str(yaml_path),
    ), yaml_path, read_names


def test_read_data_yaml_names_roundtrips_write_data_yaml(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    write_data_yaml(str(yaml_path), str(tmp_path), ["palm tree", "car", "a, b"], "detect")
    assert read_data_yaml_names(str(yaml_path)) == ["palm tree", "car", "a, b"]


def test_register_rejects_missing_weights(tmp_path):
    try:
        custom_registry.register_custom_model(
            tmp_path, name="x", weights_path=str(tmp_path / "nope.pt"), class_names={0: "a"},
        )
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_register_rejects_empty_class_names(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    try:
        custom_registry.register_custom_model(tmp_path, name="x", weights_path=str(weights), class_names={})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_registered_model_round_trips_through_disk(tmp_path):
    entry, _, names = _register_one(tmp_path)
    reloaded = custom_registry.get_custom_model(tmp_path, entry.model_id)
    assert reloaded is not None
    assert reloaded.name == "Test Trained Model"
    assert reloaded.class_names == dict(enumerate(names))
    assert all(isinstance(k, int) for k in reloaded.class_names)  # JSON int-key coercion survived


def test_list_catalog_merges_in_custom_models_without_touching_static_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    static_count = len(CATALOG)
    assert len(list_catalog()) == static_count

    entry, _, _ = _register_one(tmp_path)
    cat = list_catalog()
    assert len(cat) == static_count + 1
    ids = [m.model_id for m in cat]
    assert entry.model_id in ids
    # static catalog itself (the dict object) must be untouched
    assert len(CATALOG) == static_count
    assert entry.model_id not in CATALOG


def test_get_metadata_finds_custom_model_and_reports_it_honestly(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    entry, _, names = _register_one(tmp_path)
    meta = get_metadata(entry.model_id)
    assert meta.class_names == dict(enumerate(names))
    assert "no benchmark report" in meta.notes.lower()


def test_build_model_dispatches_custom_model_to_custom_detector(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    entry, _, _ = _register_one(tmp_path)
    model = build_model(entry.model_id)
    assert isinstance(model, CustomYoloDetector)
    assert model.weights_path == entry.weights_path


def test_delete_custom_model_removes_it_from_the_merged_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    entry, _, _ = _register_one(tmp_path)
    assert entry.model_id in [m.model_id for m in list_catalog()]
    assert custom_registry.delete_custom_model(tmp_path, entry.model_id) is True
    assert entry.model_id not in [m.model_id for m in list_catalog()]
    assert custom_registry.delete_custom_model(tmp_path, entry.model_id) is False


def _run_all():
    import inspect
    import tempfile

    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        sig = inspect.signature(t)
        with tempfile.TemporaryDirectory() as td:
            kwargs = {}
            if "tmp_path" in sig.parameters:
                kwargs["tmp_path"] = Path(td)
            if "monkeypatch" in sig.parameters:
                class _MP:
                    def setattr(self, obj, name, value):
                        setattr(obj, name, value)
                kwargs["monkeypatch"] = _MP()
            t(**kwargs)
        passed += 1
        print(f"  OK  {t.__name__}")
    print(f"\n{passed}/{len(tests)} custom-model tests passed.")


if __name__ == "__main__":
    _run_all()
