"""The Library's approved model shows up as a detection model and runs the approved weights."""
import json

from fastapi.testclient import TestClient


def _version(data, group, version, names):
    run = data / "models" / group / version
    (run / "run" / "weights").mkdir(parents=True)
    (run / "run" / "weights" / "best.onnx").write_bytes(b"onnx")
    (run / "model.json").write_text(json.dumps({"dataset": group, "dataset_version": "v001"}))
    ds = data / "datasets" / group / "v001"
    ds.mkdir(parents=True, exist_ok=True)
    from app.training.dataset_builder import write_data_yaml
    write_data_yaml(str(ds / "data.yaml"), str(ds), names, "obb", test=True)
    return run


def _client(monkeypatch):
    monkeypatch.setenv("SYNTRA_AUTH", "off")
    from app.main import app
    return TestClient(app)


def _entry(client, model_id):
    body = client.get("/api/models").json()
    rows = body if isinstance(body, list) else body.get("models", [])
    return next((m for m in rows if m["model_id"] == model_id), None)


def test_approve_adds_the_class_model_and_rollback_removes_it(monkeypatch):
    from app.core import layout
    from app.core.config import settings

    data = settings.DATA_DIR
    run = _version(data, "boat", "v001", ["boat"])
    client = _client(monkeypatch)
    assert _entry(client, "class_boat") is None                       # nothing approved yet

    layout.set_approved("boat", "v001")
    m = _entry(client, "class_boat")
    assert m and m["version"] == "v001 approved" and m["status"] == "validation"
    assert "No test score yet" in m["notes"]

    (run / "run" / "test_metrics.json").write_text(json.dumps({"map50": 0.71, "precision": 0.8, "recall": 0.7, "f1": 0.75}))
    m = _entry(client, "class_boat")
    assert m["status"] == "ready" and "F1 0.75" in m["notes"]          # ready only with a measured score

    layout.set_approved("boat", None)
    assert _entry(client, "class_boat") is None


def test_build_model_uses_the_approved_weights_and_keeps_a_pinned_version():
    from app.core import layout
    from app.core.config import settings
    from app.models.custom_detector import CustomYoloDetector
    from app.models.factory import build_model

    data = settings.DATA_DIR
    _version(data, "boat", "v001", ["boat"])
    _version(data, "boat", "v002", ["boat"])
    layout.set_approved("boat", "v002")

    m = build_model("class_boat")
    assert isinstance(m, CustomYoloDetector)
    assert "v002" in m.weights_path and m.metadata.class_names == {0: "boat"}
    # a run started on v001 keeps v001 even after v002 is approved
    assert "v001" in build_model("class_boat", version="v001").weights_path


def test_a_version_without_weights_is_not_listed():
    from app.core import layout
    from app.core.config import settings
    from app.models.approved import get_approved

    (settings.DATA_DIR / "models" / "roof" / "v001").mkdir(parents=True)
    layout.set_approved("roof", "v001")
    assert get_approved("class_roof") is None


def test_a_hand_edited_data_yaml_does_not_break_the_model_list():
    from app.core import layout
    from app.core.config import settings
    from app.models.approved import get_approved

    run = _version(settings.DATA_DIR, "boat", "v001", ["boat"])
    (settings.DATA_DIR / "datasets" / "boat" / "v001" / "data.yaml").write_text("names:\n  0: boat\n")   # unquoted
    layout.set_approved("boat", "v001")
    m = get_approved("class_boat")
    assert m and m.class_names == {0: "boat"} and run.name == "v001"
