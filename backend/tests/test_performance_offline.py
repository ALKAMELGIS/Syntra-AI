"""Model performance: training history from results.csv and from shipped pack metrics."""
import json

from fastapi.testclient import TestClient

CSV = """epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0
1,10,1.5,2.0,1.2,0.30,0.20,0.25,0.10,1.4,1.9,1.1,0.01
2,20,1.2,1.5,1.1,0.50,0.40,0.45,0.20,1.3,1.6,1.05,0.01
3,30,1.0,1.1,1.0,0.60,0.55,0.41,0.21,1.35,1.7,1.08,0.01
"""


def test_parse_results_csv_sums_losses_and_reads_box_metrics(tmp_path):
    from app.engine.performance import parse_results_csv

    p = tmp_path / "results.csv"
    p.write_text(CSV)
    ep = parse_results_csv(p)
    assert [e["epoch"] for e in ep] == [1, 2, 3]
    assert ep[0]["train_loss"] == 4.7 and ep[0]["val_loss"] == 4.4
    assert ep[1]["map50"] == 0.45 and ep[2]["recall"] == 0.55


def test_trained_model_performance_over_http(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    run = tmp_path / "training_runs" / "job1" / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"pt")
    (run / "results.csv").write_text(CSV)
    (tmp_path / "custom_models.json").write_text(json.dumps([{
        "model_id": "custom_palm", "name": "Palms, site A", "weights_path": str(run / "weights" / "best.pt"),
        "class_names": {"0": "palm"}, "task": "detect"}]))
    with TestClient(app) as c:
        r = c.get("/api/models/custom_palm/performance")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["best"] == {"epoch": 2, "map50": 0.45}
        assert len(body["epochs"]) == 3
        assert any(m["model_id"] == "custom_palm" for m in c.get("/api/models/performance").json())
        assert c.get("/api/models/ship_detection/performance").status_code == 404
        assert c.get("/api/models/nope/performance").status_code == 404


def test_drone_pack_ships_its_metrics():
    from app.engine.performance import model_performance

    perf = model_performance("drone_objects")
    assert perf and perf["per_class"]["vehicle"] > 0.9
    assert set(perf["hidden_classes"]) == {"fence", "bicycle"}
    assert perf["best"]["map50"] > 0.4
