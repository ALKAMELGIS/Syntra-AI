"""Model performance follows a training run epoch by epoch, from the run's own events."""


def test_epoch_event_reads_validation_metrics_and_losses():
    from app.training.train_worker import epoch_event

    ev = epoch_event(3, 30, {"train/box_loss": 1.0, "train/seg_loss": 2.0, "train/cls_loss": 0.5},
                     {"metrics/precision(B)": 0.4, "metrics/precision(M)": 0.6, "metrics/recall(M)": 0.5,
                      "metrics/mAP50(M)": 0.45, "metrics/mAP50-95(M)": 0.2, "val/box_loss": 1.2, "val/seg_loss": 2.4,
                      "val/cls_loss": 0.6, "fitness": 0.3}, 61.04)
    assert ev["epoch"] == 3 and ev["epochs"] == 30
    assert ev["train_loss"] == 3.5 and ev["val_loss"] == 4.2
    assert ev["precision"] == 0.6 and ev["map50"] == 0.45           # mask metrics for segmentation
    assert ev["elapsed_s"] == 61.0


def test_a_running_job_shows_its_epochs_live(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    settings.ensure_dirs()
    from app.core.jobs import get_store
    from app.main import app

    c = TestClient(app)
    s = get_store()
    jid = s.create("train", status="processing", model_group="ships", model_version="v002",
                   config={"epochs": 30, "device": "cpu"})
    s.create("train", status="error", error="never started")         # no epoch: not listed
    for e, m in ((1, 0.1), (2, 0.3)):
        s.append(jid, "history", {"type": "epoch", "epoch": e, "epochs": 30, "train_loss": 5 - e,
                                  "val_loss": 6 - e, "precision": m, "recall": m, "map50": m}, max_len=500)
    lst = c.get("/api/models/performance").json()
    run = next(x for x in lst if x.get("run_id") == jid)
    assert run["live"] and run["epoch"] == 2 and run["planned_epochs"] == 30 and run["name"] == "ships v002"
    assert sum(1 for x in lst if x.get("run_id")) == 1
    perf = c.get(f"/api/models/run:{jid}/performance").json()
    assert [e["epoch"] for e in perf["epochs"]] == [1, 2] and perf["best"] == {"epoch": 2, "map50": 0.3}
    # an older event without train_loss still draws: the parts are summed
    s.append(jid, "history", {"type": "epoch", "epoch": 3, "box_loss": 1.0, "cls_loss": 0.5, "dfl_loss": 0.5}, max_len=500)
    assert c.get(f"/api/models/run:{jid}/performance").json()["epochs"][-1]["train_loss"] == 2.0
    assert c.get("/api/models/run:nope/performance").status_code == 404
