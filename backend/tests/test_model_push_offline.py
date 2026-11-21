"""A Library model version to Hugging Face: token needed, version checked, card written, private by default."""
import json

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    settings.ensure_dirs()
    run = tmp_path / "models" / "date_palm" / "v001"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"pt")
    (run / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.81\n2,0.93\n")
    (run / "model.json").write_text(json.dumps({"dataset": "date_palm", "dataset_version": "v001", "version": "v001"}))
    ds = tmp_path / "datasets" / "date_palm" / "v001"
    (ds / "images" / "train").mkdir(parents=True)
    (ds / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: date_palm\n")
    (ds / "manifest.json").write_text(json.dumps({"source": "Palm-Tree-Dataset (Al Ain)"}))
    from app.main import app
    return TestClient(app), tmp_path


def test_push_needs_token_and_a_real_version(env, monkeypatch):
    c, tmp = env
    assert c.post("/api/library/date_palm/push", json={"version": "v001"}).status_code == 409
    from app.core.secrets import set_secret
    set_secret("hf_token", "hf_x"); set_secret("hf_user", "ali")
    assert c.post("/api/library/date_palm/push", json={"version": "v009"}).status_code == 404
    assert c.post("/api/library/..%2Fx/push", json={"version": "v001"}).status_code in (404, 405)

    seen = {}
    from app.engine import hf_push

    def fake_push(model_dir, dataset_dir, repo_id, token, card, public=False, progress=None):
        seen.update(model_dir=model_dir, dataset_dir=dataset_dir, repo=repo_id, card=card, public=public)
        return {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}", "private": not public}

    monkeypatch.setattr(hf_push, "push_model", fake_push)
    r = c.post("/api/library/date_palm/push", json={"version": "v001"})
    assert r.status_code == 200 and r.json()["repo"] == "ali/syntra-date-palm"
    job = c.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job
    assert seen["public"] is False and seen["dataset_dir"].name == "v001"
    assert "0.93" in seen["card"] and "date_palm" in seen["card"] and "Palm-Tree-Dataset" in seen["card"]
    m = c.get("/api/library").json()
    assert (tmp / "models" / "date_palm" / "v001" / "pushed.json").exists()


def test_push_model_stages_weights_data_and_card(env, monkeypatch, tmp_path):
    from app.engine import hf_push

    calls = {}

    class FakeApi:
        def create_repo(self, repo_id, repo_type, private, exist_ok):
            calls["private"] = private
        def repo_info(self, repo_id, repo_type):
            return type("I", (), {"private": calls["private"]})()
        def upload_folder(self, folder_path, repo_id, repo_type, commit_message):
            import os
            calls["files"] = sorted(os.path.relpath(os.path.join(r, f), folder_path)
                                    for r, _, fs in os.walk(folder_path) for f in fs)
            return type("C", (), {"oid": "abc"})()

    monkeypatch.setattr(hf_push, "_api", lambda token: FakeApi())
    c, tmp = env
    run, ds = tmp / "models" / "date_palm" / "v001", tmp / "datasets" / "date_palm" / "v001"
    res = hf_push.push_model(run, ds, "ali/m", "t", "# card\n")
    assert res["private"] is True and calls["private"] is True
    assert {"best.pt", "results.csv", "README.md", "model.json", os.path.join("data", "data.yaml")} <= set(calls["files"])
    res = hf_push.push_model(run, None, "ali/m", "t", "# card\n", public=True)
    assert res["private"] is False and not any(f.startswith("data") for f in calls["files"])


import os  # noqa: E402
