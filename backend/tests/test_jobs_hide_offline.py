"""Old runs: listed per image, and a removed run is not restored again."""


def test_list_by_image_and_hide(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    settings.ensure_dirs()
    from app.core.jobs import get_store
    from app.main import app

    c = TestClient(app)
    s = get_store()
    a1 = s.create("predict", model_id="ships", image_id="img-a", status="completed")
    s.create("predict", model_id="ships", image_id="img-b", status="completed")
    a2 = s.create("predict", model_id="cars", image_id="img-a", status="completed")
    ids = {j["job_id"] for j in c.get("/api/jobs?kind=predict&image_id=img-a").json()}
    assert {a1, a2} <= ids and all(j["image_id"] == "img-a" for j in c.get("/api/jobs?kind=predict&image_id=img-a").json())
    assert c.post(f"/api/jobs/{a1}/hide").json()["hidden"] is True
    ids = {j["job_id"] for j in c.get("/api/jobs?kind=predict&image_id=img-a").json()}
    assert a1 not in ids and a2 in ids
    assert a1 in {j["job_id"] for j in c.get("/api/jobs?kind=predict&image_id=img-a&include_hidden=true").json()}
    assert c.post("/api/jobs/nope/hide").status_code == 404
