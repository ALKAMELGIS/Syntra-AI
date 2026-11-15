"""Training samples to Hugging Face: token kept locally, private repos only, one folder per push."""
import json

import pytest

pytest.importorskip("huggingface_hub")


class FakeApi:
    """Stands in for huggingface_hub.HfApi: records calls, no network."""
    repos: dict = {}
    uploads: list = []

    def __init__(self, token=None):
        self.token = token

    def whoami(self):
        if self.token != "hf_good":
            raise RuntimeError("401 Unauthorized")
        return {"name": "ali"}

    def repo_info(self, repo_id, repo_type=None):
        from huggingface_hub.errors import RepositoryNotFoundError
        if repo_id not in FakeApi.repos:
            err = RepositoryNotFoundError.__new__(RepositoryNotFoundError)   # its __init__ wants a live HTTP response
            Exception.__init__(err, "404 not found")
            raise err
        return type("Info", (), {"private": FakeApi.repos[repo_id]})()

    def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
        FakeApi.repos[repo_id] = private

    def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type, commit_message):
        from pathlib import Path
        files = sorted(str(p.relative_to(folder_path)) for p in Path(folder_path).rglob("*") if p.is_file())
        FakeApi.uploads.append({"repo": repo_id, "path": path_in_repo, "files": files,
                                "source": json.loads((Path(folder_path) / "syntra_source.json").read_text())})
        return type("Commit", (), {"oid": "abc123"})()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    import huggingface_hub
    from fastapi.testclient import TestClient

    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    settings.ensure_dirs()
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    FakeApi.repos, FakeApi.uploads = {}, []
    from app.main import app
    return TestClient(app), tmp_path


def _export(tmp_path, n_images=3):
    from app.core.jobs import get_store

    d = tmp_path / "datasets" / "exp1"
    for split in ("train", "val"):
        (d / "images" / split).mkdir(parents=True)
        (d / "labels" / split).mkdir(parents=True)
        for i in range(n_images):
            (d / "images" / split / f"t{i}.jpg").write_bytes(b"jpg")
            (d / "labels" / split / f"t{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (d / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [ship]\n")
    jid = get_store().create("dataset_export", image_id="img1", task="obb", class_names=["ship"], image_name="harbour.tif")
    get_store().update(jid, status="completed", data_yaml=str(d / "data.yaml"), info={"tiles": 6})
    return jid


def test_token_is_checked_kept_locally_and_never_returned(env):
    c, tmp = env
    assert c.put("/api/samples/hf", json={"token": "hf_bad"}).status_code == 400
    r = c.put("/api/samples/hf", json={"token": "hf_good"}).json()
    assert r == {"configured": True, "user": "ali", "repo": "ali/syntra-samples", "library": True, "auto_backup": False}
    assert "hf_good" not in json.dumps(c.get("/api/samples/hf").json())
    assert json.loads((tmp / "secrets.json").read_text())["hf_token"] == "hf_good"
    assert c.put("/api/samples/hf", json={"repo": "no-slash"}).status_code == 400
    assert c.put("/api/samples/hf", json={"token": ""}).json()["configured"] is False


def test_push_creates_a_private_repo_and_one_folder_per_push(env):
    c, tmp = env
    c.put("/api/samples/hf", json={"token": "hf_good"})
    jid = _export(tmp)
    r = c.post("/api/samples/push", json={"dataset_job_id": jid}).json()
    st = c.get(f"/api/samples/push/{r['job_id']}").json()
    assert st["status"] == "completed", st
    assert FakeApi.repos["ali/syntra-samples"] is True                      # created private
    up = FakeApi.uploads[0]
    assert up["path"].endswith("_harbour_obb") and "data.yaml" in up["files"]
    assert "shards/train-000.tar" in up["files"] and "syntra_source.json" in up["files"]   # tiles packed
    assert up["source"]["classes"] == ["ship"] and up["source"]["task"] == "obb"
    assert st["result"]["url"].startswith("https://huggingface.co/datasets/ali/syntra-samples/tree/main/")


def test_push_refused_into_a_public_repo(env):
    c, tmp = env
    c.put("/api/samples/hf", json={"token": "hf_good", "repo": "ali/open-samples"})
    FakeApi.repos["ali/open-samples"] = False                                 # exists, public
    r = c.post("/api/samples/push", json={"dataset_job_id": _export(tmp)}).json()
    st = c.get(f"/api/samples/push/{r['job_id']}").json()
    assert st["status"] == "error" and "public" in st["error"] and not FakeApi.uploads


def test_push_needs_a_token_and_a_finished_export(env):
    c, tmp = env
    assert c.post("/api/samples/push", json={"dataset_job_id": "x"}).status_code == 409
    c.put("/api/samples/hf", json={"token": "hf_good"})
    assert c.post("/api/samples/push", json={"dataset_job_id": "missing"}).status_code == 409


def test_folder_over_the_hub_limit_is_refused(tmp_path, monkeypatch):
    import app.engine.hf_push as hp

    d = tmp_path / "ds"
    (d / "images").mkdir(parents=True)
    (d / "data.yaml").write_text("names: [x]\n")
    for i in range(12):
        (d / "images" / f"{i}.jpg").write_bytes(b"x")
    monkeypatch.setattr(hp, "MAX_FILES_PER_FOLDER", 10)
    with pytest.raises(hp.PushError, match="images holds 12 files"):
        hp.check_folder_sizes(d)


def test_errors_say_whether_the_token_or_the_network_is_the_problem():
    from app.engine.hf_push import _explain

    class Resp:
        def __init__(self, code):
            self.status_code = code

    class HubError(Exception):
        def __init__(self, msg, code):
            super().__init__(msg)
            self.response = Resp(code)

    class ProxyError(Exception):
        pass

    assert "did not accept this token" in _explain(HubError("Invalid credentials", 401))
    assert "proxy or firewall" in _explain(ProxyError("403 Forbidden"))
    assert "internet connection" in _explain(ConnectionError("Name or service not known"))


def test_stage_packs_tiles_into_shards_and_adds_the_original_labels(tmp_path, monkeypatch):
    import tarfile

    from app.engine import hf_push

    ds = tmp_path / "ds"
    for split in ("train", "val", "test"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
        for k in range(5):
            (ds / "images" / split / f"t{k}.jpg").write_bytes(b"j")
            (ds / "labels" / split / f"t{k}.txt").write_text("0 0.5 0.5 0.1 0.1")
    (ds / "data.yaml").write_text("train: images/train")
    lib = tmp_path / "classes" / "ship" / "labels"
    lib.mkdir(parents=True)
    (lib / "7f3a91c2.geojson").write_text("{}")
    monkeypatch.setattr(hf_push, "SHARD_FILES", 4)
    out = tmp_path / "up"
    n = hf_push.stage(ds, out, {"label_files": ["ship/labels/7f3a91c2.geojson"], "classes_dir": str(tmp_path / "classes")})
    assert n == 9                                                            # 10 files per split, 4 per shard
    with tarfile.open(out / "shards" / "train-000.tar") as t:
        assert t.getnames()[0].startswith(("images/train/", "labels/train/"))
    assert (out / "library" / "ship" / "labels" / "7f3a91c2.geojson").exists() and (out / "data.yaml").exists()
