"""GPU training on RunPod, without RunPod: the real pod script runs here against
stand-in huggingface_hub / ultralytics packages, and the watcher runs against a
stand-in RunPod that records every call."""
import base64
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

FAKE_HUB = '''
import json, os, shutil
from pathlib import Path
STORE = Path(os.environ["FAKE_HUB_DIR"])
class HfApi:
    def __init__(self, token=None): self.token = token
    def _p(self, repo, path): return STORE / repo / path
    def file_exists(self, repo, path, repo_type=None): return self._p(repo, path).exists()
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
        dst = self._p(repo_id, path_in_repo); dst.parent.mkdir(parents=True, exist_ok=True)
        data = path_or_fileobj if isinstance(path_or_fileobj, bytes) else Path(path_or_fileobj).read_bytes()
        dst.write_bytes(data)
    def hf_hub_download(self, repo, path, repo_type=None, force_download=False):
        return str(self._p(repo, path))
def snapshot_download(repo_id, repo_type=None, allow_patterns=None, local_dir=None, token=None):
    folder = allow_patterns[0].split("/")[0]
    shutil.copytree(STORE / repo_id / folder, Path(local_dir) / folder, dirs_exist_ok=True)
    return local_dir
'''

FAKE_ULTRA = '''
from pathlib import Path
class _T:
    def __init__(self, n): self.epochs, self.epoch, self.metrics, self.loss_items = n, 0, {}, [1.0, 0.5, 0.2]
class _R:
    def __init__(self, d): self.save_dir = d
class YOLO:
    def __init__(self, w): self.w, self.cbs = w, []
    def add_callback(self, name, fn): self.cbs.append(fn)
    def train(self, data, epochs, imgsz, batch, workers, device, optimizer, lr0, patience, project, name, exist_ok, verbose):
        assert Path(data).exists() and "task: obb" in Path(data).read_text()
        d = Path(project) / name; (d / "weights").mkdir(parents=True, exist_ok=True)
        t = _T(epochs)
        for e in range(epochs):
            t.epoch = e; t.metrics = {"metrics/mAP50(B)": 0.1 * (e + 1)}
            for cb in self.cbs: cb(t)
        (d / "weights" / "best.pt").write_bytes(b"PT")
        (d / "results.csv").write_text("epoch,metrics/mAP50(B)\\n1,0.1\\n")
        return _R(d)
    def export(self, format, imgsz):
        out = Path(self.w).with_suffix(".onnx"); out.write_bytes(b"ONNX"); return str(out)
'''


@pytest.fixture()
def fakes(tmp_path):
    mods = tmp_path / "mods"
    (mods / "huggingface_hub").mkdir(parents=True)
    (mods / "huggingface_hub" / "__init__.py").write_text(FAKE_HUB)
    (mods / "ultralytics").mkdir()
    (mods / "ultralytics" / "__init__.py").write_text(FAKE_ULTRA)
    hub = tmp_path / "hub"
    ds = hub / "ali/syntra-samples" / "20261115_harbour_obb"
    (ds / "images" / "train").mkdir(parents=True)
    (ds / "images" / "train" / "a.jpg").write_bytes(b"x")
    (ds / "data.yaml").write_text('path: "C:/Users/Use/x"\ntrain: images/train\nval: images/val\ntask: obb\nnc: 1\nnames:\n  0: "ship"\n')
    return mods, hub


def _run_pod(tmp_path, mods, hub, run_id="run1"):
    from app.engine.cloud_train import pod_script

    script = tmp_path / "pod.py"
    script.write_text(pod_script().replace('_pl.Path("/workspace/syntra")', f'_pl.Path(r"{tmp_path / "ws"}")'))
    env = {**os.environ, "PYTHONPATH": str(mods), "FAKE_HUB_DIR": str(hub), "HF_TOKEN": "t",
           "HF_REPO": "ali/syntra-samples", "HF_FOLDER": "20261115_harbour_obb", "RUN_ID": run_id,
           "JOB_JSON": json.dumps({"model": "yolo11n-obb.pt", "epochs": 3, "imgsz": 640, "batch": 16, "workers": 2})}
    return subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=60)


def test_pod_script_trains_uploads_and_marks_done(tmp_path, fakes):
    mods, hub = fakes
    r = _run_pod(tmp_path, mods, hub)
    assert r.returncode == 0, r.stdout + r.stderr
    out = hub / "ali/syntra-samples" / "runs" / "run1"
    assert json.loads((out / "done.json").read_text())["status"] == "done"
    assert (out / "best.pt").read_bytes() == b"PT" and (out / "best.onnx").read_bytes() == b"ONNX"
    epochs = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines() if '"epoch"' in x]
    assert [e["epoch"] for e in epochs if e["type"] == "epoch"] == [1, 2, 3]


def test_pod_restart_does_not_train_twice(tmp_path, fakes):
    mods, hub = fakes
    assert _run_pod(tmp_path, mods, hub).returncode == 0
    (hub / "ali/syntra-samples" / "runs" / "run1" / "best.pt").write_bytes(b"FIRST")
    r = _run_pod(tmp_path, mods, hub)                       # the container started again
    assert "already finished" in r.stdout
    assert (hub / "ali/syntra-samples" / "runs" / "run1" / "best.pt").read_bytes() == b"FIRST"


def test_pod_failure_still_leaves_done_json(tmp_path, fakes):
    mods, hub = fakes
    (hub / "ali/syntra-samples" / "20261115_harbour_obb" / "data.yaml").write_text("names: [ship]\n")   # no task: obb
    r = _run_pod(tmp_path, mods, hub)
    done = json.loads((hub / "ali/syntra-samples" / "runs" / "run1" / "done.json").read_text())
    assert r.returncode == 1 and done["status"] == "error" and done["message"]


def test_pod_shell_always_tries_to_terminate():
    from app.engine.cloud_train import pod_args

    a = pod_args()
    train_at = a.index("python /syntra_train.py")
    assert a.index("/action") > train_at and '{"action":"terminate"}' in a.replace("'\"'\"'", "")
    assert 'timeout "$MAX_SECONDS"' in a
    assert subprocess.run(["bash", "-n", "-c", a]).returncode == 0


class FakeRunPod:
    def __init__(self):
        self.calls = []
        self.status = "RUNNING"

    def create_pod(self, **kw):
        self.calls.append(("create", kw))
        return {"id": "pod1", "status": "PROVISIONING", "cost": 0.34}

    def get_pod(self, pod_id):
        self.calls.append(("get", pod_id))
        return {"id": pod_id, "status": self.status, "cost": 0.34}

    def terminate(self, pod_id):
        self.calls.append(("terminate", pod_id))


def _hub_factory(hub, monkeypatch):
    monkeypatch.setenv("FAKE_HUB_DIR", str(hub))
    import importlib.util

    spec = importlib.util.spec_from_loader("fake_hub", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(FAKE_HUB, mod.__dict__)
    return lambda token: mod.HfApi(token)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.core.jobs import get_store

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    settings.ensure_dirs()
    return get_store()


def test_watcher_downloads_the_model_and_terminates_the_pod(tmp_path, fakes, store, monkeypatch):
    from app.engine import cloud_train

    mods, hub = fakes
    _run_pod(tmp_path, mods, hub, run_id="JOB")                 # what the GPU would leave on the Hub
    jid = store.create("train", job_id="JOB", where="runpod")
    rp = FakeRunPod()
    monkeypatch.setattr(cloud_train, "watch", lambda *a, **k: None)   # start() only; watch below in this thread
    cloud_train.start("JOB", runpod=rp, hf_token="t", hf_repo="ali/syntra-samples", hf_folder="20261115_harbour_obb",
                      job_cfg={"model": "yolo11n-obb.pt", "epochs": 3, "imgsz": 640, "batch": 16}, gpu="rtx4090",
                      cloud="COMMUNITY", max_hours=1, store=store, run_dir=tmp_path / "data" / "training_runs" / "JOB")
    create = rp.calls[0][1]
    assert create["gpu_id"] == "NVIDIA GeForce RTX 4090" and create["cloud"] == "COMMUNITY"
    assert create["env"]["MAX_SECONDS"] == "3600" and base64.b64decode(create["env"]["SYNTRA_SCRIPT"])
    cloud_train._watch(jid, rp, "t", store, _hub_factory(hub, monkeypatch), poll_s=0)
    job = store.get(jid)
    assert job["status"] == "done" and Path(job["result"]["best"]).read_bytes() == b"PT"
    assert Path(job["result"]["onnx"]).read_bytes() == b"ONNX"
    assert ("terminate", "pod1") in rp.calls and job["cost_per_hour"] == 0.34
    assert [h["epoch"] for h in job["history"]] == [1, 2, 3]


def test_watcher_stops_the_pod_at_the_time_limit(tmp_path, store, monkeypatch, fakes):
    import time

    from app.engine import cloud_train

    _, hub = fakes
    jid = store.create("train", where="runpod", status="processing", pod_id="pod9", hf_repo="ali/syntra-samples",
                       run_dir=str(tmp_path), started_at=time.time() - 7200, cost_per_hour=0.34, deadline=time.time() - 1)
    rp = FakeRunPod()
    cloud_train._watch(jid, rp, "t", store, _hub_factory(hub, monkeypatch), poll_s=0)
    job = store.get(jid)
    assert job["status"] == "error" and "time limit" in job["error"]
    assert ("terminate", "pod9") in rp.calls and job["cost_usd"] == pytest.approx(0.68, abs=0.01)


def test_gpu_runs_are_not_marked_interrupted_at_restart(store):
    jid = store.create("train", where="runpod", status="processing", pod_id="p")
    local = store.create("train", status="processing")
    store.mark_interrupted()
    assert store.get(jid)["status"] == "processing" and store.get(local)["status"] == "interrupted"


def test_runpod_client_sends_the_v2_body_and_explains_errors():
    import io
    import urllib.error

    from app.engine.runpod_client import RunPod, RunPodError

    sent = []

    def opener(req, timeout=None):
        sent.append((req.get_method(), req.full_url, json.loads(req.data) if req.data else None, dict(req.header_items())))
        if req.full_url.endswith("/pods/bad/action"):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"title":"Unauthorized","status":401}'))

        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"id": "p1", "status": "PROVISIONING", "cost": 0.34}'
        return R()

    rp = RunPod("k", opener=opener)
    rp.create_pod(name="n", image="img", gpu_id="NVIDIA GeForce RTX 4090", cloud="COMMUNITY", disk_gb=40, env={"A": "1"}, args="bash -c 'x'")
    m, url, body, headers = sent[0]
    assert (m, url) == ("POST", "https://api.runpod.io/v2/pods")
    assert body == {"name": "n", "image": "img", "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1}, "cloud": "COMMUNITY",
                    "disk": 40, "env": {"A": "1"}, "args": "bash -c 'x'", "ports": []}
    assert headers["Authorization"] == "Bearer k"
    with pytest.raises(RunPodError, match="did not accept this API key"):
        rp.terminate("bad")
