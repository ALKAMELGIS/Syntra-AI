"""Google satellite base map: key kept server-side, tiles passed through, not stored."""
import io
import json

import pytest


class FakeResp(io.BytesIO):
    def __init__(self, data, ctype="application/json"):
        super().__init__(data)
        self.headers = {"Content-Type": ctype}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def c(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import basemaps
    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    basemaps._session.clear()
    calls = []

    def fake_open(req, timeout=0):
        calls.append(req.full_url)
        if "createSession" in req.full_url:
            if "key=bad" in req.full_url:
                import urllib.error
                raise urllib.error.HTTPError(req.full_url, 403, "no", {}, io.BytesIO(b"API not enabled"))
            assert json.loads(req.data)["mapType"] == "satellite"
            return FakeResp(json.dumps({"session": "S1", "expiry": "9999999999"}).encode())
        return FakeResp(b"\xff\xd8jpeg", "image/jpeg")

    monkeypatch.setattr(basemaps, "_open", fake_open)
    from app.main import app
    return TestClient(app), calls, tmp_path


def test_key_is_checked_kept_and_never_returned(c):
    client, calls, tmp = c
    assert client.get("/api/basemaps/google").json() == {"configured": False}
    assert client.get("/api/basemaps/google/tile/3/4/2").status_code == 409
    r = client.put("/api/basemaps/google", json={"key": "bad"})
    assert r.status_code == 400 and "Map Tiles API" in r.json()["detail"]
    assert client.get("/api/basemaps/google").json() == {"configured": False}
    r = client.put("/api/basemaps/google", json={"key": "AIzaGood"})
    assert r.json() == {"configured": True} and "AIzaGood" not in r.text


def test_tiles_pass_through_without_storing(c):
    client, calls, tmp = c
    client.put("/api/basemaps/google", json={"key": "AIzaGood"})
    files_before = sorted(p.name for p in tmp.rglob("*"))
    t = client.get("/api/basemaps/google/tile/3/4/2")
    assert t.status_code == 200 and t.content == b"\xff\xd8jpeg" and t.headers["cache-control"] == "no-store"
    assert "2dtiles/3/4/2?session=S1&key=AIzaGood" in calls[-1]
    assert sorted(p.name for p in tmp.rglob("*")) == files_before            # nothing written
    assert client.get("/api/basemaps/google/tile/3/9/2").status_code == 400  # x outside the world at z3
