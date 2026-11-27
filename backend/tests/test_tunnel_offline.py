"""Requests through a tunnel or proxy connect from this machine but are not local."""
import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("header", ["cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-real-ip", "forwarded"])
def test_proxied_request_is_not_local(monkeypatch, header):
    monkeypatch.delenv("SYNTRA_ALLOW_ENV_ACTIONS", raising=False)
    from fastapi import FastAPI, Request
    from app.core.security import environment_actions_allowed, is_loopback
    app = FastAPI()

    @app.get("/x")
    def x(request: Request):
        return {"local": is_loopback(request), "env": environment_actions_allowed(request)}

    c = TestClient(app)
    assert c.get("/x").json() == {"local": True, "env": True}               # this machine
    assert c.get("/x", headers={header: "203.0.113.7"}).json() == {"local": False, "env": False}


def test_first_account_cannot_be_claimed_through_a_tunnel(monkeypatch):
    monkeypatch.setenv("SYNTRA_AUTH", "on")
    from app.main import app
    c = TestClient(app)
    r = c.post("/api/auth/signup", json={"name": "x", "email": "a@example.test", "password": "longenough1"},
               headers={"cf-connecting-ip": "203.0.113.7"})
    assert r.status_code == 403
