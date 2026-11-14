"""Accounts: first-run, sign up, sign in, sessions, throttling, and the API gate."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app.core import auth
    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.delenv("SYNTRA_API_KEY", raising=False)
    monkeypatch.delenv("SYNTRA_AUTH", raising=False)
    monkeypatch.delenv("SYNTRA_ALLOW_SIGNUP", raising=False)
    auth._FAILS.clear()
    from app.main import app
    return TestClient(app)


def _signup(c, email="a@b.co", pw="correct-horse", name="Ali"):
    return c.post("/api/auth/signup", json={"email": email, "password": pw, "name": name})


def test_first_run_is_open_on_this_machine_then_closes(client):
    s = client.get("/api/auth/status").json()
    assert s["accounts_exist"] is False and s["first_run_here"] is True
    assert client.get("/api/images").status_code == 200          # first run: the app loads
    r = _signup(client)
    assert r.status_code == 200 and r.json()["user"]["is_admin"] is True
    assert client.get("/api/images").status_code == 200          # signed in by the cookie
    client.post("/api/auth/signout")
    assert client.get("/api/images").status_code == 401          # account exists: sign-in required


def test_password_is_hashed_and_session_token_not_stored(client, tmp_path):
    import sqlite3

    r = _signup(client)
    token = r.cookies.get("syntra_session") or client.cookies.get("syntra_session")
    db = sqlite3.connect(tmp_path / "syntra.db")
    pw_hash = db.execute("SELECT pw_hash FROM users").fetchone()[0]
    assert b"correct-horse" not in pw_hash and len(pw_hash) == 32
    stored = [row[0] for row in db.execute("SELECT token_hash FROM sessions")]
    assert token and token not in stored and len(stored) == 1


def test_signup_validation(client):
    assert _signup(client, email="nope").json()["detail"]["field"] == "email"
    assert _signup(client, pw="short").json()["detail"]["field"] == "password"
    assert _signup(client).status_code == 200
    client.post("/api/auth/signout")
    r = _signup(client, email="A@B.co")                        # emails compare case-insensitively
    assert r.status_code == 409 and r.json()["detail"]["field"] == "email"


def test_second_account_is_not_admin_and_signup_can_be_closed(client, monkeypatch):
    _signup(client)
    client.post("/api/auth/signout")
    r = _signup(client, email="second@b.co")
    assert r.status_code == 200 and r.json()["user"]["is_admin"] is False
    client.post("/api/auth/signout")
    monkeypatch.setenv("SYNTRA_ALLOW_SIGNUP", "0")
    assert client.get("/api/auth/status").json()["signup_open"] is False
    assert _signup(client, email="third@b.co").status_code == 403


def test_signin_wrong_password_then_right(client):
    _signup(client)
    client.post("/api/auth/signout")
    bad = client.post("/api/auth/signin", json={"email": "a@b.co", "password": "wrong-pass"})
    assert bad.status_code == 401
    unknown = client.post("/api/auth/signin", json={"email": "x@b.co", "password": "wrong-pass"})
    assert unknown.json()["detail"] == bad.json()["detail"]     # no hint which emails exist
    ok = client.post("/api/auth/signin", json={"email": "A@b.co ", "password": "correct-horse"})
    assert ok.status_code == 200
    assert client.get("/api/auth/status").json()["user"]["email"] == "a@b.co"


def test_signin_is_throttled_after_five_failures(client):
    _signup(client)
    client.post("/api/auth/signout")
    for _ in range(5):
        client.post("/api/auth/signin", json={"email": "a@b.co", "password": "nope-nope"})
    r = client.post("/api/auth/signin", json={"email": "a@b.co", "password": "correct-horse"})
    assert r.status_code == 429


def test_first_account_only_from_this_machine(client, monkeypatch):
    import app.api.auth as api_auth

    monkeypatch.setattr(api_auth, "is_loopback", lambda req: False)
    assert _signup(client).status_code == 403


def test_api_key_still_works_without_a_session(client, monkeypatch):
    _signup(client)
    client.post("/api/auth/signout")
    monkeypatch.setenv("SYNTRA_API_KEY", "k" * 24)
    assert client.get("/api/images").status_code == 401
    assert client.get("/api/images", headers={"X-API-Key": "k" * 24}).status_code == 200


def test_auth_off_restores_open_access(client, monkeypatch):
    _signup(client)
    client.post("/api/auth/signout")
    monkeypatch.setenv("SYNTRA_AUTH", "off")
    assert client.get("/api/images").status_code == 200
