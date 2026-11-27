"""Credits, prices, owner console, data ownership, Stripe and OAuth (providers mocked)."""
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from app.core import auth
    from app.core.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "images")
    monkeypatch.setattr(settings, "RESULTS_DIR", tmp_path / "results")
    for k in ("SYNTRA_API_KEY", "SYNTRA_AUTH", "SYNTRA_ALLOW_SIGNUP"):
        monkeypatch.delenv(k, raising=False)
    auth._FAILS.clear()
    from app.main import app
    owner, user = TestClient(app), TestClient(app)
    owner.post("/api/auth/signup", json={"email": "owner@x.co", "password": "owner-pass-1", "name": "Owner"})
    r = user.post("/api/auth/signup", json={"email": "user@x.co", "password": "user-pass-12", "name": "User"})
    return owner, user, r.json()["user"]["id"]


def test_run_cost_by_area():
    from app.core.billing import DEFAULT_PRICING, run_cost
    assert [run_cost(a, DEFAULT_PRICING) for a in (None, 5, 100, 100.1, 1000, 1500, 2600)] == [150, 150, 150, 200, 200, 250, 300]


def test_price_list_is_validated_and_ladder_rises():
    from app.core.billing import DEFAULT_PRICING, validate_pricing
    per_usd = [p["credits"] / p["usd"] for p in DEFAULT_PRICING["packages"]]
    assert per_usd == sorted(per_usd)                              # a bigger package is never a worse buy
    with pytest.raises(ValueError):
        validate_pricing({"run_tiers": [{"up_to_km2": 100, "credits": 150}, {"up_to_km2": 50, "credits": 200}]})
    with pytest.raises(ValueError):
        validate_pricing({"packages": [{"id": "a", "usd": 0, "credits": 10}]})


def test_welcome_credits_and_owner_unlimited(env):
    owner, user, _ = env
    assert user.get("/api/billing/me").json()["credits"] == 5000
    me = owner.get("/api/billing/me").json()
    assert me["unlimited"] is True and me["credits"] is None


def test_charge_refuses_when_short_and_refunds(env):
    from fastapi import HTTPException

    from app.api import predict
    from app.core import billing
    owner, user, uid = env
    led = billing.get_ledger()

    class Req:                                                     # a request from the user's session
        headers, query_params = {}, {}
        cookies = dict(user.cookies)
        client = type("C", (), {"host": "127.0.0.1"})()
    assert predict._charge("job-1", Req(), 50) == 150 and led.balance(uid) == 4850
    assert predict._charge("job-2", Req(), 1500) == 250 and led.balance(uid) == 4600
    led.set_balance(uid, 100, "test")
    with pytest.raises(HTTPException) as e:
        predict._charge("job-3", Req(), 10)
    assert e.value.status_code == 402 and e.value.detail["needed"] == 150
    from app.core.jobs import get_store
    get_store().create("predict", job_id="job-9", owner_id=uid, credits_charged=150)
    predict._refund("job-9"); predict._refund("job-9")             # once only
    assert led.balance(uid) == 250


def test_owner_console(env):
    owner, user, uid = env
    assert user.get("/api/admin/users").status_code == 403
    rows = owner.get("/api/admin/users").json()
    assert [r["role"] for r in rows] == ["owner", "user"] and rows[1]["credits"] == 5000
    assert owner.post(f"/api/admin/users/{uid}/credits", json={"delta": 1000, "note": "promo"}).json()["credits"] == 6000
    assert owner.post(f"/api/admin/users/{uid}/credits", json={"set_to": 200}).json()["credits"] == 200
    assert owner.post(f"/api/admin/users/{uid}/credits", json={"delta": -500}).status_code == 422
    hist = owner.get(f"/api/admin/ledger?user_id={uid}").json()
    assert hist[0]["reason"].startswith("Set by owner") and hist[1]["reason"] == "Added by owner: promo"
    assert owner.patch(f"/api/admin/users/{uid}", json={"status": "disabled"}).json()["status"] == "disabled"
    assert user.get("/api/billing/me").status_code == 401        # its sessions stop at once
    assert owner.patch("/api/admin/users/1", json={"status": "disabled"}).status_code == 400
    p = owner.get("/api/admin/pricing").json()
    p["free_credits"] = 2000
    assert owner.put("/api/admin/pricing", json=p).json()["free_credits"] == 2000
    assert owner.put("/api/admin/stripe", json={"secret_key": "pk_live_nope"}).status_code == 422
    s = owner.put("/api/admin/stripe", json={"secret_key": "sk_test_123"}).json()
    assert s["secret_key"] is True and "sk_test_123" not in json.dumps(owner.get("/api/admin/stripe").json())
    assert owner.put("/api/admin/signin", json={"signup_open": False}).json()["signup_open"] is False


def test_users_see_only_their_own_jobs_and_images(env):
    from app.core.jobs import get_store
    owner, user, uid = env
    get_store().create("predict", job_id="mine", owner_id=uid, image_id="i1")
    get_store().create("predict", job_id="owners", owner_id=None, image_id="i2")
    assert [j["job_id"] for j in user.get("/api/jobs?kind=predict").json()] == ["mine"]
    assert {j["job_id"] for j in owner.get("/api/jobs?kind=predict").json()} == {"mine", "owners"}
    assert user.get("/api/jobs/owners").status_code == 404
    assert user.get("/api/images").json() == []


def test_stripe_webhook_signature_and_once_only(env, monkeypatch):
    from app.api.billing import verify_stripe_signature
    from app.core import billing
    from app.core.secrets import set_secret
    owner, user, uid = env
    secret = "whsec_test"
    set_secret("stripe_webhook_secret", secret)
    led = billing.get_ledger()
    led.new_payment(uid, "stripe", "cs_1", {"id": "p20", "usd": 20, "credits": 2500}, "usd")
    body = json.dumps({"type": "checkout.session.completed", "data": {"object": {"id": "cs_1", "payment_status": "paid"}}}).encode()
    t = int(time.time())
    sig = hmac.new(secret.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    assert verify_stripe_signature(body, f"t={t},v1={sig}", secret)
    assert not verify_stripe_signature(body, f"t={t},v1={'0' * 64}", secret)
    assert owner.post("/api/billing/webhook", content=body, headers={"stripe-signature": f"t={t},v1=bad"}).status_code == 400
    for _ in range(2):
        assert owner.post("/api/billing/webhook", content=body, headers={"stripe-signature": f"t={t},v1={sig}"}).status_code == 200
    assert led.balance(uid) == 7500                               # 5000 + 2500, added once
    assert led.payments(uid)[0]["status"] == "paid"


def test_oauth_links_verified_email_and_creates_new(env, monkeypatch):
    from app.api import oauth
    from app.core.secrets import set_secret
    owner, user, uid = env
    set_secret("oauth_github_client_id", "cid")
    set_secret("oauth_github_client_secret", "sec")
    from app.main import app
    c = TestClient(app)
    assert c.get("/api/auth/oauth/providers").json() == {"google": False, "github": True}
    r = c.get("/api/auth/oauth/github/start", follow_redirects=False)
    assert r.status_code == 302 and "github.com/login/oauth/authorize" in r.headers["location"]
    state = c.cookies.get("syntra_oauth_state").split(":", 1)[1]
    monkeypatch.setattr(oauth, "fetch_identity", lambda *a: {"id": "42", "email": "user@x.co", "name": "U"})
    r = c.get(f"/api/auth/oauth/github/callback?code=x&state={state}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert c.get("/api/auth/status").json()["user"]["id"] == uid           # linked to the same account
    bad = TestClient(app).get("/api/auth/oauth/github/callback?code=x&state=forged", follow_redirects=False)
    assert "signin_error" in bad.headers["location"]
    c2 = TestClient(app)
    c2.get("/api/auth/oauth/github/start", follow_redirects=False)
    st2 = c2.cookies.get("syntra_oauth_state").split(":", 1)[1]
    monkeypatch.setattr(oauth, "fetch_identity", lambda *a: {"id": "7", "email": "new@x.co", "name": "New"})
    c2.get(f"/api/auth/oauth/github/callback?code=x&state={st2}", follow_redirects=False)
    me = c2.get("/api/billing/me").json()
    assert me["credits"] == 5000 and c2.get("/api/auth/status").json()["user"]["provider"] == "github"
