"""
The owner's console: accounts, credits, prices, payments and sign-in methods.
Every route here is owner-only (`require_owner`). Keys and client secrets are
written to secrets.json and only ever reported as set or not set.
"""
from __future__ import annotations

import json
import os
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.core import auth, billing
from app.core.config import settings
from app.core.jobs import get_store
from app.core.secrets import get_secret, set_secret

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(auth.require_owner)])

OAUTH_PROVIDERS = ("google", "github")


# --- accounts settings (sign-up open, beside env SYNTRA_ALLOW_SIGNUP) ----------
def _accounts_path():
    return settings.DATA_DIR / "accounts.json"


def accounts_settings() -> dict:
    try:
        return json.loads(_accounts_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_accounts(d: dict) -> None:
    p = _accounts_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
    os.replace(tmp, p)


# --- summary and users ------------------------------------------------------
def _runs_by_user() -> dict:
    out: dict = {}
    for j in get_store().list("predict", limit=100000):
        uid = j.get("owner_id")
        out.setdefault(uid, {"runs": 0, "last_run": None})
        out[uid]["runs"] += 1
        c = j.get("created_at")
        if c and (out[uid]["last_run"] is None or c > out[uid]["last_run"]):
            out[uid]["last_run"] = c
    return out


@router.get("/summary")
def summary():
    users = auth.get_auth_store().list()
    led = billing.get_ledger()
    pays = [p for p in led.payments(limit=100000) if p["status"] == "paid"]
    month = time.time() - 30 * 86400
    runs = _runs_by_user()
    return {
        "users": len(users),
        "active_7d": sum(1 for u in users if u.last_seen and u.last_seen > time.time() - 7 * 86400),
        "disabled": sum(1 for u in users if u.status != "active"),
        "credits_held": sum(u.credits for u in users if not u.is_admin),
        "runs": sum(r["runs"] for r in runs.values()),
        "revenue_cents_30d": sum(p["amount_cents"] for p in pays if (p["paid_at"] or 0) > month),
        "revenue_cents": sum(p["amount_cents"] for p in pays),
        "payments": stripe_status(),
    }


@router.get("/users")
def users():
    runs = _runs_by_user()
    led = billing.get_ledger()
    out = []
    for u in auth.get_auth_store().list():
        r = runs.get(u.id, {"runs": 0, "last_run": None})
        out.append({**u.admin_view(), "runs": r["runs"], "last_run": r["last_run"],
                    "spent_30d": led.spent(u.id, time.time() - 30 * 86400)})
    return out


class CreditChange(BaseModel):
    delta: int | None = None        # add (or take, when negative)
    set_to: int | None = None       # or set the balance
    note: str = ""


@router.post("/users/{user_id}/credits")
def change_credits(user_id: int, body: CreditChange, request: Request):
    store = auth.get_auth_store()
    u = store.get(user_id)
    if u is None:
        raise HTTPException(404, detail="Unknown account")
    if u.is_admin:
        raise HTTPException(400, detail="The owner has no credit limit")
    actor = auth.current_user(request)
    led = billing.get_ledger()
    note = body.note.strip()[:120]
    if body.set_to is not None:
        if body.set_to < 0:
            raise HTTPException(422, detail="A balance can't be negative")
        new = led.set_balance(user_id, body.set_to, f"Set by owner{': ' + note if note else ''}", actor_id=actor.id if actor else None)
    elif body.delta:
        try:
            new = led.change(user_id, body.delta, f"{'Added' if body.delta > 0 else 'Taken'} by owner{': ' + note if note else ''}",
                             actor_id=actor.id if actor else None)
        except billing.InsufficientCredits:
            raise HTTPException(422, detail="That would take the balance below zero")
    else:
        raise HTTPException(422, detail="Give an amount")
    return {"credits": new}


class UserPatch(BaseModel):
    status: str | None = None       # active | disabled
    name: str | None = None


@router.patch("/users/{user_id}")
def patch_user(user_id: int, body: UserPatch):
    store = auth.get_auth_store()
    u = store.get(user_id)
    if u is None:
        raise HTTPException(404, detail="Unknown account")
    fields = {}
    if body.status is not None:
        if body.status not in ("active", "disabled"):
            raise HTTPException(422, detail="Status is active or disabled")
        if u.is_admin and body.status != "active":
            raise HTTPException(400, detail="The owner's account can't be disabled")
        fields["status"] = body.status
    if body.name is not None:
        fields["name"] = body.name.strip()[:80] or u.name
    u = store.set_fields(user_id, **fields)
    if body.status == "disabled":
        store.end_sessions_of(user_id)
    return u.admin_view()


@router.delete("/users/{user_id}")
def delete_user(user_id: int):
    """Removes the account and its sign-ins. Its images, runs and results stay on disk
    and become the owner's; its ledger rows are kept for the record."""
    store = auth.get_auth_store()
    u = store.get(user_id)
    if u is None:
        raise HTTPException(404, detail="Unknown account")
    if u.is_admin:
        raise HTTPException(400, detail="The owner's account can't be deleted here")
    store.delete(user_id)
    return {"deleted": user_id}


@router.get("/ledger")
def ledger(user_id: int | None = None, limit: int = 200):
    return billing.get_ledger().history(user_id, limit=min(limit, 1000))


@router.get("/payments")
def payments(limit: int = 200):
    return billing.get_ledger().payments(limit=min(limit, 1000))


# --- prices -----------------------------------------------------------------
@router.get("/pricing")
def get_pricing():
    return {**billing.pricing(), "defaults": billing.DEFAULT_PRICING}


@router.put("/pricing")
def put_pricing(body: dict):
    try:
        return billing.save_pricing(body)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(422, detail=str(e) or "Check the values")


# --- payments provider ------------------------------------------------------
def stripe_status() -> dict:
    return {"provider": "stripe", "secret_key": bool(get_secret("stripe_secret_key")),
            "webhook_secret": bool(get_secret("stripe_webhook_secret"))}


class StripeKeys(BaseModel):
    secret_key: str | None = None
    webhook_secret: str | None = None


@router.get("/stripe")
def get_stripe():
    return stripe_status()


@router.put("/stripe")
def put_stripe(body: StripeKeys):
    if body.secret_key is not None:
        k = body.secret_key.strip()
        if k and not k.startswith(("sk_", "rk_")):
            raise HTTPException(422, detail="A Stripe secret key starts with sk_ (or rk_ for a restricted key)")
        set_secret("stripe_secret_key", k or None)
    if body.webhook_secret is not None:
        w = body.webhook_secret.strip()
        if w and not w.startswith("whsec_"):
            raise HTTPException(422, detail="A webhook signing secret starts with whsec_")
        set_secret("stripe_webhook_secret", w or None)
    return stripe_status()


# --- sign-in methods ---------------------------------------------------------
def signin_status(request: Request | None = None) -> dict:
    s = accounts_settings()
    out = {"signup_open": auth.signup_open(), "providers": {}}
    for p in OAUTH_PROVIDERS:
        out["providers"][p] = {"client_id": get_secret(f"oauth_{p}_client_id") or "",
                               "secret": bool(get_secret(f"oauth_{p}_client_secret")),
                               "enabled": bool(s.get(f"{p}_enabled", True))}
    if request is not None:
        from app.api.billing import _public_url
        out["callback_base"] = _public_url(request) + "/api/auth/oauth/"
    return out


class SigninSettings(BaseModel):
    signup_open: bool | None = None
    provider: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    enabled: bool | None = None


@router.get("/signin")
def get_signin(request: Request):
    return signin_status(request)


@router.put("/signin")
def put_signin(body: SigninSettings, request: Request):
    s = accounts_settings()
    if body.signup_open is not None:
        s["signup_open"] = bool(body.signup_open)
    if body.provider is not None:
        if body.provider not in OAUTH_PROVIDERS:
            raise HTTPException(422, detail="Provider is google or github")
        p = body.provider
        if body.client_id is not None:
            set_secret(f"oauth_{p}_client_id", body.client_id.strip() or None)
        if body.client_secret is not None:
            set_secret(f"oauth_{p}_client_secret", body.client_secret.strip() or None)
        if body.enabled is not None:
            s[f"{p}_enabled"] = bool(body.enabled)
    _save_accounts(s)
    return signin_status(request)
