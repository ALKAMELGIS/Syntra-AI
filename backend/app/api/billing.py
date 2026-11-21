"""
Credits for the signed-in account: balance, prices, history, and buying a
package with Stripe Checkout.

Stripe is used through its REST API (no SDK). The owner saves a secret key
(and, for a public server, a webhook signing secret) in Settings > Admin; they
are kept in secrets.json and never returned. Credits are added when Stripe
says the checkout is paid, either by the webhook or, on a computer Stripe
can't reach (localhost), when the buyer comes back from the checkout page and
the engine asks Stripe for the session. Either way a checkout adds its credits
once.
"""
from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.core import auth, billing
from app.core.secrets import get_secret

router = APIRouter(prefix="/api/billing", tags=["billing"])
webhook_router = APIRouter(prefix="/api/billing", tags=["billing"])   # open: Stripe calls it
STRIPE = "https://api.stripe.com/v1"


def _public_url(request: Request) -> str:
    import os
    return (os.getenv("SYNTRA_PUBLIC_URL") or str(request.base_url)).rstrip("/")


def stripe_ready() -> bool:
    return bool(get_secret("stripe_secret_key"))


@router.get("/me")
def me(request: Request):
    user = auth.current_user(request)
    owner = auth.is_owner(request)
    price = billing.pricing()
    out = {"owner": owner, "unlimited": owner, "payments": stripe_ready(),
           "packages": price["packages"], "run_tiers": price["run_tiers"], "over_last": price["over_last"],
           "currency": price["currency"], "credits": None, "history": []}
    if user is not None and not owner:
        led = billing.get_ledger()
        out["credits"] = led.balance(user.id)
        out["history"] = led.history(user.id, limit=50)
    return out


class Checkout(BaseModel):
    package_id: str


def _stripe(method: str, path: str, data: dict | None = None) -> dict:
    import httpx

    key = get_secret("stripe_secret_key")
    if not key:
        raise HTTPException(status_code=409, detail="Payments are not set up yet. Ask the owner")
    r = httpx.request(method, f"{STRIPE}{path}", data=data, auth=(key, ""), timeout=30.0)
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", {}).get("message")
        except ValueError:
            msg = r.text[:200]
        raise HTTPException(status_code=502, detail=f"Stripe: {msg}")
    return r.json()


@router.post("/checkout")
def checkout(body: Checkout, request: Request):
    user = auth.current_user(request)
    if user is None or auth.is_owner(request):
        raise HTTPException(status_code=400, detail="The owner's account has no credits to buy")
    price = billing.pricing()
    pkg = next((p for p in price["packages"] if p["id"] == body.package_id), None)
    if pkg is None:
        raise HTTPException(status_code=404, detail="Unknown package")
    base = _public_url(request)
    sess = _stripe("POST", "/checkout/sessions", {
        "mode": "payment",
        "line_items[0][quantity]": 1,
        "line_items[0][price_data][currency]": price["currency"],
        "line_items[0][price_data][unit_amount]": int(round(pkg["usd"] * 100)),
        "line_items[0][price_data][product_data][name]": f"Syntra credits · {pkg['credits']:,}",
        "client_reference_id": str(user.id),
        "customer_email": user.email,
        "metadata[package_id]": pkg["id"],
        "metadata[user_id]": str(user.id),
        "success_url": f"{base}/api/billing/return?session_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{base}/?credits=cancelled",
    })
    billing.get_ledger().new_payment(user.id, "stripe", sess["id"], pkg, price["currency"])
    return {"url": sess["url"]}


@router.get("/return")
def checkout_return(session_id: str, request: Request):
    """Stripe sends the buyer back here: confirm with Stripe, add the credits, open the app."""
    user = auth.current_user(request)
    rows = [p for p in billing.get_ledger().payments(user.id if user else None, limit=200) if p["session_id"] == session_id]
    if not rows:
        return RedirectResponse("/?credits=unknown", status_code=303)
    sess = _stripe("GET", f"/checkout/sessions/{session_id}")
    if sess.get("payment_status") == "paid":
        billing.get_ledger().complete_payment(session_id)
        return RedirectResponse("/?credits=added", status_code=303)
    return RedirectResponse("/?credits=pending", status_code=303)


def verify_stripe_signature(payload: bytes, header: str, secret: str, tolerance: int = 300) -> bool:
    """Stripe-Signature: t=<unix>,v1=<hex hmac-sha256 of "<t>.<payload>">."""
    try:
        parts = dict(kv.split("=", 1) for kv in header.split(","))
        t = int(parts["t"])
    except (ValueError, KeyError):
        return False
    if abs(time.time() - t) > tolerance:
        return False
    expected = hmac.new(secret.encode(), f"{t}.".encode() + payload, hashlib.sha256).hexdigest()
    sigs = [v for k, v in (kv.split("=", 1) for kv in header.split(",")) if k == "v1"]
    return any(hmac.compare_digest(expected, s) for s in sigs)


@webhook_router.post("/webhook")
async def webhook(request: Request):
    import json

    secret = get_secret("stripe_webhook_secret")
    payload = await request.body()
    if not secret or not verify_stripe_signature(payload, request.headers.get("stripe-signature", ""), secret):
        raise HTTPException(status_code=400, detail="Bad signature")
    event = json.loads(payload)
    if event.get("type") in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
        obj = event.get("data", {}).get("object", {})
        if obj.get("payment_status") == "paid":
            billing.get_ledger().complete_payment(obj.get("id", ""))
    return {"received": True}
