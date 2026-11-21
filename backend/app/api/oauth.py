"""
Sign in with Google or GitHub (OAuth 2.0 authorization code flow).

The owner registers an OAuth app with the provider, using the callback URL
shown in Settings > Admin > Sign-in (`<public url>/api/auth/oauth/<provider>/callback`),
and saves its client id and secret there. Then:

  /start     sets a one-time `state` in an HttpOnly cookie and sends the browser
             to the provider
  /callback  checks `state`, swaps the code for a token, reads the verified
             email, then signs into the account with that provider id, or links an
             existing account with the same verified email, or creates one (when
             sign-up is open) with the welcome credits

Open routes: they are how a session starts.
"""
from __future__ import annotations

import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.core import auth
from app.core.secrets import get_secret

router = APIRouter(prefix="/api/auth/oauth", tags=["auth"])
STATE_COOKIE = "syntra_oauth_state"

PROVIDERS = {
    "google": {
        "authorize": "https://accounts.google.com/o/oauth2/v2/auth",
        "token": "https://oauth2.googleapis.com/token",
        "scope": "openid email profile",
    },
    "github": {
        "authorize": "https://github.com/login/oauth/authorize",
        "token": "https://github.com/login/oauth/access_token",
        "scope": "read:user user:email",
    },
}


def _creds(p: str) -> tuple[str, str] | None:
    from app.api.admin import accounts_settings

    cid, sec = get_secret(f"oauth_{p}_client_id"), get_secret(f"oauth_{p}_client_secret")
    if not (cid and sec) or not accounts_settings().get(f"{p}_enabled", True):
        return None
    return cid, sec


def _callback(request: Request, p: str) -> str:
    from app.api.billing import _public_url
    return f"{_public_url(request)}/api/auth/oauth/{p}/callback"


@router.get("/providers")
def providers():
    return {p: _creds(p) is not None for p in PROVIDERS}


@router.get("/{provider}/start")
def start(provider: str, request: Request):
    if provider not in PROVIDERS or _creds(provider) is None:
        raise HTTPException(404, detail="This sign-in method is not set up")
    cid, _ = _creds(provider)
    state = secrets.token_urlsafe(24)
    q = {"client_id": cid, "redirect_uri": _callback(request, provider), "response_type": "code",
         "scope": PROVIDERS[provider]["scope"], "state": state}
    if provider == "google":
        q["prompt"] = "select_account"
    resp = RedirectResponse(f"{PROVIDERS[provider]['authorize']}?{urlencode(q)}", status_code=302)
    resp.set_cookie(STATE_COOKIE, f"{provider}:{state}", max_age=600, httponly=True, samesite="lax",
                    secure=request.url.scheme == "https", path="/")
    return resp


def fetch_identity(provider: str, code: str, redirect_uri: str, cid: str, sec: str) -> dict:
    """{id, email, name} with a verified email, from the provider. Separate for tests."""
    import httpx

    with httpx.Client(timeout=20.0, headers={"Accept": "application/json"}) as c:
        tok = c.post(PROVIDERS[provider]["token"], data={
            "client_id": cid, "client_secret": sec, "code": code, "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"}).json()
        at = tok.get("access_token")
        if not at:
            raise ValueError(tok.get("error_description") or tok.get("error") or "no token")
        h = {"Authorization": f"Bearer {at}"}
        if provider == "google":
            u = c.get("https://openidconnect.googleapis.com/v1/userinfo", headers=h).json()
            if not u.get("email_verified"):
                raise ValueError("Google says this email is not verified")
            return {"id": str(u["sub"]), "email": u["email"], "name": u.get("name") or ""}
        u = c.get("https://api.github.com/user", headers=h).json()
        emails = c.get("https://api.github.com/user/emails", headers=h).json()
        primary = next((e for e in emails if e.get("primary") and e.get("verified")), None) or \
            next((e for e in emails if e.get("verified")), None)
        if not primary:
            raise ValueError("GitHub has no verified email on this account")
        return {"id": str(u["id"]), "email": primary["email"], "name": u.get("name") or u.get("login") or ""}


def _fail(msg: str) -> RedirectResponse:
    return RedirectResponse(f"/?signin_error={urlencode({'m': msg})[2:]}", status_code=303)


@router.get("/{provider}/callback")
def callback(provider: str, request: Request, code: str = "", state: str = ""):
    from app.api.auth import _set_cookie
    from app.core.security import is_loopback

    creds = _creds(provider) if provider in PROVIDERS else None
    saved = request.cookies.get(STATE_COOKIE, "")
    if creds is None or not code or not state or not secrets.compare_digest(saved, f"{provider}:{state}"):
        return _fail("Sign-in expired. Try again")
    try:
        ident = fetch_identity(provider, code, _callback(request, provider), *creds)
    except Exception as e:  # noqa: BLE001 - the provider said no: tell the person, not a 500
        return _fail(f"{provider.title()} sign-in failed: {e}")
    store = auth.get_auth_store()
    user = store.by_provider(provider, ident["id"])
    if user is None:
        user = store.by_email(ident["email"])
        if user is not None:
            store.link_provider(user.id, provider, ident["id"])    # same verified email: the same person
        else:
            n = store.count()
            if n == 0 and not is_loopback(request):
                return _fail("Create the first account on the computer running Syntra")
            if n > 0 and not auth.signup_open():
                return _fail("New accounts are closed. Ask the owner")
            user = store.create_external(ident["email"], ident["name"], provider, ident["id"])
            auth.welcome(user)
    if user.status != "active":
        return _fail("This account is disabled. Ask the owner")
    resp = RedirectResponse("/", status_code=303)
    _set_cookie(resp, request, store.new_session(user.id))
    resp.delete_cookie(STATE_COOKIE, path="/")
    return resp
