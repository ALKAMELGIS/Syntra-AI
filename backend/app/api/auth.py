"""Sign in, create account, sign out. Open routes: they are how a session starts."""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core import auth
from app.core.security import is_loopback

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignUp(BaseModel):
    name: str = ""
    email: str
    password: str


class SignIn(BaseModel):
    email: str
    password: str


def _set_cookie(response: Response, request: Request, token: str, max_age: int | None = None) -> None:
    """The session cookie. Over https it is SameSite=None; Secure; Partitioned, so it also
    works when Syntra is shown inside another site's frame (the Hugging Face Space page);
    cross-site form posts are stopped by the Origin check in main.py instead."""
    https = request.url.scheme == "https"
    response.set_cookie(auth.COOKIE, token, max_age=auth.SESSION_DAYS * 86400 if max_age is None else max_age,
                        httponly=True, samesite="none" if https else "lax", secure=https, path="/")
    if https:                                    # Starlette has no 'partitioned' yet: add it to the header
        for i in range(len(response.raw_headers) - 1, -1, -1):
            k, v = response.raw_headers[i]
            if k == b"set-cookie" and v.startswith(auth.COOKIE.encode() + b"="):
                response.raw_headers[i] = (k, v + b"; Partitioned")
                break


@router.get("/status")
def status(request: Request):
    store = auth.get_auth_store()
    n = store.count()
    user = auth.current_user(request)
    return {
        "enabled": auth.auth_enabled(),
        "accounts_exist": n > 0,
        "signup_open": n == 0 or auth.signup_open(),
        "user": user.public() if user else None,
        # the very first account can only be made from this machine
        "first_run_here": n == 0 and (is_loopback(request) or bool(os.getenv("SYNTRA_OWNER_EMAIL"))),
        "oauth": __import__("app.api.oauth", fromlist=["providers"]).providers(),
    }


@router.post("/signup")
async def signup(body: SignUp, request: Request, response: Response):
    store = auth.get_auth_store()
    n = store.count()
    if n == 0 and not auth.may_create_first(request, body.email):
        raise HTTPException(403, detail={"field": None, "message": "Create the first account on the computer running Syntra"})
    if n > 0 and not auth.signup_open():
        raise HTTPException(403, detail={"field": None, "message": "New accounts are closed. Ask the admin"})
    email = body.email.strip()
    if not auth.valid_email(email):
        raise HTTPException(422, detail={"field": "email", "message": "Enter a valid email"})
    if len(body.password) < auth.MIN_PASSWORD:
        raise HTTPException(422, detail={"field": "password", "message": f"Use at least {auth.MIN_PASSWORD} characters"})
    if body.password.strip().lower() == email.lower():
        raise HTTPException(422, detail={"field": "password", "message": "Use a password that is not your email"})
    name = body.name.strip()[:80] or email.split("@")[0]
    try:
        user = await run_in_threadpool(store.create, email, name, body.password)
    except ValueError:
        raise HTTPException(409, detail={"field": "email", "message": "An account with this email exists. Sign in instead"})
    auth.welcome(user)
    user = store.get(user.id)
    _set_cookie(response, request, store.new_session(user.id))
    return {"user": user.public()}


@router.post("/signin")
async def signin(body: SignIn, request: Request, response: Response):
    store = auth.get_auth_store()
    email = body.email.strip().lower()
    ip = request.client.host if request.client else "?"
    keys = (f"e:{email}", f"ip:{ip}")
    wait = auth.throttled(*keys)
    if wait:
        raise HTTPException(429, detail={"field": None, "message": f"Too many attempts. Try again in {max(1, wait // 60)} min"})
    user = await run_in_threadpool(store.verify, email, body.password)
    if not user:
        auth.record_failure(*keys)
        raise HTTPException(401, detail={"field": "password", "message": "Email or password is wrong"})
    auth.clear_failures(*keys)
    _set_cookie(response, request, store.new_session(user.id))
    return {"user": user.public()}


@router.post("/signout")
def signout(request: Request, response: Response):
    auth.get_auth_store().end_session(request.cookies.get(auth.COOKIE))
    _set_cookie(response, request, "", max_age=0)          # same attributes, or the browser keeps it
    return {"ok": True}
