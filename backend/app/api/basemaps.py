"""
Google satellite as a base map, through the official Map Tiles API.

Display only. Google's Map Tiles API policies forbid image analysis, object
detection, storing or caching tiles and extracting data from them, so these
tiles are never sent to a model, never written to disk and never offered as
training data: the engine passes each tile straight through to the map.

The key stays on this server (secrets.json, never returned). The browser asks
/api/basemaps/google/tile/{z}/{x}/{y}; the engine adds the session and key.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core.secrets import get_secret, set_secret

router = APIRouter(prefix="/api/basemaps", tags=["basemaps"])

_KEY = "google_maps_key"
API = "https://tile.googleapis.com/v1"
_session: dict = {}
_lock = threading.Lock()
_open = urllib.request.urlopen          # replaced in tests


class KeyBody(BaseModel):
    key: str


def _create_session(key: str) -> dict:
    body = json.dumps({"mapType": "satellite", "language": "en-US", "region": "US"}).encode()
    req = urllib.request.Request(f"{API}/createSession?key={key}", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with _open(req, timeout=15) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", "replace")[:300]
        raise HTTPException(status_code=400, detail=f"Google refused the key ({exc.code}). Enable Map Tiles API for it. {msg}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"Can't reach Google: {exc}") from None
    return {"token": d["session"], "expires": float(d.get("expiry", time.time() + 86400)), "key": key}


def _current() -> dict:
    key = get_secret(_KEY)
    if not key:
        raise HTTPException(status_code=409, detail="Add a Google Maps key first.")
    with _lock:
        s = _session.get("s")
        if not s or s["key"] != key or s["expires"] - time.time() < 3600:
            s = _create_session(key)
            _session["s"] = s
        return s


@router.get("/google")
def google_status():
    return {"configured": bool(get_secret(_KEY))}


@router.put("/google")
async def google_save(body: KeyBody):
    key = body.key.strip()
    if not key:
        set_secret(_KEY, None)
        _session.clear()
        return google_status()
    s = await run_in_threadpool(_create_session, key)      # a key that can't open a session is not kept
    set_secret(_KEY, key)
    with _lock:
        _session["s"] = s
    return google_status()


@router.get("/google/tile/{z}/{x}/{y}")
def google_tile(z: int, x: int, y: int):
    if not (0 <= z <= 22 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise HTTPException(status_code=400, detail="Tile outside the world.")
    s = _current()
    req = urllib.request.Request(f"{API}/2dtiles/{z}/{x}/{y}?session={s['token']}&key={s['key']}")
    try:
        with _open(req, timeout=15) as r:
            data, ctype = r.read(), r.headers.get("Content-Type", "image/jpeg")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            _session.clear()
        raise HTTPException(status_code=404 if exc.code == 404 else 502, detail="No Google tile here.") from None
    except (urllib.error.URLError, TimeoutError):
        raise HTTPException(status_code=502, detail="Can't reach Google.") from None
    # passed through, not stored: no-store keeps the browser from building its own cache of it
    return Response(content=data, media_type=ctype, headers={"Cache-Control": "no-store"})
