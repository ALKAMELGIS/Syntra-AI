"""
Credentials the engine keeps for the user (Hugging Face token, later RunPod).

Stored in `SYNTRA_DATA_DIR/secrets.json`, readable by this user only where the
file system allows it. The API never sends a secret back: it reports only
whether one is set and who it belongs to.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from app.core.config import settings

_LOCK = threading.Lock()


def _path() -> Path:
    return settings.DATA_DIR / "secrets.json"


def _load() -> dict:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}


def get_secret(name: str) -> str | None:
    return _load().get(name) or None


def set_secret(name: str, value: str | None) -> None:
    with _LOCK:
        d = _load()
        if value:
            d[name] = value
        else:
            d.pop(name, None)
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, p)
