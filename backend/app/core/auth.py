"""
Accounts and sign-in sessions.

Users and sessions live in the same SQLite file as jobs
(`SYNTRA_DATA_DIR/syntra.db`). Passwords are hashed with scrypt and a
random salt per user. A session is a random token; the browser holds it in
an HttpOnly cookie and the database only stores its SHA-256, so a copied
database file does not let anyone sign in.

Access rule for the API (`require_access`):
  - a valid SYNTRA_API_KEY in the request is always enough (scripts, QGIS);
  - otherwise a valid session cookie is required;
  - before the first account exists, requests from this machine are let in
    so the first person can open the app and create the admin account.
SYNTRA_AUTH=off turns accounts off entirely (single-user desktop setups that
want the old behaviour).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request

from app.core.config import settings
from app.core.security import configured_api_key, is_loopback

COOKIE = "syntra_session"
SESSION_DAYS = 30
MIN_PASSWORD = 8
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


def auth_enabled() -> bool:
    return os.getenv("SYNTRA_AUTH", "on").strip().lower() not in ("off", "0", "false", "no")


def signup_open() -> bool:
    """After the first (owner) account, more accounts only when allowed: the owner's
    switch in Settings > Admin (accounts.json), else SYNTRA_ALLOW_SIGNUP."""
    import json
    try:
        saved = json.loads((settings.DATA_DIR / "accounts.json").read_text(encoding="utf-8"))
        if "signup_open" in saved:
            return bool(saved["signup_open"])
    except (OSError, ValueError):
        pass
    return os.getenv("SYNTRA_ALLOW_SIGNUP", "1").strip().lower() not in ("0", "off", "false", "no")


def welcome(user: "User") -> None:
    """A new account (not the owner) starts with the free credits of the price list."""
    if user.is_admin:
        return
    from app.core import billing
    led = billing.get_ledger()                 # (on first use it already welcomes accounts without history)
    free = int(billing.pricing()["free_credits"])
    if free and not led.history(user.id, limit=1):
        led.change(user.id, free, "Welcome credits")


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class User:
    id: int
    email: str
    name: str
    is_admin: bool
    created_at: float
    credits: int = 0
    status: str = "active"          # active | disabled
    provider: str = "password"      # password | google | github
    last_seen: float | None = None

    def public(self) -> dict:
        return {"id": self.id, "email": self.email, "name": self.name, "is_admin": self.is_admin,
                "role": "owner" if self.is_admin else "user", "credits": self.credits, "provider": self.provider}

    def admin_view(self) -> dict:
        return {**self.public(), "status": self.status, "created_at": self.created_at, "last_seen": self.last_seen}


_USER_COLS = "id, email, name, is_admin, created_at, credits, status, provider, last_seen"
# columns added after the first release: (name, SQL type and default)
_ADDED = (("credits", "INTEGER NOT NULL DEFAULT 0"), ("status", "TEXT NOT NULL DEFAULT 'active'"),
          ("provider", "TEXT NOT NULL DEFAULT 'password'"), ("provider_id", "TEXT"), ("last_seen", "REAL"))


class AuthStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                pw_hash BLOB NOT NULL,
                salt BLOB NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at REAL NOT NULL
            );
        """)
        have = {r[1] for r in self._conn.execute("PRAGMA table_info(users)")}
        for col, decl in _ADDED:
            if col not in have:
                self._conn.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")
        self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS users_provider ON users (provider, provider_id) WHERE provider_id IS NOT NULL")

    # users ---------------------------------------------------------------
    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def _row(self, row) -> User | None:
        if not row:
            return None
        return User(id=row[0], email=row[1], name=row[2], is_admin=bool(row[3]), created_at=row[4],
                    credits=int(row[5] or 0), status=row[6] or "active", provider=row[7] or "password", last_seen=row[8])

    def get(self, user_id: int) -> User | None:
        with self._lock:
            return self._row(self._conn.execute(f"SELECT {_USER_COLS} FROM users WHERE id=?", (user_id,)).fetchone())

    def by_email(self, email: str) -> User | None:
        with self._lock:
            return self._row(self._conn.execute(f"SELECT {_USER_COLS} FROM users WHERE email=?",
                                                (email.strip().lower(),)).fetchone())

    def list(self) -> list[User]:
        with self._lock:
            return [self._row(r) for r in self._conn.execute(f"SELECT {_USER_COLS} FROM users ORDER BY id").fetchall()]

    def set_fields(self, user_id: int, **fields) -> User | None:
        allowed = {"name", "status", "is_admin", "last_seen"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if sets:
            with self._lock:
                self._conn.execute(f"UPDATE users SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?",
                                   (*sets.values(), user_id))
        return self.get(user_id)

    def delete(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            self._conn.execute("DELETE FROM users WHERE id=?", (user_id,))

    def end_sessions_of(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))

    def by_provider(self, provider: str, provider_id: str) -> User | None:
        with self._lock:
            return self._row(self._conn.execute(f"SELECT {_USER_COLS} FROM users WHERE provider=? AND provider_id=?",
                                                (provider, provider_id)).fetchone())

    def create_external(self, email: str, name: str, provider: str, provider_id: str) -> User:
        """An account that signs in with Google or GitHub: no password of its own (an
        unguessable one is stored, so password sign-in can never match)."""
        email = email.strip().lower()
        salt = secrets.token_bytes(16)
        with self._lock:
            first = self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
            try:
                cur = self._conn.execute(
                    "INSERT INTO users (email, name, pw_hash, salt, is_admin, created_at, provider, provider_id) VALUES (?,?,?,?,?,?,?,?)",
                    (email, name.strip()[:80] or email.split("@")[0], _hash(secrets.token_urlsafe(32), salt), salt,
                     int(first), time.time(), provider, provider_id))
            except sqlite3.IntegrityError:
                raise ValueError("email_taken")
            user_id = cur.lastrowid
        return self.get(user_id)

    def link_provider(self, user_id: int, provider: str, provider_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE users SET provider_id=COALESCE(provider_id, ?), provider=CASE WHEN provider_id IS NULL THEN ? ELSE provider END WHERE id=?",
                               (provider_id, provider, user_id))

    def create(self, email: str, name: str, password: str) -> User:
        email = email.strip().lower()
        salt = secrets.token_bytes(16)
        with self._lock:
            first = self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
            try:
                cur = self._conn.execute(
                    "INSERT INTO users (email, name, pw_hash, salt, is_admin, created_at) VALUES (?,?,?,?,?,?)",
                    (email, name.strip(), _hash(password, salt), salt, int(first), time.time()))
            except sqlite3.IntegrityError:
                raise ValueError("email_taken")
            user_id = cur.lastrowid
        return self.get(user_id)

    def verify(self, email: str, password: str) -> User | None:
        with self._lock:
            row = self._conn.execute("SELECT id, pw_hash, salt FROM users WHERE email=?",
                                     (email.strip().lower(),)).fetchone()
        if not row:
            _hash(password, b"0" * 16)        # same work either way: no timing hint that the email exists
            return None
        if not hmac.compare_digest(_hash(password, row[2]), row[1]):
            return None
        user = self.get(row[0])
        return user if user and user.status == "active" else None

    # sessions ------------------------------------------------------------
    def new_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
            self._conn.execute("INSERT INTO sessions VALUES (?,?,?)",
                               (_token_hash(token), user_id, time.time() + SESSION_DAYS * 86400))
        return token

    def user_for(self, token: str | None) -> User | None:
        if not token:
            return None
        with self._lock:
            row = self._conn.execute("SELECT user_id, expires_at FROM sessions WHERE token_hash=?",
                                     (_token_hash(token),)).fetchone()
        if not row or row[1] < time.time():
            return None
        user = self.get(row[0])
        if user is None or user.status != "active":
            return None                         # a disabled account's sessions stop working at once
        if not user.last_seen or time.time() - user.last_seen > 300:
            with self._lock:
                self._conn.execute("UPDATE users SET last_seen=? WHERE id=?", (time.time(), user.id))
        return user

    def end_session(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._conn.execute("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),))


_STORES: dict[Path, AuthStore] = {}
_STORES_LOCK = threading.Lock()


def get_auth_store() -> AuthStore:
    path = (settings.DATA_DIR / "syntra.db").resolve()
    with _STORES_LOCK:
        store = _STORES.get(path)
        if store is None:
            store = _STORES[path] = AuthStore(path)
        return store


# ---------------------------------------------------------------------------
# Sign-in throttling: 5 failed attempts per email or address in 10 minutes.
# ---------------------------------------------------------------------------
_FAILS: dict[str, list[float]] = {}
_FAILS_LOCK = threading.Lock()
_WINDOW, _MAX_FAILS = 600.0, 5


def throttled(*keys: str) -> int:
    """Seconds until another attempt is allowed (0 = allowed now)."""
    now = time.time()
    wait = 0
    with _FAILS_LOCK:
        for k in keys:
            hits = [t for t in _FAILS.get(k, []) if now - t < _WINDOW]
            _FAILS[k] = hits
            if len(hits) >= _MAX_FAILS:
                wait = max(wait, int(_WINDOW - (now - hits[0])) + 1)
    return wait


def record_failure(*keys: str) -> None:
    with _FAILS_LOCK:
        for k in keys:
            _FAILS.setdefault(k, []).append(time.time())


def clear_failures(*keys: str) -> None:
    with _FAILS_LOCK:
        for k in keys:
            _FAILS.pop(k, None)


def valid_email(email: str) -> bool:
    return bool(_EMAIL.match(email.strip())) and len(email) <= 254


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
def _key_ok(request: Request) -> bool | None:
    """True/False when a key is configured, None when there is none."""
    expected = configured_api_key()
    if expected is None:
        return None
    supplied = request.headers.get("x-api-key") or request.query_params.get("api_key") or ""
    return hmac.compare_digest(supplied.encode(), expected.encode())


def current_user(request: Request) -> User | None:
    return get_auth_store().user_for(request.cookies.get(COOKIE))


def is_owner(request: Request) -> bool:
    """The installation's owner: the first (admin) account, an API-key caller,
    or anyone when sign-in is turned off (one person on one machine)."""
    if _key_ok(request):
        return True
    if not auth_enabled():
        return True
    user = current_user(request)
    if user is None and _key_ok(request) is None and get_auth_store().count() == 0 and is_loopback(request):
        return True     # first run on this machine, before the owner's account exists
    return bool(user and user.is_admin)


def acting_user_id(request: Request) -> int | None:
    """The account a request acts for, or None for the owner acting as the installation
    (API key, sign-in off, first run). Data a user creates is stamped with this id."""
    if _key_ok(request) or not auth_enabled():
        return None
    user = current_user(request)
    return user.id if user else None


def can_see(request: Request, owner_id) -> bool:
    """The owner sees everything; a user sees only what they made. Data from before
    accounts (no owner id) belongs to the owner."""
    if is_owner(request):
        return True
    uid = acting_user_id(request)
    return owner_id is not None and uid is not None and int(owner_id) == uid


def can_see_any(request: Request, owners) -> bool:
    """For things several accounts can hold (an image uploaded by two people)."""
    if is_owner(request):
        return True
    uid = acting_user_id(request)
    return uid is not None and uid in set(int(o) for o in (owners or []))


def require_owner(request: Request) -> None:
    if not is_owner(request):
        raise HTTPException(status_code=403, detail="Only the owner can do this")


_REQUEST: ContextVar[Request | None] = ContextVar("syntra_request", default=None)


def current_request() -> Request | None:
    """The request being served (set by require_access), for helpers deep in a route."""
    return _REQUEST.get()


async def require_access(request: Request) -> None:
    # async on purpose: the context variable set here is copied into the thread a
    # sync route runs in, so helpers can ask whose request it is
    _REQUEST.set(request)
    key = _key_ok(request)
    if key:
        return
    if not auth_enabled():
        if key is False:
            raise HTTPException(status_code=401, detail="Missing or invalid API key.")
        return
    if current_user(request):
        return
    if get_auth_store().count() == 0 and key is None and is_loopback(request):
        return          # first run on this machine: let the app load so the admin account can be made
    raise HTTPException(status_code=401, detail="Sign in required.")
