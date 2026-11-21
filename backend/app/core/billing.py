"""
Credits: every account has a balance; each detection Run costs credits by the
area it covers; credits are bought in packages. The owner (the first account,
an API-key caller, or anyone when sign-in is off) is never charged.

Everything lives in `SYNTRA_DATA_DIR/syntra.db` beside accounts and jobs:

  users.credits      the balance (column added to the accounts table)
  credit_ledger      every change: delta, balance after, reason, who did it
  payments           checkout sessions and their outcome

The price list (free credits for a new account, the cost of a Run by area,
the packages) is `SYNTRA_DATA_DIR/billing.json`, edited by the owner in
Settings > Admin; DEFAULT_PRICING applies until then.

A Run is charged when it starts and refunded if it fails, so a user never pays
for an error. The balance can't go below zero: a Run that costs more than the
balance is refused with 402 and the interface offers the packages.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path

from app.core.config import settings

# Price list. Credits per US dollar rise with the package (125, 130, 150), so a
# larger package is always the better buy; the $50 package is 6,500 rather than
# 6,000 for that reason.
DEFAULT_PRICING = {
    "free_credits": 5000,
    "run_tiers": [                       # first tier whose up_to_km2 covers the area
        {"up_to_km2": 100, "credits": 150},
        {"up_to_km2": 1000, "credits": 200},
    ],
    "over_last": {"per_km2": 1000, "credits": 50},   # past the last tier: +50 per started 1,000 km²
    "packages": [
        {"id": "p20", "usd": 20, "credits": 2500},
        {"id": "p50", "usd": 50, "credits": 6500},
        {"id": "p100", "usd": 100, "credits": 15000},
    ],
    "currency": "usd",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS credit_ledger (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    delta       INTEGER NOT NULL,
    balance     INTEGER NOT NULL,
    reason      TEXT NOT NULL,
    ref         TEXT,
    actor_id    INTEGER,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_user ON credit_ledger (user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS payments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    provider     TEXT NOT NULL,
    session_id   TEXT NOT NULL UNIQUE,
    package_id   TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    currency     TEXT NOT NULL,
    credits      INTEGER NOT NULL,
    status       TEXT NOT NULL,
    created_at   REAL NOT NULL,
    paid_at      REAL
);
"""


class InsufficientCredits(Exception):
    def __init__(self, needed: int, balance: int):
        super().__init__(f"This Run costs {needed:,} credits; the balance is {balance:,}")
        self.needed, self.balance = needed, balance


# --- price list ---------------------------------------------------------------
def _pricing_path() -> Path:
    return settings.DATA_DIR / "billing.json"


def pricing() -> dict:
    p = _pricing_path()
    try:
        saved = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULT_PRICING, **{k: v for k, v in saved.items() if k in DEFAULT_PRICING}}


def validate_pricing(p: dict) -> dict:
    """Checked copy of an owner's edit; ValueError says what is wrong."""
    out = {**DEFAULT_PRICING}
    fc = int(p.get("free_credits", out["free_credits"]))
    if not 0 <= fc <= 1_000_000:
        raise ValueError("Free credits: 0 to 1,000,000")
    out["free_credits"] = fc
    tiers = p.get("run_tiers", out["run_tiers"])
    clean, last = [], 0
    for t in tiers:
        up, c = float(t["up_to_km2"]), int(t["credits"])
        if up <= last or c < 0:
            raise ValueError("Run tiers: areas must rise and credits can't be negative")
        clean.append({"up_to_km2": up, "credits": c})
        last = up
    if not clean:
        raise ValueError("Keep at least one Run tier")
    out["run_tiers"] = clean
    ol = p.get("over_last", out["over_last"])
    if float(ol["per_km2"]) <= 0 or int(ol["credits"]) < 0:
        raise ValueError("Beyond the last tier: a positive area step and credits of 0 or more")
    out["over_last"] = {"per_km2": float(ol["per_km2"]), "credits": int(ol["credits"])}
    pk, ids = [], set()
    for x in p.get("packages", out["packages"]):
        pid, usd, cr = str(x["id"]), float(x["usd"]), int(x["credits"])
        if not pid or pid in ids or usd <= 0 or cr <= 0:
            raise ValueError("Packages: a unique id, a price above 0 and credits above 0")
        ids.add(pid)
        pk.append({"id": pid, "usd": usd, "credits": cr})
    out["packages"] = pk
    return out


def save_pricing(p: dict) -> dict:
    clean = validate_pricing(p)
    path = _pricing_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return clean


def run_cost(area_km2: float | None, price: dict | None = None) -> int:
    """Credits for one Run over `area_km2` (unknown area = the first tier)."""
    price = price or pricing()
    tiers = price["run_tiers"]
    a = float(area_km2 or 0.0)
    for t in tiers:
        if a <= t["up_to_km2"]:
            return int(t["credits"])
    over = price["over_last"]
    extra = math.ceil((a - tiers[-1]["up_to_km2"]) / over["per_km2"])
    return int(tiers[-1]["credits"] + extra * over["credits"])


# --- balances -----------------------------------------------------------------
class Ledger:
    def __init__(self, db_path: Path):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        from app.core.auth import get_auth_store
        get_auth_store()                         # make sure the users table (and its columns) exist
        # accounts made before credits existed: the welcome credits, once
        old = self._conn.execute(
            "SELECT id FROM users WHERE is_admin=0 AND id NOT IN (SELECT DISTINCT user_id FROM credit_ledger)").fetchall()
        free = int(pricing()["free_credits"])
        for (uid,) in old:
            if free:
                self.change(uid, free, "Welcome credits")

    def balance(self, user_id: int) -> int:
        with self._lock:
            row = self._conn.execute("SELECT credits FROM users WHERE id=?", (user_id,)).fetchone()
        return int(row[0]) if row else 0

    def change(self, user_id: int, delta: int, reason: str, *, ref: str | None = None,
               actor_id: int | None = None, allow_negative: bool = False) -> int:
        """Add `delta` (negative = spend) in one transaction; returns the new balance."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT credits FROM users WHERE id=?", (user_id,)).fetchone()
                if row is None:
                    raise KeyError(user_id)
                new = int(row[0]) + int(delta)
                if new < 0 and not allow_negative:
                    raise InsufficientCredits(-int(delta), int(row[0]))
                self._conn.execute("UPDATE users SET credits=? WHERE id=?", (new, user_id))
                self._conn.execute(
                    "INSERT INTO credit_ledger (user_id, delta, balance, reason, ref, actor_id, created_at) VALUES (?,?,?,?,?,?,?)",
                    (user_id, int(delta), new, reason[:200], ref, actor_id, time.time()))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return new

    def set_balance(self, user_id: int, value: int, reason: str, actor_id: int | None = None) -> int:
        return self.change(user_id, int(value) - self.balance(user_id), reason, actor_id=actor_id, allow_negative=True)

    def history(self, user_id: int | None = None, limit: int = 100) -> list[dict]:
        q = "SELECT id, user_id, delta, balance, reason, ref, actor_id, created_at FROM credit_ledger"
        args: tuple = ()
        if user_id is not None:
            q += " WHERE user_id=?"
            args = (user_id,)
        q += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(q, args + (int(limit),)).fetchall()
        keys = ("id", "user_id", "delta", "balance", "reason", "ref", "actor_id", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    def spent(self, user_id: int, since: float = 0.0) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(-SUM(delta),0) FROM credit_ledger WHERE user_id=? AND delta<0 AND created_at>=?",
                                     (user_id, since)).fetchone()
        return int(row[0] or 0)

    # payments ------------------------------------------------------------
    def new_payment(self, user_id: int, provider: str, session_id: str, package: dict, currency: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO payments (user_id, provider, session_id, package_id, amount_cents, currency, credits, status, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (user_id, provider, session_id, package["id"], int(round(package["usd"] * 100)), currency,
                 int(package["credits"]), "open", time.time()))

    def complete_payment(self, session_id: str) -> dict | None:
        """Mark a checkout paid and add its credits, once: a repeated webhook adds nothing."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT id, user_id, credits, status FROM payments WHERE session_id=?",
                                         (session_id,)).fetchone()
                if row is None or row[3] == "paid":
                    self._conn.execute("COMMIT")
                    return None
                self._conn.execute("UPDATE payments SET status='paid', paid_at=? WHERE id=?", (time.time(), row[0]))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        new = self.change(row[1], row[2], "Credits bought", ref=session_id)
        return {"user_id": row[1], "credits": row[2], "balance": new}

    def payments(self, user_id: int | None = None, limit: int = 100) -> list[dict]:
        q = "SELECT id, user_id, provider, session_id, package_id, amount_cents, currency, credits, status, created_at, paid_at FROM payments"
        args: tuple = ()
        if user_id is not None:
            q += " WHERE user_id=?"
            args = (user_id,)
        q += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(q, args + (int(limit),)).fetchall()
        keys = ("id", "user_id", "provider", "session_id", "package_id", "amount_cents", "currency", "credits", "status", "created_at", "paid_at")
        return [dict(zip(keys, r)) for r in rows]


_LEDGERS: dict[Path, Ledger] = {}
_LOCK = threading.Lock()


def get_ledger() -> Ledger:
    path = (settings.DATA_DIR / "syntra.db").resolve()
    with _LOCK:
        led = _LEDGERS.get(path)
        if led is None:
            led = _LEDGERS[path] = Ledger(path)
        return led
