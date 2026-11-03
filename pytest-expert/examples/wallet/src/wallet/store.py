"""Persistence: JSON files under a base directory, and a SQLite store."""

from __future__ import annotations

import json
import re
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path

from wallet.core import Wallet


class StoreError(ValueError):
    """Bad wallet name, path escape, or corrupt stored data."""


_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")


def wallet_path(base: Path, name: str) -> Path:
    """Map a user-supplied wallet name to a file that is guaranteed to live under base."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise StoreError(f"invalid wallet name: {name!r}")
    root = base.resolve()
    path = (root / f"{name}.json").resolve()
    if not path.is_relative_to(root):
        raise StoreError(f"wallet path escapes the store: {name!r}")
    return path


def save_wallet(base: Path, wallet: Wallet) -> Path:
    path = wallet_path(base, wallet.owner)
    path.write_text(json.dumps({"owner": wallet.owner, "balance": str(wallet.balance)}), encoding="utf-8")
    return path


def load_wallet(base: Path, name: str) -> Wallet:
    """Load with json, never pickle: stored bytes are data, not code."""
    raw = wallet_path(base, name).read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
        balance = Decimal(data["balance"])
        owner = data["owner"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, InvalidOperation) as exc:
        raise StoreError(f"corrupt wallet file for {name!r}") from exc
    if owner != name or not balance.is_finite() or balance < 0:
        raise StoreError(f"inconsistent wallet file for {name!r}")
    return Wallet(owner=owner, balance=balance)


class SqliteWalletStore:
    """Every query is parameterized; prefix search avoids LIKE wildcards entirely."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._db = sqlite3.connect(path)
        with self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS wallets (owner TEXT PRIMARY KEY, balance TEXT NOT NULL)")

    def put(self, wallet: Wallet) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO wallets (owner, balance) VALUES (?, ?) "
                "ON CONFLICT(owner) DO UPDATE SET balance = excluded.balance",
                (wallet.owner, str(wallet.balance)),
            )

    def get(self, owner: str) -> Wallet | None:
        row = self._db.execute("SELECT owner, balance FROM wallets WHERE owner = ?", (owner,)).fetchone()
        return Wallet(owner=row[0], balance=Decimal(row[1])) if row else None

    def find_by_prefix(self, prefix: str) -> list[Wallet]:
        rows = self._db.execute(
            "SELECT owner, balance FROM wallets WHERE substr(owner, 1, ?) = ? ORDER BY owner",
            (len(prefix), prefix),
        ).fetchall()
        return [Wallet(owner=o, balance=Decimal(b)) for o, b in rows]

    def count(self) -> int:
        return self._db.execute("SELECT count(*) FROM wallets").fetchone()[0]

    def close(self) -> None:
        self._db.close()
