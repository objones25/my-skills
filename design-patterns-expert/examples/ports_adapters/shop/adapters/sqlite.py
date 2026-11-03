"""Infrastructure adapter. The only module that knows about SQL."""
from __future__ import annotations

import sqlite3


class SqliteInventory:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_stock(self, sku: str) -> int | None:
        row = self._conn.execute("SELECT qty FROM stock WHERE sku = ?", (sku,)).fetchone()
        return None if row is None else int(row[0])

    def reserve(self, sku: str, quantity: int) -> int:
        with self._conn:   # one transaction: decrement and read back
            self._conn.execute("UPDATE stock SET qty = qty - ? WHERE sku = ?", (quantity, sku))
            return int(self._conn.execute("SELECT qty FROM stock WHERE sku = ?", (sku,)).fetchone()[0])
