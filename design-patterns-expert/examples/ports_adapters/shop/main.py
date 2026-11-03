"""Composition root: the one place that names concrete adapters."""
from __future__ import annotations

import sqlite3

from shop.adapters.sqlite import SqliteInventory
from shop.domain import OrderRequest, OutOfStock, place_order


def build_inventory() -> SqliteInventory:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE stock (sku TEXT PRIMARY KEY, qty INTEGER NOT NULL)")
    conn.executemany("INSERT INTO stock VALUES (?, ?)", [("MUG", 5), ("TEE", 1)])
    return SqliteInventory(conn)


def main() -> None:
    inventory = build_inventory()
    print(place_order(OrderRequest("MUG", 2), inventory))
    try:
        place_order(OrderRequest("TEE", 3), inventory)
    except OutOfStock as exc:
        print("OutOfStock:", exc)


if __name__ == "__main__":
    main()
