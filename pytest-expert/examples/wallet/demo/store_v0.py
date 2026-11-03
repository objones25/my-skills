"""v0 store: builds SQL with an f-string. Injectable."""

import sqlite3


class StoreV0:
    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE wallets (owner TEXT PRIMARY KEY, balance TEXT)")
        self.db.executemany("INSERT INTO wallets VALUES (?, ?)", [("alice", "10.00"), ("bob", "99.00")])

    def get(self, owner: str):
        return self.db.execute(f"SELECT owner, balance FROM wallets WHERE owner = '{owner}'").fetchone()
