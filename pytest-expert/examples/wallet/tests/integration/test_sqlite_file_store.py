"""Integration: the real SQLite engine with a file on disk, across connections."""

from decimal import Decimal

from wallet.core import Wallet
from wallet.store import SqliteWalletStore


def test_wallet_survives_reopening_the_database(tmp_path):
    path = tmp_path / "wallets.db"
    first = SqliteWalletStore(path)
    first.put(Wallet(owner="alice", balance=Decimal("12.50")))
    first.close()
    second = SqliteWalletStore(path)
    assert second.get("alice") == Wallet(owner="alice", balance=Decimal("12.50"))
    second.close()


def test_put_updates_an_existing_wallet_instead_of_duplicating(tmp_path):
    store = SqliteWalletStore(tmp_path / "wallets.db")
    store.put(Wallet(owner="alice", balance=Decimal("1.00")))
    store.put(Wallet(owner="alice", balance=Decimal("2.00")))
    assert (store.count(), store.get("alice").balance) == (1, Decimal("2.00"))
    store.close()
