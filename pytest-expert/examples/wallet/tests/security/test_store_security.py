"""Security regressions for the persistence boundary."""

import json
import pickle
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from wallet.core import Wallet
from wallet.store import SqliteWalletStore, StoreError, load_wallet, save_wallet, wallet_path

pytestmark = pytest.mark.security

TRAVERSAL = [
    "../etc/passwd", "..", "../../secret", "/etc/passwd", "a/../../b", "a\\..\\b",
    "alice\n", "alice\x00", "\uff0e\uff0e", "%2e%2e%2f", "Alice", "",
]
INJECTION = ["' OR '1'='1", "alice'; DROP TABLE wallets; --", "%", "_", "a%", "\\"]


@pytest.fixture
def store():
    db = SqliteWalletStore()
    for owner in ("alice", "alfred", "bob"):
        db.put(Wallet(owner=owner, balance=Decimal("10.00")))
    yield db
    db.close()


@pytest.mark.parametrize("name", TRAVERSAL, ids=repr)
def test_wallet_path_rejects_hostile_names(tmp_path, name):
    with pytest.raises(StoreError, match="invalid wallet name"):
        wallet_path(tmp_path, name)


@given(st.text(max_size=60))
def test_wallet_path_never_escapes_the_base_directory(tmp_path_factory, name):
    base = tmp_path_factory.getbasetemp() / "store"
    try:
        path = wallet_path(base, name)
    except StoreError:
        return
    assert path.is_relative_to(base.resolve())


def test_load_wallet_refuses_pickle_payloads(tmp_path):
    (tmp_path / "mallory.json").write_bytes(pickle.dumps({"owner": "mallory"}))
    with pytest.raises(StoreError, match="corrupt wallet file"):
        load_wallet(tmp_path, "mallory")


def test_save_wallet_returns_the_file_it_wrote(tmp_path):
    path = save_wallet(tmp_path, Wallet(owner="alice", balance=Decimal("1.00")))
    assert path == tmp_path.resolve() / "alice.json"
    assert path.read_text(encoding="utf-8") == '{"owner": "alice", "balance": "1.00"}'


def test_wallet_path_rejects_a_symlink_that_leaves_the_base_directory(tmp_path):
    base = tmp_path / "store"
    base.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (base / "alice.json").symlink_to(outside)
    with pytest.raises(StoreError, match="^wallet path escapes the store: 'alice'$"):
        wallet_path(base, "alice")


@pytest.mark.parametrize(
    "stored",
    [
        {"owner": "mallory", "balance": "5.00"},
        {"owner": "alice", "balance": "-0.01"},
        {"owner": "alice", "balance": "NaN"},
        {"owner": "alice", "balance": "Infinity"},
    ],
    ids=["owner-mismatch", "negative-balance", "nan-balance", "infinite-balance"],
)
def test_load_wallet_rejects_inconsistent_files(tmp_path, stored):
    (tmp_path / "alice.json").write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(StoreError, match="^inconsistent wallet file for 'alice'$"):
        load_wallet(tmp_path, "alice")


def test_load_wallet_accepts_a_zero_balance(tmp_path):
    (tmp_path / "alice.json").write_text('{"owner": "alice", "balance": "0.00"}', encoding="utf-8")
    assert load_wallet(tmp_path, "alice") == Wallet(owner="alice", balance=Decimal("0.00"))


@pytest.mark.parametrize(
    "stored",
    [b"\xff\xfe", b"{not json", b'{"owner": "alice"}', b"[]", b'{"owner": "alice", "balance": "abc"}'],
    ids=["not-utf8", "not-json", "missing-balance", "wrong-shape", "balance-not-a-number"],
)
def test_load_wallet_reports_corrupt_files(tmp_path, stored):
    (tmp_path / "alice.json").write_bytes(stored)
    with pytest.raises(StoreError, match="^corrupt wallet file for 'alice'$"):
        load_wallet(tmp_path, "alice")


def test_load_wallet_round_trips(tmp_path):
    save_wallet(tmp_path, Wallet(owner="alice", balance=Decimal("12.50")))
    assert load_wallet(tmp_path, "alice") == Wallet(owner="alice", balance=Decimal("12.50"))


def test_get_returns_the_stored_wallet(store):
    assert store.get("alice") == Wallet(owner="alice", balance=Decimal("10.00"))


def test_find_by_prefix_returns_matching_wallets_in_owner_order(store):
    assert [w.owner for w in store.find_by_prefix("al")] == ["alfred", "alice"]


@pytest.mark.parametrize("payload", INJECTION, ids=repr)
def test_get_treats_injection_payloads_as_plain_data(store, payload):
    assert store.get(payload) is None
    assert store.count() == 3


@pytest.mark.parametrize("payload", INJECTION, ids=repr)
def test_prefix_search_treats_wildcards_literally(store, payload):
    assert store.find_by_prefix(payload) == []


@given(st.text(max_size=12))
def test_prefix_search_only_returns_matching_owners(prefix):
    db = SqliteWalletStore()
    db.put(Wallet(owner="alice"))
    db.put(Wallet(owner="bob"))
    owners = [w.owner for w in db.find_by_prefix(prefix)]
    db.close()
    assert owners == sorted(o for o in ("alice", "bob") if o.startswith(prefix))
