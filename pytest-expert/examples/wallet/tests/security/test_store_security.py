"""Security regressions for the persistence boundary."""

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


def test_load_wallet_round_trips(tmp_path):
    save_wallet(tmp_path, Wallet(owner="alice", balance=Decimal("12.50")))
    assert load_wallet(tmp_path, "alice") == Wallet(owner="alice", balance=Decimal("12.50"))


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
