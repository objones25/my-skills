"""Each test here targets a planted bug in demo/*_v0.py. Most are EXPECTED TO FAIL.

Run from examples/wallet:  pytest demo -p no:randomly --no-cov
"""

import math

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from money_v0 import AmountError, parse_amount, transfer_fee_cents
from store_v0 import StoreV0
from wallet_v0 import InsufficientFunds, Wallet

amounts = st.integers(min_value=1, max_value=10**8)
accounts = st.integers(min_value=10**9, max_value=10**10 - 1)


@settings(database=None)
@given(st.floats())
def test_withdraw_never_leaves_a_non_finite_balance_random_only(amount):
    wallet = Wallet(balance=100.0)
    try:
        wallet.withdraw(amount)
    except (ValueError, InsufficientFunds):
        return
    assert math.isfinite(wallet.balance)


@settings(database=None)
@given(st.floats())
@example(math.nan)
@example(math.inf)
def test_withdraw_never_leaves_a_non_finite_balance(amount):
    wallet = Wallet(balance=100.0)
    try:
        wallet.withdraw(amount)
    except (ValueError, InsufficientFunds):
        return
    assert math.isfinite(wallet.balance)


@settings(database=None)
@given(st.text(max_size=40))
def test_parse_amount_raises_only_amount_error(text):
    try:
        value = parse_amount(text)
    except AmountError:
        return
    assert value > 0


@settings(max_examples=1000, database=None)
@given(amounts, accounts)
def test_fee_never_negative_random_search(amount_cents, account):
    try:
        fee = transfer_fee_cents(amount_cents, account)
    except AmountError:
        return
    assert fee >= 0


@settings(backend="crosshair", max_examples=200, database=None, deadline=None)
@given(amounts, accounts)
def test_fee_never_negative_solver_search(amount_cents, account):
    try:
        fee = transfer_fee_cents(amount_cents, account)
    except AmountError:
        return
    assert fee >= 0


@pytest.fixture
def store():
    """Close the connection: on Python 3.13+ an unclosed sqlite3.Connection emits
    ResourceWarning, which filterwarnings = "error" turns into a crash at exit."""
    s = StoreV0()
    yield s
    s.db.close()


@pytest.mark.parametrize("payload", ["' OR '1'='1", "nobody' OR owner LIKE '%"], ids=["tautology", "like-wildcard"])
def test_get_returns_nothing_for_injection_payloads(store, payload):
    assert store.get(payload) is None
