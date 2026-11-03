from decimal import Decimal

import pytest
from wallet.core import InsufficientFunds, Wallet, transfer
from wallet.money import AmountError


@pytest.fixture
def funded() -> Wallet:
    wallet = Wallet(owner="alice")
    wallet.deposit(Decimal("100.00"))
    return wallet


def test_withdraw_reduces_balance(funded):
    assert funded.withdraw(Decimal("30.00")) == Decimal("70.00")


def test_withdraw_entire_balance_leaves_exactly_zero(funded):
    assert funded.withdraw(Decimal("100.00")) == Decimal("0.00")


def test_withdraw_over_balance_raises(funded):
    with pytest.raises(InsufficientFunds, match=r"cannot withdraw 100\.01"):
        funded.withdraw(Decimal("100.01"))


@pytest.mark.parametrize(
    "bad",
    [Decimal("NaN"), Decimal("Infinity"), Decimal("-1"), Decimal("0"), 5.0],
    ids=["nan", "infinity", "negative", "zero", "float-not-decimal"],
)
def test_withdraw_rejects_non_amounts(funded, bad):
    with pytest.raises(AmountError, match="invalid amount"):
        funded.withdraw(bad)


def test_failed_transfer_changes_neither_wallet(funded):
    target = Wallet(owner="bob")
    with pytest.raises(InsufficientFunds, match="cannot withdraw"):
        transfer(funded, target, Decimal("500.00"))
    assert (funded.balance, target.balance) == (Decimal("100.00"), Decimal("0.00"))
