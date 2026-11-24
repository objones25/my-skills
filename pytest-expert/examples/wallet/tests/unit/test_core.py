from decimal import Decimal

import pytest
from wallet.core import InsufficientFunds, Wallet, transfer
from wallet.money import AmountError


@pytest.fixture
def funded() -> Wallet:
    wallet = Wallet(owner="alice")
    wallet.deposit(Decimal("100.00"))
    return wallet


class _RefusesDeposits(Wallet):
    """A target whose deposit fails after the source was already debited."""

    def deposit(self, amount: Decimal) -> Decimal:
        raise RuntimeError("deposit refused")


def test_deposit_returns_the_new_balance(funded):
    assert funded.deposit(Decimal("0.50")) == Decimal("100.50")


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


def test_transfer_moves_exactly_the_amount(funded):
    target = Wallet(owner="bob", balance=Decimal("5.00"))
    transfer(funded, target, Decimal("40.00"))
    assert (funded.balance, target.balance) == (Decimal("60.00"), Decimal("45.00"))


def test_transfer_whose_deposit_fails_leaves_the_source_untouched(funded):
    target = _RefusesDeposits(owner="bob")
    with pytest.raises(RuntimeError, match="deposit refused"):
        transfer(funded, target, Decimal("40.00"))
    assert (funded.balance, target.balance) == (Decimal("100.00"), Decimal("0.00"))


def test_transfer_to_the_same_wallet_changes_nothing(funded):
    transfer(funded, funded, Decimal("40.00"))
    assert funded.balance == Decimal("100.00")


def test_withdraw_of_a_sub_unit_amount_works(funded):
    assert funded.withdraw(Decimal("0.50")) == Decimal("99.50")
