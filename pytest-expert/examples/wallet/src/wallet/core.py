"""Wallet state. Accepts only Decimal amounts that came through parse_amount."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from wallet.money import AmountError


class InsufficientFunds(ValueError):
    """Withdrawal exceeds the available balance."""


def _require_amount(amount: object) -> Decimal:
    if not isinstance(amount, Decimal) or not amount.is_finite() or amount <= 0:
        raise AmountError(f"invalid amount: {amount!r}")
    return amount


@dataclass
class Wallet:
    owner: str
    balance: Decimal = field(default_factory=lambda: Decimal("0.00"))

    def deposit(self, amount: Decimal) -> Decimal:
        self.balance += _require_amount(amount)
        return self.balance

    def withdraw(self, amount: Decimal) -> Decimal:
        _require_amount(amount)
        if amount > self.balance:
            raise InsufficientFunds(f"cannot withdraw {amount} from balance {self.balance}")
        self.balance -= amount
        return self.balance


def transfer(source: Wallet, target: Wallet, amount: Decimal) -> None:
    """Move money between wallets; either both sides change or neither does."""
    source.withdraw(amount)
    target.deposit(amount)
