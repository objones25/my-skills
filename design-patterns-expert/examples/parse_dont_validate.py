"""Parse, don't validate + Value Objects.

Raw input is parsed ONCE at the boundary into types that cannot hold invalid values.
Everything downstream takes the parsed type and does no re-checking.
Toy domain: an incoming order request (a JSON-ish dict from an HTTP handler).
Run: python examples/parse_dont_validate.py
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class ParseError(ValueError):
    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


@dataclass(frozen=True, slots=True)
class Sku:
    value: str
    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Z]{3,8}", self.value):
            raise ValueError("must be 3-8 uppercase letters")


@dataclass(frozen=True, slots=True)
class Quantity:
    value: int
    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int) or not 1 <= self.value <= 99:
            raise ValueError("must be an integer from 1 to 99")


@dataclass(frozen=True, slots=True)
class Money:                       # value object: equality by value, integer cents, no floats
    cents: int
    currency: str = "USD"
    def __add__(self, other: Money) -> Money:
        if other.currency != self.currency:
            raise ValueError("currency mismatch")
        return Money(self.cents + other.cents, self.currency)
    def __mul__(self, n: int) -> Money:
        return Money(self.cents * n, self.currency)


@dataclass(frozen=True, slots=True)
class OrderRequest:
    sku: Sku
    quantity: Quantity


def _sku(v: object) -> Sku:
    if not isinstance(v, str):
        raise ValueError("must be a string")
    return Sku(v)


def _quantity(v: object) -> Quantity:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError("must be an integer from 1 to 99")
    return Quantity(v)


def _field(raw: Mapping[str, object], name: str, parse: Callable[[object], T],
           errors: dict[str, str]) -> T | None:
    if name not in raw:
        errors[name] = "missing"
        return None
    try:
        return parse(raw[name])
    except ValueError as exc:
        errors[name] = str(exc)
        return None


def parse_order(raw: Mapping[str, object]) -> OrderRequest:
    """The only function that sees untrusted data. Collects every error, not just the first."""
    errors: dict[str, str] = {}
    sku = _field(raw, "sku", _sku, errors)
    quantity = _field(raw, "quantity", _quantity, errors)
    if sku is None or quantity is None:
        raise ParseError(errors)
    return OrderRequest(sku, quantity)


PRICES = {Sku("MUG"): Money(1200), Sku("TEE"): Money(1800)}


def line_total(req: OrderRequest) -> Money:      # no validation here: the type already guarantees it
    return PRICES[req.sku] * req.quantity.value


if __name__ == "__main__":
    good = parse_order({"sku": "MUG", "quantity": 2})
    print(good, "->", line_total(good))
    for raw in ({"sku": "mug", "quantity": 0}, {"quantity": True}):
        try:
            parse_order(raw)
        except ParseError as exc:
            print("rejected", raw, "->", exc.errors)
    print("value equality:", Money(500) + Money(700) == Money(1200),
          "| hashable as dict key:", Sku("MUG") in PRICES)
