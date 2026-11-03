"""Exhaustive dispatch on a closed set of types: `match` + a Union + `assert_never`.

The Pythonic replacement for Visitor and for `if isinstance(...) elif isinstance(...)`
chains when the set of types is CLOSED (you own all of them). Add a new event type to
the Union and forget a `case`, and mypy/pyright flag the `assert_never` line.
If the set is OPEN (other modules add types), use functools.singledispatch instead.
Toy domain: order events feeding a projection.
Run: python examples/match_dispatch.py      Check: mypy examples/match_dispatch.py
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import assert_never


@dataclass(frozen=True, slots=True)
class OrderPlaced:
    order_id: str
    total_cents: int


@dataclass(frozen=True, slots=True)
class OrderRefunded:
    order_id: str
    amount_cents: int


@dataclass(frozen=True, slots=True)
class OrderShipped:
    order_id: str
    carrier: str


OrderEvent = OrderPlaced | OrderRefunded | OrderShipped     # the closed set


def revenue_delta(event: OrderEvent) -> int:
    match event:
        case OrderPlaced(total_cents=cents):
            return cents
        case OrderRefunded(amount_cents=cents):
            return -cents
        case OrderShipped():
            return 0
        case _:
            assert_never(event)      # unreachable if every member of the Union has a case


@singledispatch                      # open set: other modules can register new types
def describe(event: object) -> str:
    raise TypeError(f"no describer for {type(event).__name__}")


@describe.register
def _(event: OrderPlaced) -> str:
    return f"{event.order_id} placed for ${event.total_cents / 100:.2f}"


@describe.register
def _(event: OrderRefunded | OrderShipped) -> str:       # union registration needs 3.11+
    return f"{event.order_id}: {type(event).__name__}"


if __name__ == "__main__":
    log: list[OrderEvent] = [OrderPlaced("A1", 2400), OrderShipped("A1", "UPS"),
                             OrderPlaced("A2", 1800), OrderRefunded("A1", 600)]
    print("revenue:", sum(revenue_delta(e) for e in log), "cents")
    for e in log:
        print(" ", describe(e))
