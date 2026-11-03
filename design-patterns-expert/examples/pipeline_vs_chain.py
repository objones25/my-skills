"""Chain of Responsibility vs Pipeline, on the same order domain.

Chain: handlers are tried in order and the FIRST one that can handle the request
decides; the rest never run. Pipeline: EVERY stage runs and each one's output is the
next one's input. behavioral.md previously labelled a pipeline as a chain.
The pipeline here is built from generators (the Iterator pattern, Python-native):
stages are lazy, so a million orders stream through without being held in memory.
Run: python examples/pipeline_vs_chain.py
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Refund:
    order_id: str
    amount_cents: int


Approver = Callable[[Refund], str | None]          # None means "not mine, pass it on"


def auto_approve(r: Refund) -> str | None:
    return "auto-approved" if r.amount_cents <= 5_000 else None


def team_lead(r: Refund) -> str | None:
    return "team lead approved" if r.amount_cents <= 50_000 else None


def finance(r: Refund) -> str | None:              # the terminal handler always answers
    return "sent to finance"


def decide(r: Refund, chain: list[Approver], trace: list[str] | None = None) -> str:
    for handler in chain:
        if trace is not None:
            trace.append(handler.__name__)
        verdict = handler(r)
        if verdict is not None:
            return verdict                          # early exit: the defining property
    raise LookupError("no handler accepted the request")


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    sku: str
    quantity: int
    total_cents: int = 0
    tax_cents: int = 0


PRICES = {"MUG": 1200, "TEE": 1800}


def priced(orders: Iterable[Order]) -> Iterator[Order]:
    for o in orders:
        yield replace(o, total_cents=PRICES[o.sku] * o.quantity)


def taxed(orders: Iterable[Order], rate_bp: int = 775) -> Iterator[Order]:    # basis points
    for o in orders:
        yield replace(o, tax_cents=o.total_cents * rate_bp // 10_000)


def large_only(orders: Iterable[Order], min_cents: int = 3_000) -> Iterator[Order]:
    return (o for o in orders if o.total_cents >= min_cents)


def pipeline(orders: Iterable[Order]) -> Iterator[Order]:
    return large_only(taxed(priced(orders)))        # every stage runs, in this order


if __name__ == "__main__":
    chain: list[Approver] = [auto_approve, team_lead, finance]
    for cents in (1_500, 20_000, 90_000):
        trace: list[str] = []
        print(f"refund {cents:>6} -> {decide(Refund('A1', cents), chain, trace):20s} tried {trace}")
    raw = [Order("A1", "MUG", 2), Order("A2", "TEE", 1), Order("A3", "TEE", 3)]
    for o in pipeline(raw):
        print(f"{o.order_id}: total {o.total_cents} tax {o.tax_cents}")
    stream = pipeline(Order(f"X{i}", "MUG", 3) for i in range(1_000_000))
    print("first of a million, computed lazily:", next(stream))
