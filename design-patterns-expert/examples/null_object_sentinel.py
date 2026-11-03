"""Sentinel Object and Null Object (Rhodes, python-patterns.guide/python/sentinel-object).

Sentinel: a unique object marks "argument not supplied" when None is a legitimate
value. Identity, not value, is what the code checks. Null Object: an object with the
real interface and do-nothing behaviour, so callers never branch on None.
Toy domain: order lookup with an optional discount and optional audit logging.
Run: python examples/null_object_sentinel.py
"""
from __future__ import annotations

from enum import Enum
from typing import Final, Literal, Protocol


class _Missing(Enum):
    """A one-member Enum: same identity semantics as object(), but type checkers
    can narrow on `is MISSING`, which a bare object() sentinel does not allow."""
    TOKEN = "MISSING"


MISSING: Final = _Missing.TOKEN

DISCOUNTS: dict[str, int | None] = {"A1": 10, "A2": None}   # None = "explicitly no discount"


def discount_percent(order_id: str,
                     default: int | Literal[_Missing.TOKEN] | None = MISSING) -> int | None:
    value = DISCOUNTS.get(order_id, MISSING)
    if value is not MISSING:
        return value                       # may legitimately be None
    if default is MISSING:
        raise KeyError(order_id)           # caller gave no default: unknown id is an error
    return default


class AuditLog(Protocol):
    def record(self, event: str) -> None: ...


class ListAuditLog:
    def __init__(self) -> None:
        self.events: list[str] = []
    def record(self, event: str) -> None:
        self.events.append(event)


class NullAuditLog:                         # same interface, deliberately does nothing
    def record(self, event: str) -> None:
        pass


def apply_discount(order_id: str, total_cents: int, audit: AuditLog) -> int:
    pct = discount_percent(order_id, default=None) or 0
    audit.record(f"{order_id}: {pct}% off")     # no `if audit is not None` anywhere
    return total_cents * (100 - pct) // 100


if __name__ == "__main__":
    print("A1 ->", discount_percent("A1"), "| A2 ->", discount_percent("A2"),
          "| ZZ with default ->", discount_percent("ZZ", default=0))
    try:
        discount_percent("ZZ")
    except KeyError as exc:
        print("ZZ without default -> KeyError", exc)
    log = ListAuditLog()
    print(apply_discount("A1", 2400, log), log.events)
    print(apply_discount("A1", 2400, NullAuditLog()), "(nothing recorded, no None checks)")
