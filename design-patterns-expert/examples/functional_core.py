"""Functional Core, Imperative Shell.

The core DECIDES and returns effects as plain data; it never does I/O. The shell
gathers inputs, calls the core, and executes the effects. Tests of the core need no
mocks, no database, and no clock. Toy domain: nightly restock of low inventory.
Run: python examples/functional_core.py
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class StockLevel:
    sku: str
    on_hand: int
    reorder_point: int
    reorder_qty: int


@dataclass(frozen=True, slots=True)
class PlacePurchaseOrder:      # an effect, described, not performed
    sku: str
    quantity: int


@dataclass(frozen=True, slots=True)
class NotifyBuyer:
    message: str


Effect = PlacePurchaseOrder | NotifyBuyer


def plan_restock(levels: list[StockLevel], today: date, open_po_skus: frozenset[str]) -> list[Effect]:
    """Pure: same inputs, same output. All the business rules live here."""
    effects: list[Effect] = []
    for lvl in levels:
        if lvl.on_hand <= lvl.reorder_point and lvl.sku not in open_po_skus:
            effects.append(PlacePurchaseOrder(lvl.sku, lvl.reorder_qty))
    if today.weekday() == 0 and effects:                      # Mondays: summary to the buyer
        effects.append(NotifyBuyer(f"{len(effects)} purchase order(s) placed"))
    return effects


class Shell:
    """Imperative: talks to the world. Kept thin enough that reading it is the test."""
    def __init__(self, db: dict[str, StockLevel], open_pos: set[str], outbox: list[str]) -> None:
        self.db, self.open_pos, self.outbox = db, open_pos, outbox

    def run(self, today: date) -> None:
        effects = plan_restock(list(self.db.values()), today, frozenset(self.open_pos))
        for eff in effects:
            match eff:
                case PlacePurchaseOrder(sku, qty):
                    self.open_pos.add(sku)
                    self.outbox.append(f"PO {sku} x{qty}")
                case NotifyBuyer(msg):
                    self.outbox.append(f"email buyer: {msg}")


if __name__ == "__main__":
    levels = {s.sku: s for s in [StockLevel("MUG", 3, 5, 20), StockLevel("TEE", 40, 10, 50),
                                 StockLevel("CAP", 0, 2, 10)]}
    print("core, Monday :", plan_restock(list(levels.values()), date(2026, 10, 5), frozenset({"CAP"})))
    outbox: list[str] = []
    Shell(levels, set(), outbox).run(date(2026, 10, 5))
    print("shell outbox :", outbox)
