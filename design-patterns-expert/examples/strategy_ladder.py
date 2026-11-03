"""Strategy, climbed one rung at a time, on one toy domain: exporting orders.

Rung 1: plain functions in a dict. Rung 2: a closure carrying configuration.
Rung 3: a dataclass, once the variant has state and more than one operation.
All three satisfy the same `Exporter` alias, so the caller never changes.
Run: python examples/strategy_ladder.py
"""
from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    sku: str
    quantity: int
    total_cents: int


Exporter = Callable[[list[Order]], str]          # the whole contract


def export_json(orders: list[Order]) -> str:       # rung 1: stateless function
    return json.dumps([asdict(o) for o in orders])


def export_csv(orders: list[Order]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["order_id", "sku", "quantity", "total_cents"])
    writer.writerows((o.order_id, o.sku, o.quantity, o.total_cents) for o in orders)
    return buf.getvalue()


def redacting(inner: Exporter, *, hide_totals: bool) -> Exporter:   # rung 2: closure
    def export(orders: list[Order]) -> str:
        if hide_totals:
            orders = [Order(o.order_id, o.sku, o.quantity, 0) for o in orders]
        return inner(orders)
    return export


@dataclass
class BatchedExporter:                              # rung 3: state + behaviour
    inner: Exporter
    batch_size: int = 2
    batches_written: int = 0

    def __call__(self, orders: list[Order]) -> str:
        chunks = [orders[i:i + self.batch_size] for i in range(0, len(orders), self.batch_size)]
        self.batches_written += len(chunks)
        return "\n---\n".join(self.inner(chunk) for chunk in chunks)


def run_export(orders: list[Order], exporter: Exporter) -> str:   # the caller never changes
    return exporter(orders)


def build_exporters() -> dict[str, Exporter]:      # composition root: the only concrete wiring
    return {
        "json": export_json,
        "csv": export_csv,
        "csv-public": redacting(export_csv, hide_totals=True),
    }


SAMPLE = [Order("A1", "MUG", 2, 2400), Order("A2", "TEE", 1, 1800), Order("A3", "CAP", 3, 4500)]

if __name__ == "__main__":
    exporters = build_exporters()
    for name in ("csv", "csv-public"):
        print(f"[{name}]\n{run_export(SAMPLE, exporters[name])}")
    batched = BatchedExporter(export_json, batch_size=2)   # keep the concrete type to read its state
    print("[json-batched]", run_export(SAMPLE, batched).count("---") + 1, "chunks;",
          "batches_written =", batched.batches_written)
