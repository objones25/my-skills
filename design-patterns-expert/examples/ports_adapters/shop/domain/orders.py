from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class OrderRequest:
    sku: str
    quantity: int


@dataclass(frozen=True, slots=True)
class OrderPlaced:
    sku: str
    quantity: int
    remaining: int


class UnknownSKU(Exception):
    def __init__(self, sku: str) -> None:
        super().__init__(sku)
        self.sku = sku


class OutOfStock(Exception):
    def __init__(self, sku: str, available: int, requested: int) -> None:
        super().__init__(f"{sku}: {available} available, {requested} requested")
        self.sku, self.available, self.requested = sku, available, requested


class InventoryPort(Protocol):           # the port, in domain vocabulary
    def get_stock(self, sku: str) -> int | None: ...
    def reserve(self, sku: str, quantity: int) -> int: ...


def place_order(req: OrderRequest, inventory: InventoryPort) -> OrderPlaced:
    available = inventory.get_stock(req.sku)
    if available is None:
        raise UnknownSKU(req.sku)
    if available < req.quantity:
        raise OutOfStock(req.sku, available, req.quantity)
    return OrderPlaced(req.sku, req.quantity, inventory.reserve(req.sku, req.quantity))
