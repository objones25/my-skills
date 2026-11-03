"""Dict-backed adapter: the test double is also a real implementation."""
from __future__ import annotations


class InMemoryInventory:
    def __init__(self, stock: dict[str, int]) -> None:
        self._stock = dict(stock)

    def get_stock(self, sku: str) -> int | None:
        return self._stock.get(sku)

    def reserve(self, sku: str, quantity: int) -> int:
        self._stock[sku] -= quantity
        return self._stock[sku]
