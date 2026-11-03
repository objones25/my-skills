"""Server-side idempotency keys, following the IETF draft (-07, expired) and Stripe's docs.

Cache key is (tenant, client key), never the client key alone. A fingerprint of the
body detects key reuse with a different payload (422). A retry while the first call is
in flight gets 409. A retry after completion gets the STORED response, even an error.
Toy domain: charging an order; the response to the first call is "lost in flight".
Run: python examples/idempotency.py
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

Response = tuple[int, dict[str, object]]


class Phase(Enum):
    IN_FLIGHT = auto()
    DONE = auto()


@dataclass
class Record:
    fingerprint: str
    phase: Phase
    response: Response | None = None


@dataclass
class IdempotencyStore:
    records: dict[tuple[str, str], Record] = field(default_factory=dict)

    def handle(self, tenant: str, key: str | None, body: dict[str, object],
               execute: Callable[[dict[str, object]], Response]) -> Response:
        if key is None:
            return 400, {"title": "Idempotency-Key is missing"}
        fp = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        slot = (tenant, key)
        rec = self.records.get(slot)
        if rec is not None:
            if rec.fingerprint != fp:
                return 422, {"title": "Idempotency-Key is already used"}
            if rec.phase is Phase.IN_FLIGHT:
                return 409, {"title": "A request is outstanding for this Idempotency-Key"}
            assert rec.response is not None
            return rec.response                                  # replay, success or error
        self.records[slot] = Record(fp, Phase.IN_FLIGHT)         # reserve BEFORE executing
        try:
            response = execute(body)
        except Exception:
            del self.records[slot]                               # crashed before a result: stay retryable
            raise
        self.records[slot] = Record(fp, Phase.DONE, response)
        return response


@dataclass
class Gateway:
    charges: list[int] = field(default_factory=list)

    def charge(self, body: dict[str, object]) -> Response:
        amount = body["amount_cents"]
        assert isinstance(amount, int)
        self.charges.append(amount)
        return 201, {"charge_id": f"ch_{len(self.charges)}", "amount_cents": amount}


if __name__ == "__main__":
    store, gw = IdempotencyStore(), Gateway()
    body: dict[str, object] = {"order_id": "A1", "amount_cents": 2400}
    first = store.handle("tenant-1", "k-123", body, gw.charge)   # client never sees this
    retries = [store.handle("tenant-1", "k-123", body, gw.charge) for _ in range(3)]
    print("first:", first)
    print("retries:", retries)
    print("gateway charged", len(gw.charges), "time(s)")
    print("reuse with different body:", store.handle("tenant-1", "k-123",
          {**body, "amount_cents": 9999}, gw.charge)[0])
    print("same key, other tenant:", store.handle("tenant-2", "k-123", body, gw.charge)[0],
          "-> charges now", len(gw.charges))
    print("missing key:", store.handle("tenant-1", None, body, gw.charge)[0])
