"""Retry with Full Jitter, a retry budget (token bucket), a deadline, and Retry-After.

Corrects the reference row "429/503 -> back-pressure, not retry": AWS SDKs DO retry
throttling errors, with a longer base delay and a per-client token bucket that
stops retrying once retries keep failing. Toy domain: charging a payment gateway.
Time and randomness are injected so the demo is deterministic and instant.
Run: python examples/retry_budget.py
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass


class Permanent(Exception):          # 400, 401, 422: never retry
    pass


class Transient(Exception):          # timeout, 500, 502, 504
    retry_after: float | None = None


class Throttled(Transient):          # 429, 503: retry, but slower, and honour Retry-After
    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("throttled")
        self.retry_after = retry_after


@dataclass
class RetryBudget:
    """Per-client token bucket, AWS SDK defaults: 500 tokens, a retry costs 14 (5 if
    throttled), a success after retries refunds what it spent, a first-try success adds 1."""
    capacity: int = 500
    tokens: int = 500
    cost_transient: int = 14
    cost_throttled: int = 5

    def try_spend(self, exc: Transient) -> int:
        """Return the tokens spent (0 means the retry is refused)."""
        cost = self.cost_throttled if isinstance(exc, Throttled) else self.cost_transient
        if self.tokens < cost:
            return 0
        self.tokens -= cost
        return cost

    def on_success(self, spent: int) -> None:
        self.tokens = min(self.capacity, self.tokens + (spent if spent else 1))


@dataclass
class Clock:
    t: float = 0.0
    def now(self) -> float: return self.t
    def sleep(self, s: float) -> None: self.t += s


def call_with_retry(op: Callable[[], str], *, budget: RetryBudget | None, clock: Clock,
                    rng: random.Random, deadline_s: float = 10.0, max_attempts: int = 3,
                    base_s: float = 0.05, throttled_base_s: float = 1.0, cap_s: float = 5.0,
                    stats: dict[str, int] | None = None) -> str:
    stats = stats if stats is not None else {}
    deadline = clock.now() + deadline_s
    spent = 0
    for attempt in range(max_attempts):
        stats["attempts"] = stats.get("attempts", 0) + 1
        try:
            result = op()
        except Permanent:
            raise
        except Transient as exc:
            if attempt == max_attempts - 1:
                raise
            if budget is not None:
                cost = budget.try_spend(exc)
                if not cost:
                    stats["refused_by_budget"] = stats.get("refused_by_budget", 0) + 1
                    raise
                spent += cost
            base = throttled_base_s if isinstance(exc, Throttled) else base_s
            delay = rng.uniform(0, min(cap_s, base * 2 ** attempt))      # Full Jitter
            if exc.retry_after is not None:
                delay = max(delay, exc.retry_after)                       # honour Retry-After
            if clock.now() + delay > deadline:
                raise
            clock.sleep(delay)
        else:
            if budget is not None:
                budget.on_success(spent)
            return result
    raise AssertionError("unreachable")


def simulate(failure: Callable[[random.Random], Exception | None], calls: int,
             budget: RetryBudget | None, seed: int = 7) -> dict[str, int]:
    rng, clock, stats = random.Random(seed), Clock(), {"ok": 0, "failed": 0}
    def gateway() -> str:
        exc = failure(rng)
        if exc is not None:
            raise exc
        return "charged"
    for _ in range(calls):
        try:
            call_with_retry(gateway, budget=budget, clock=clock, rng=rng, stats=stats)
            stats["ok"] += 1
        except (Transient, Permanent):
            stats["failed"] += 1
    return stats


def flaky(rng: random.Random) -> Exception | None:      # healthy: 5% transient failures
    return Transient("timeout") if rng.random() < 0.05 else None


def outage(rng: random.Random) -> Exception | None:     # down: everything throttled
    return Throttled(retry_after=0.5)


if __name__ == "__main__":
    rows = [("healthy, no budget", flaky, None), ("healthy, budget", flaky, RetryBudget()),
            ("outage, no budget", outage, None), ("outage, budget", outage, RetryBudget())]
    print(f"{'scenario':20s} {'calls':>5s} {'attempts':>8s} {'ok':>5s} {'failed':>6s} {'refused':>7s}")
    for label, mode, budget in rows:
        s = simulate(mode, 1000, budget)
        print(f"{label:20s} {1000:5d} {s['attempts']:8d} {s['ok']:5d} {s['failed']:6d} "
              f"{s.get('refused_by_budget', 0):7d}")
