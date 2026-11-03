"""Hedged requests (Dean & Barroso, "The Tail at Scale", CACM 2013).

Send the request to one replica; if it has not answered by the p95 latency, send a
second copy to another replica and take whichever answers first. Because only requests
slower than p95 get a hedge, extra load is bounded near 5%. Only for IDEMPOTENT reads.
Part 1 simulates the latency maths; part 2 is a working asyncio implementation.
Toy domain: inventory lookups against replicated stock servers.
Run: python examples/hedged_request.py
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Coroutine
from typing import Any
from statistics import quantiles


def replica_latency_ms(rng: random.Random) -> float:
    """Mostly ~10 ms, but 2% of calls hit a 150-250 ms stall (GC pause, noisy neighbour)."""
    base = rng.lognormvariate(2.3, 0.25)
    return base + rng.uniform(150, 250) if rng.random() < 0.02 else base


def pct(xs: list[float], p: int) -> float:
    return quantiles(xs, n=100)[p - 1]


def simulate(n: int = 200_000, seed: int = 1) -> None:
    rng = random.Random(seed)
    plain = [replica_latency_ms(rng) for _ in range(n)]
    hedge_at = pct(plain, 95)
    hedged, extra = [], 0
    for _ in range(n):
        first = replica_latency_ms(rng)
        if first > hedge_at:
            extra += 1
            first = min(first, hedge_at + replica_latency_ms(rng))
        hedged.append(first)
    print(f"hedge after p95 = {hedge_at:.1f} ms; extra requests = {extra / n:.1%}")
    print(f"{'':10s} {'p50':>7s} {'p99':>7s} {'p99.9':>7s}")
    for label, xs in (("plain", plain), ("hedged", hedged)):
        p999 = quantiles(xs, n=1000)[998]
        print(f"{label:10s} {pct(xs, 50):7.1f} {pct(xs, 99):7.1f} {p999:7.1f}")


async def hedged_call(call: Callable[[int], Coroutine[Any, Any, str]],
                      hedge_after_s: float) -> tuple[str, int]:
    """Run call(0); if still pending after hedge_after_s, also run call(1). First result wins,
    the loser is cancelled. Returns (result, number of copies sent)."""
    primary: asyncio.Task[str] = asyncio.create_task(call(0))
    done, _ = await asyncio.wait({primary}, timeout=hedge_after_s)
    if done:
        return primary.result(), 1
    backup: asyncio.Task[str] = asyncio.create_task(call(1))
    done, pending = await asyncio.wait({primary, backup}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return done.pop().result(), 2


async def demo_async() -> None:
    delays = {0: 0.200, 1: 0.010}            # replica 0 is stalled, replica 1 is fast
    async def lookup(replica: int) -> str:
        await asyncio.sleep(delays[replica])
        return f"stock=7 from replica {replica}"
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    result, copies = await hedged_call(lookup, hedge_after_s=0.020)
    print(f"async: {result!r}, copies sent={copies}, took {1000 * (loop.time() - t0):.0f} ms")


if __name__ == "__main__":
    simulate()
    asyncio.run(demo_async())
