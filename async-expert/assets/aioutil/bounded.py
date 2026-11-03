"""Bounded fan-out. The concurrency limit is the API, not an afterthought.

`asyncio.gather(*[fetch(u) for u in urls])` over 5,000 urls opens 5,000 sockets
at once. It will not be faster than 20 at a time — you will hit the connection
pool, the remote rate limiter, or your own file-descriptor ceiling, and the
failure mode is a wall of timeouts that looks like the remote host being down.

Verified against Python 3.14.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class Ok(Generic[R]):
    index: int
    value: R


@dataclass(frozen=True)
class Err:
    index: int
    error: BaseException


async def bounded_map(
    fn: Callable[[T], Awaitable[R]],
    items: Sequence[T],
    *,
    limit: int = 10,
) -> list[R]:
    """Run `fn` over `items` with at most `limit` in flight. All-or-nothing.

    Uses a TaskGroup, so the first failure cancels every sibling and the call
    raises an ExceptionGroup. Results come back in input order regardless of
    completion order.

    Use this when a partial result is useless — a batch that must be complete,
    a migration step, a set of calls whose outputs get combined.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    sem = asyncio.Semaphore(limit)
    results: list[R] = [None] * len(items)  # type: ignore[list-item]

    async def run(index: int, item: T) -> None:
        async with sem:
            results[index] = await fn(item)

    async with asyncio.TaskGroup() as tg:
        for i, item in enumerate(items):
            tg.create_task(run(i, item), name=f"bounded_map[{i}]")

    return results


async def bounded_map_settled(
    fn: Callable[[T], Awaitable[R]],
    items: Sequence[T],
    *,
    limit: int = 10,
) -> list[Ok[R] | Err]:
    """Same, but one failure does not cancel the rest. Every item gets a verdict.

    Use this when partial results have value — scraping 10,000 URLs where 30
    are dead, enriching a dataset, any batch where you want the successes and a
    list of what to look at.

    `asyncio.gather(..., return_exceptions=True)` is the stdlib equivalent, but
    it loses which input produced which exception once you have more than a
    handful, and it gives you no concurrency limit.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    sem = asyncio.Semaphore(limit)
    results: list[Ok[R] | Err] = [None] * len(items)  # type: ignore[list-item]

    async def run(index: int, item: T) -> None:
        async with sem:
            try:
                results[index] = Ok(index, await fn(item))
            except asyncio.CancelledError:
                raise  # a cancel is not a per-item failure; let it propagate
            except Exception as exc:  # noqa: BLE001 - deliberately settling
                results[index] = Err(index, exc)

    async with asyncio.TaskGroup() as tg:
        for i, item in enumerate(items):
            tg.create_task(run(i, item), name=f"settled[{i}]")

    return results


def partition(results: Sequence[Ok[R] | Err]) -> tuple[list[R], list[Err]]:
    """Split a settled run into values and failures."""
    values = [r.value for r in results if isinstance(r, Ok)]
    errors = [r for r in results if isinstance(r, Err)]
    return values, errors


async def cpu_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int | None = None,
) -> list[R]:
    """Run a CPU-bound sync function across processes without blocking the loop.

    `fn` must be a module-level function (picklable), and the calling script
    needs an `if __name__ == "__main__":` guard — ProcessPoolExecutor re-imports
    the module in each worker.

    Threads do not help here. Under the GIL, `asyncio.to_thread` on CPU-bound
    work gives you the same wall-clock as running it serially, plus overhead.

    On failure or cancellation the pool is shut down without waiting, so the
    loop is not blocked while the remaining items finish. A `with
    ProcessPoolExecutor()` block would call `shutdown(wait=True)` on the loop
    thread and freeze every task until the slowest item returned.
    """
    loop = asyncio.get_running_loop()
    pool = ProcessPoolExecutor(max_workers=max_workers)
    try:
        futures = [loop.run_in_executor(pool, fn, item) for item in items]
        results = list(await asyncio.gather(*futures, return_exceptions=False))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    # Every item has finished, so joining the workers is quick, but it is still
    # a blocking call: do it off the loop.
    await asyncio.to_thread(pool.shutdown, wait=True)
    return results
