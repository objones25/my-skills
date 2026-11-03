"""Download concurrently, process in parallel, stay inside your limits.

The shape of almost every real pipeline: an I/O-bound stage that wants
concurrency, and a CPU-bound stage that wants parallelism. They need different
tools, and using the wrong one is the most common reason "I added async and it
got no faster."

    IO-bound (network, disk)   -> asyncio + a semaphore.  One thread, many waits.
    CPU-bound (hash, decode)   -> process pool.           Many cores, real parallelism.

Run:  python good_pipeline.py
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aioutil import (  # noqa: E402
    RetryPolicy,
    bounded_map_settled,
    build_client,
    cpu_map,
    partition,
    with_retry,
)

DOWNLOAD_LIMIT = 8
"""Concurrent requests in flight. Pick this from the remote host's documented
rate limit, not from how many items you happen to have."""

CPU_WORKERS = os.cpu_count() or 4
OUT_DIR = Path("downloads")


def dest_for(url: str) -> Path:
    return OUT_DIR / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".bin")


async def stream_to_disk(client: httpx.AsyncClient, url: str) -> Path:
    """Stream one response to disk. Never holds the whole body in memory.

    `read` in the client timeout is the gap between chunks, not the total
    download time, so a 2 GB file over a slow link does not need a 2-hour
    timeout — it needs one longer than the worst pause between packets.
    """
    dest = dest_for(url)
    # Even opening and closing the file is blocking I/O, so all three of open,
    # write, and close go to a thread. This is what "no aiofiles dependency"
    # looks like when you take it seriously.
    handle = await asyncio.to_thread(dest.open, "wb")
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                await asyncio.to_thread(handle.write, chunk)
    finally:
        await asyncio.to_thread(handle.close)
    return dest


def checksum(path: Path) -> tuple[str, str]:
    """CPU-bound. A plain sync module-level function, so it can be pickled."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return path.name, digest.hexdigest()


async def main(urls: list[str]) -> None:
    started = time.perf_counter()

    # One client for the whole run. Rebuilding it per request throws away the
    # connection pool, which is most of what makes this fast.
    async with build_client(max_connections=DOWNLOAD_LIMIT) as client:

        async def fetch_one(url: str) -> Path:
            # httpx's transport-level `retries=` covers connection setup only,
            # so read timeouts are retried here instead. A 503 surfaces as
            # HTTPStatusError, which with_retry's default set does not retry;
            # use request_with_retry when status-based retry matters.
            return await with_retry(
                lambda: stream_to_disk(client, url),
                policy=RetryPolicy(attempts=3),
            )

        # Settled, not all-or-nothing: one dead URL should not discard 200 good
        # downloads. Failures come back attached to their input index.
        settled = await bounded_map_settled(fetch_one, urls, limit=DOWNLOAD_LIMIT)

    paths, failures = partition(settled)
    fetched_at = time.perf_counter()

    # Second stage: real parallelism, because hashing is CPU work.
    digests = await cpu_map(checksum, paths, max_workers=CPU_WORKERS)

    print(f"downloaded {len(paths)}/{len(urls)} in {fetched_at - started:.2f}s")
    print(f"hashed     {len(digests)} in {time.perf_counter() - fetched_at:.2f}s")
    for name, digest in digests:
        print(f"  {name} {digest[:16]}")
    for err in failures:
        print(f"  FAILED [{err.index}] {urls[err.index]}: {type(err.error).__name__}: {err.error}")


if __name__ == "__main__":
    # Required: ProcessPoolExecutor re-imports this module inside each worker.
    OUT_DIR.mkdir(exist_ok=True)  # sync context: no loop to block
    TARGETS = [f"https://httpbin.org/bytes/{n}" for n in (1024, 2048, 4096, 8192)]
    asyncio.run(main(TARGETS))
