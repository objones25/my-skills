"""Every mistake the audit catches, in one file. Do not copy this.

Run `python scripts/audit_async.py assets/examples/bad_example.py` and you should
see one finding per numbered comment below. This file is the audit's fixture: if
you add a check to audit_async.py, add its trigger here.
"""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import threading
import time
import urllib.request

import httpx

URLS = [f"https://example.com/{i}" for i in range(500)]


async def fetch(url: str) -> str:
    # 1. BLOCKING_SLEEP — freezes every other task on the loop for a full second
    time.sleep(1.0)
    # 2. BLOCKING_HTTP — the sync API, called from a coroutine
    response = httpx.get(url)
    # 3. BLOCKING_HTTP — same problem, different library
    urllib.request.urlopen(url)
    # 4. BLOCKING_OPEN — synchronous file I/O on the loop thread
    with open("/tmp/cache.txt", "w") as handle:
        handle.write(response.text)
    # 5. BLOCKING_SUBPROCESS
    subprocess.run(["echo", "done"], check=False)
    # 6. BLOCKING_SYNC_DB
    sqlite3.connect("cache.db")
    # 7. BLOCKING_LOCK — a threading primitive parks the loop thread
    threading.Lock()
    return response.text


async def fetch_all_wrong() -> list[str]:
    # 8. NO_CLIENT_TIMEOUT — inherits the 5s default nobody chose
    # 9. CLIENT_IN_LOOP — a fresh connection pool per iteration
    out = []
    for url in URLS:
        async with httpx.AsyncClient() as client:
            # 10. SEQUENTIAL_AWAIT — 500 round trips, one at a time
            out.append(await fetch(url))
    return out


async def gather_wrong() -> None:
    # 11. BARE_GATHER — no failure policy chosen
    # 12. UNBOUNDED_FANOUT — 500 concurrent sockets, no semaphore anywhere
    await asyncio.gather(*[fetch(u) for u in URLS])

    # 13. WAIT_ON_COROUTINES — forbidden since Python 3.11
    await asyncio.wait([fetch(u) for u in URLS[:3]])


async def tasks_wrong() -> None:
    # 14. FLOATING_TASK — the loop keeps only a weak reference to this
    asyncio.create_task(fetch(URLS[0]))
    # 15. UNAWAITED_COROUTINE — never runs at all, just warns at GC time
    fetch(URLS[1])


async def cancel_wrong() -> None:
    try:
        await asyncio.sleep(10)
    # 16. SWALLOWED_CANCEL — this task can now never be cancelled or timed out
    except asyncio.CancelledError:
        print("ignoring cancellation")


async def rows_wrong():
    async with httpx.AsyncClient(timeout=5.0) as client:
        # 17. ASYNC_GEN_YIELD_IN_CM: if the consumer stops early, the client is
        #     only closed when the generator is finalized, possibly never
        yield await client.get(URLS[0])


def legacy_entrypoint() -> None:
    # 18. LEGACY_LOOP: raises RuntimeError on Python 3.14
    loop = asyncio.get_event_loop()
    # 19. LEGACY_LOOP
    loop.run_until_complete(gather_wrong())
