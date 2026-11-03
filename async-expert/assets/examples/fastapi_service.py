"""A FastAPI service that does not block its own event loop.

Three rules do most of the work:

1. One `AsyncClient` for the process, created in `lifespan`, not per request.
2. `async def` handlers contain only awaitable work. Anything blocking either
   moves to a thread explicitly, or the handler becomes a plain `def` so
   Starlette runs it in its threadpool for you.
3. Every outbound call has a timeout shorter than the caller's patience.

Run:  uvicorn fastapi_service:app --reload
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypedDict

import anyio.to_thread
import httpx
from fastapi import FastAPI, HTTPException, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aioutil import bounded_map_settled, build_client, partition  # noqa: E402

UPSTREAM_LIMIT = 10


class State(TypedDict):
    http: httpx.AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    """Startup and shutdown. `on_event("startup")` is the deprecated spelling."""
    async with build_client(base_url="https://httpbin.org", max_connections=UPSTREAM_LIMIT) as client:
        # Blocking work runs in two separate pools, sized separately:
        # - anyio's default limiter (40 tokens): plain `def` handlers, sync
        #   dependencies, run_in_threadpool, anyio.to_thread.run_sync.
        # - the event loop's default executor, min(32, cpu_count + 4) threads:
        #   asyncio.to_thread and run_in_executor(None, ...). On a 2-vCPU
        #   container that is 6 threads, whatever the anyio limiter says.
        # Size both against the database's own connection limit.
        anyio.to_thread.current_default_thread_limiter().total_tokens = 100
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=40))
        yield {"http": client}


app = FastAPI(lifespan=lifespan)


def get_client(request: Request) -> httpx.AsyncClient:
    return request.state.http


@app.get("/proxy/{path}")
async def proxy(path: str, request: Request) -> dict:
    """Correct: awaits an async client, bounded by an outer timeout."""
    client = get_client(request)
    try:
        async with asyncio.timeout(5.0):
            response = await client.get(f"/{path}")
    except TimeoutError:
        # Return 504 rather than hanging the caller for the client's own timeout.
        raise HTTPException(status_code=504, detail="upstream timed out") from None
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream failed: {exc!r}") from exc
    return {"status": response.status_code, "bytes": len(response.content)}


@app.get("/fanout")
async def fanout(request: Request, n: int = 5) -> dict:
    """Correct: bounded concurrency, and partial failure does not 500 the request."""
    client = get_client(request)
    paths = [f"/bytes/{128 * (i + 1)}" for i in range(n)]

    async def one(path: str) -> int:
        response = await client.get(path)
        response.raise_for_status()
        return len(response.content)

    settled = await bounded_map_settled(one, paths, limit=UPSTREAM_LIMIT)
    sizes, failures = partition(settled)
    return {"ok": len(sizes), "failed": len(failures), "total_bytes": sum(sizes)}


def slow_report(rows: int) -> int:
    """A blocking third-party call: sync DB driver, pandas, PDF render, whatever."""
    time.sleep(0.5)
    return rows * 2


@app.get("/report-sync")
def report_sync(rows: int = 10) -> dict:
    """Correct: a plain `def` handler. Starlette runs it in the threadpool, so
    the blocking call never touches the event loop thread."""
    return {"value": slow_report(rows)}


@app.get("/report-async")
async def report_async(rows: int = 10) -> dict:
    """Also correct: an `async def` handler that hands the blocking call off
    explicitly. Use this form when the handler has awaitable work too."""
    value = await asyncio.to_thread(slow_report, rows)
    return {"value": value}


# WRONG, kept as a labeled counterexample:
#
# @app.get("/report-broken")
# async def report_broken(rows: int = 10) -> dict:
#     return {"value": slow_report(rows)}   # 0.5s of dead air for EVERY request
#                                           # in flight, not just this one.
