"""One shared HTTP client, configured deliberately.

httpx defaults every timeout to 5 seconds and the pool to 100 connections. Both
are fine for a demo and wrong for almost everything else: an LLM completion needs
a read timeout in the minutes, a health check needs one in the hundreds of
milliseconds, and a connect timeout of 5s means a dead host stalls your fan-out
for 5 seconds per attempt.

Verified against httpx 0.28.1 and httpx2 2.16.0: the AsyncClient signature and every default are
identical between them. To run this module on httpx2 without editing the imports:

    import httpx2; httpx2.alias_httpx()   # before importing aioutil

"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Mapping

import httpx

# httpx's own defaults, restated so the diff is visible:
#   Timeout(timeout=5.0)  -> connect=read=write=pool=5.0
#   Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=5.0)

FAST_TIMEOUT = httpx.Timeout(connect=3.0, read=10.0, write=10.0, pool=5.0)
"""Ordinary JSON APIs. Fail fast, retry at a higher level."""

SLOW_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=10.0)
"""Long-running generation endpoints. `read` is the gap between bytes, not the
total duration, so a streaming response only needs `read` to exceed the longest
silence between chunks."""


def build_client(
    *,
    base_url: str = "",
    timeout: httpx.Timeout = FAST_TIMEOUT,
    max_connections: int = 20,
    max_keepalive: int | None = None,
    headers: Mapping[str, str] | None = None,
    http2: bool = False,
    follow_redirects: bool = True,
) -> httpx.AsyncClient:
    """An AsyncClient with every setting that matters stated explicitly.

    `max_connections` is the real concurrency ceiling for one host. Setting it
    below your semaphore limit just moves the queueing into the pool, where it
    surfaces as PoolTimeout instead of as a wait — keep pool >= semaphore.

    `follow_redirects` defaults to False in httpx, unlike requests. Flipping it
    to True here is the surprise most people want removed.
    """
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive if max_keepalive is not None else max_connections,
            keepalive_expiry=30.0,
        ),
        headers=dict(headers or {}),
        http2=http2,
        follow_redirects=follow_redirects,
    )


@contextlib.asynccontextmanager
async def client_session(**kwargs) -> AsyncIterator[httpx.AsyncClient]:
    """`async with client_session() as client:` for scripts.

    In a long-lived service, do not use this per request. Build the client once
    in the app's lifespan and hand the same instance to every handler — see
    assets/examples/fastapi_service.py.
    """
    client = build_client(**kwargs)
    try:
        yield client
    finally:
        await client.aclose()
