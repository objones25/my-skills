"""Retry with exponential backoff and full jitter.

httpx's own `AsyncHTTPTransport(retries=N)` retries connection *establishment*
only — `ConnectError` and `ConnectTimeout`. It never retries a read timeout, a
mid-flight protocol error, or a 429/503 response. That is documented behavior,
not a bug, and it means the retry policy that actually matters is yours.

Verified against httpx 0.28.1 and httpx2 2.16.0.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field

import httpx

ALWAYS_RETRYABLE: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)
"""Safe to retry for any method: the request never reached the server. A pool
timeout means it never left the client; a connect failure means no connection
was ever established."""

IDEMPOTENT_ONLY_RETRYABLE: tuple[type[Exception], ...] = (
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)
"""The request WAS sent. The server may have received and applied it, and only
the response went missing. Retrying a POST here can double-charge a card.
Retried only for idempotent methods unless the caller opts in."""

RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = ALWAYS_RETRYABLE + IDEMPOTENT_ONLY_RETRYABLE
"""The full set, for idempotent requests. Note what is absent: WriteTimeout and
WriteError, which fire mid-body and are never safe to replay blindly."""

IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE", "PUT", "DELETE"})
"""RFC 9110 §9.2.2. PUT and DELETE are idempotent (repeating them lands the same
state) even though they are not safe. POST and PATCH are neither."""

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

ALWAYS_RETRYABLE_STATUS = frozenset({408, 425, 429})
"""Statuses where the server says it did not act on the request: 408 (it never
received the whole request, RFC 9110 15.5.9), 425 (it refused to risk a replay,
RFC 8470), 429 (rate limited, RFC 6585). Safe to resend for any method. A 500,
502, 503 or 504 after a POST gives no such guarantee: the handler or upstream
may already have applied it, so those are retried for idempotent methods only."""


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 4
    """Total tries, not extra tries. 4 means 1 request + 3 retries."""
    base_delay: float = 0.5
    max_delay: float = 30.0
    retry_status: frozenset[int] = RETRYABLE_STATUS
    retry_exceptions: tuple[type[Exception], ...] = field(default=RETRYABLE_EXCEPTIONS)
    respect_retry_after: bool = True
    retry_non_idempotent: bool = False
    """Off by default. When False, a ReadTimeout or RemoteProtocolError on a
    POST or PATCH is raised rather than retried, and a 500/502/503/504 is
    returned rather than replayed, because the server may already have applied it. Set True only when the endpoint is genuinely safe to
    replay — it takes an idempotency key, or the operation is a no-op twice."""

    def exceptions_for(self, method: str) -> tuple[type[Exception], ...]:
        """Narrow the retryable set to what is safe for this HTTP method."""
        if self.retry_non_idempotent or method.upper() in IDEMPOTENT_METHODS:
            return self.retry_exceptions
        return tuple(e for e in self.retry_exceptions if e not in IDEMPOTENT_ONLY_RETRYABLE)

    def statuses_for(self, method: str) -> frozenset[int]:
        """Same narrowing for status codes: a POST that got a 502 is not replayed."""
        if self.retry_non_idempotent or method.upper() in IDEMPOTENT_METHODS:
            return self.retry_status
        return self.retry_status & ALWAYS_RETRYABLE_STATUS

    def backoff(self, attempt: int, rng: random.Random) -> float:
        """Full jitter: uniform(0, min(cap, base * 2**attempt)).

        Full jitter beats "exponential + small jitter" because it decorrelates
        clients completely. Without it, N clients that all failed at the same
        moment retry at the same moment, and the thundering herd repeats at
        every backoff level.
        """
        ceiling = min(self.max_delay, self.base_delay * (2**attempt))
        return rng.uniform(0.0, ceiling)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None  # HTTP-date form; fall back to computed backoff


DEFAULT_POLICY = RetryPolicy()
"""Module-level singleton. A mutable default in a signature is a footgun even
when the dataclass is frozen — ruff's B008 flags the call site, not the value."""


class RetriesExhausted(Exception):
    """Raised when every attempt failed. `.attempts` holds what went wrong."""

    def __init__(self, attempts: list[BaseException | int]) -> None:
        self.attempts = attempts
        super().__init__(f"all {len(attempts)} attempts failed: {attempts!r}")


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    policy: RetryPolicy = DEFAULT_POLICY,
    rng: random.Random | None = None,
    **kwargs,
) -> httpx.Response:
    """Send one request, retrying transient failures. Returns the final response.

    Non-retryable status codes (a 404, a 422) come back as a normal Response —
    this function does not call `raise_for_status()` for you, because whether a
    404 is an error depends on the caller.

    The retryable sets narrow automatically for non-idempotent methods: a
    ReadTimeout on a POST is raised and a 502 on a POST is returned, not
    retried, unless the policy sets `retry_non_idempotent=True`.

    A `Retry-After` longer than `policy.max_delay` ends the retries (raising
    RetriesExhausted) instead of sleeping past the caller's budget.
    """
    rng = rng or random.Random()
    history: list[BaseException | int] = []
    retryable = policy.exceptions_for(method)
    retry_status = policy.statuses_for(method)

    for attempt in range(policy.attempts):
        delay: float | None = None
        try:
            response = await client.request(method, url, **kwargs)
        except asyncio.CancelledError:
            raise  # never swallow cancellation, never retry through it
        except retryable as exc:
            history.append(exc)
        else:
            if response.status_code not in retry_status:
                return response
            history.append(response.status_code)
            if policy.respect_retry_after:
                delay = _retry_after_seconds(response)

        if attempt == policy.attempts - 1:
            break
        if delay is not None and delay > policy.max_delay:
            break  # the server asked for a longer wait than this policy allows
        await asyncio.sleep(delay if delay is not None else policy.backoff(attempt, rng))

    raise RetriesExhausted(history)


async def with_retry(
    fn: Callable[[], Awaitable[object]],
    *,
    policy: RetryPolicy = DEFAULT_POLICY,
    retry_on: Iterable[type[Exception]] = RETRYABLE_EXCEPTIONS,
    rng: random.Random | None = None,
):
    """Generic form for anything awaitable — a DB call, an SDK method, a socket.

    Unlike `request_with_retry`, this has no HTTP method to inspect, so it cannot
    narrow the retryable set for you. The default includes ReadTimeout and
    RemoteProtocolError, which are only safe when `fn` is idempotent. Pass
    `retry_on=ALWAYS_RETRYABLE` when it is not.
    """
    rng = rng or random.Random()
    retry_on = tuple(retry_on)
    history: list[BaseException | int] = []

    for attempt in range(policy.attempts):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except retry_on as exc:
            history.append(exc)
            if attempt == policy.attempts - 1:
                break
            await asyncio.sleep(policy.backoff(attempt, rng))

    raise RetriesExhausted(history)
