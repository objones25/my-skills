# httpx / httpx2

Verified against httpx 0.28.1 **and** httpx2 2.16.0 on Python 3.14. Signatures and default values
below were read out of the installed packages, not recalled.

Contents: [Which package](#which-package-read-this-first) · [The client](#the-client-is-the-whole-thing) ·
[Defaults](#constructor-defaults-you-are-inheriting) · [Timeouts](#the-four-timeouts) ·
[Exceptions](#exception-hierarchy) · [Retries](#retries-what-httpx-does-and-does-not-do) ·
[Streaming](#streaming) · [SSE and WebSockets](#server-sent-events-and-websockets-httpx2-only) ·
[Limits](#connection-limits) · [HTTP/2](#http2) · [Testing](#testing) · [anyio](#anyio-underneath)

## Which package: read this first

There are now three things called httpx, and picking the wrong one is the first decision.

| Package | Latest | Released | Status |
|---|---|---|---|
| `httpx` | 0.28.1 | 2024-12-06 | No stable release in 22 months. PyPI trove: *Development Status :: 4 - Beta*. Issues and discussions closed to the public 2026-02-27. |
| `httpx` `1.0.devN` | 1.0.dev6 | 2026-08-31 | Pre-release of a rewrite. Its PyPI README calls the redesign *"currently in early development and not yet public."* Do not install. |
| `httpx2` | 2.16.0 | 2026-10-21 | **A fork of httpx 0.28.1 maintained by Pydantic.** Trove: *Development Status :: 5 - Production/Stable*. 21 releases since 2.0.0b1 (2026-05-11). Requires Python 3.11+ since 2.15.0. |

httpx2's own changelog states the relationship plainly: *"a fork of httpx maintained by Pydantic.
Forked from httpx 0.28.1 (commit b5addb6)… Renamed package: `httpx` -> `httpx2`… **No other public
API changed.** Renamed transitive dependency: `httpcore` -> `httpcore2`."*

Verified by diffing the two installed packages:

```
httpx2 defaults: Timeout(timeout=5.0) Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=5.0)
httpx  defaults: Timeout(timeout=5.0) Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=5.0)
AsyncClient params differ: set()                      <-- identical constructor
in httpx2 not httpx: ['EventSource', 'FunctionAuth', 'Origin', 'SSEError',
                      'ServerSentEvent', 'alias_httpx', 'query', 'websocket']
in httpx not httpx2: ['main']
```

So everything in the rest of this file applies to both: the API, the defaults, the exception
hierarchy, the retry semantics. httpx2 adds native server-sent events and WebSockets on top.
Differences that can still bite after a switch (from httpx2's migration guide): certificates are
verified against the OS trust store via `truststore` instead of `certifi` (since 2.3.0), the
User-Agent is `python-httpx2/<version>`, and the logger is `httpx2`, not `httpx`.

**For a new project, install `httpx2`.** Starlette has warned on plain httpx in its `TestClient`
since 1.2.0 (2026-05-28); still true on 1.9.0:

```
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated;
install `httpx2` instead.
```

FastAPI's `standard` and `all` extras have listed `httpx2>=2.0.0` (next to `httpx<1.0.0`) since
0.143.4 (2026-10-18), so `pip install "fastapi[standard]"` gets httpx2 and a warning-free
`TestClient`. A bare `pip install fastapi httpx` still warns. Before 0.143.4 no extra pulled httpx2.

**For an existing codebase**, `httpx2.alias_httpx()` makes `import httpx` resolve to httpx2
process-wide, so migration does not require touching every import. It must run before anything
imports `httpx`; called afterwards it raises `RuntimeError: httpx was already imported`. It is for
applications, not libraries. Verified: `assets/aioutil/` was written against httpx and passes its
test suite unmodified under httpx2 after the alias call:

```python
import httpx2
httpx2.alias_httpx()
import httpx            # -> httpx2
```

Pin whichever you choose. A loose `httpx>=0.28` constraint can pull in `1.0.devN` from a
pre-release-enabled resolver, and that is a different library.

*Benchmark comparisons between httpx, aiohttp, and requests circulating in 2026 come almost
entirely from commercial proxy and scraping vendors publishing SEO content. Treat their numbers as
marketing. No neutral primary-source benchmark was found; measure on your own workload.
`aiohttp` 3.14.5 (2026-10-11) remains the actively-released asyncio-only alternative.*

## The client is the whole thing

```python
# Wrong at any scale beyond a one-off script
for url in urls:
    r = await httpx.AsyncClient().get(url)   # new pool, new TLS handshake, every time
```

From the docs: *"When you make requests using the top-level API… HTTPX has to establish a new
connection for every single request."* A `Client` *"will reuse the underlying TCP connection,
instead of recreating one for every single request."* On HTTPS that saves a full TLS handshake per
call, which on a fan-out of 500 is most of the wall clock.

One client for the process. In a script, one `async with` around the whole run. In a service, one
created in `lifespan`. Never one per request, and never one inside a loop (`CLIENT_IN_LOOP`).

## Constructor defaults you are inheriting

Read directly from `httpx._config` on 0.28.1:

```
DEFAULT_TIMEOUT_CONFIG = Timeout(timeout=5.0)     # connect = read = write = pool = 5.0s
DEFAULT_LIMITS         = Limits(max_connections=100, max_keepalive_connections=20,
                                keepalive_expiry=5.0)
DEFAULT_MAX_REDIRECTS  = 20
follow_redirects       = False       # unlike requests
http2                  = False
AsyncHTTPTransport(retries=...)      # default 0
```

Two of these surprise people:

- **`timeout=5.0` applies to all four phases.** An LLM completion that takes 90 seconds fails at
  5. A health check that should fail in 300ms waits 5 seconds. Neither is what you want; set it.
- **`follow_redirects=False`.** `requests` follows redirects by default. Code ported from
  `requests` silently starts returning 301s with empty bodies.

## The four timeouts

| Phase | Bounds | Raises |
|---|---|---|
| `connect` | Establishing the TCP/TLS connection | `httpx.ConnectTimeout` |
| `read` | Waiting for the **next chunk** of response data | `httpx.ReadTimeout` |
| `write` | Waiting to send the next chunk of the request body | `httpx.WriteTimeout` |
| `pool` | Waiting to acquire a connection from the pool | `httpx.PoolTimeout` |

`read` is per-chunk, not total. This is the single most useful fact about httpx timeouts: a
streaming download of a 2 GB file does not need a two-hour `read` timeout, it needs one longer
than the worst pause between packets. Conversely, `read` alone does not cap total duration: a
server that dribbles one byte per second forever never trips it. Use `asyncio.timeout()` around
the call when you need a wall-clock ceiling.

```python
httpx.Timeout(10.0)                          # all four
httpx.Timeout(10.0, connect=3.0)             # 10s everywhere except connect
httpx.Timeout(connect=3.0, read=600.0, write=30.0, pool=5.0)   # each named
client.get(url, timeout=30.0)                # per-request override
httpx.Timeout(None)                          # disabled — you almost never want this
```

A `PoolTimeout` in production almost always means your concurrency limit is above
`max_connections`. Fix the limit, not the timeout.

## Exception hierarchy

```
HTTPError
├── RequestError                 (something went wrong sending / receiving)
│   ├── TransportError
│   │   ├── TimeoutException  →  ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout
│   │   ├── NetworkError      →  ConnectError, ReadError, WriteError, CloseError
│   │   ├── ProtocolError     →  LocalProtocolError, RemoteProtocolError
│   │   ├── ProxyError
│   │   └── UnsupportedProtocol
│   ├── DecodingError
│   └── TooManyRedirects
└── HTTPStatusError              (raised only by response.raise_for_status())
```

`raise_for_status()` raises `HTTPStatusError` and nothing else. `except httpx.HTTPError` catches
both branches, which is usually what a caller wants. `StreamError` (with `ResponseNotRead`,
`StreamConsumed`) sits outside `HTTPError` entirely: those are API-misuse errors, not network
failures.

Retryability comes in three tiers, and the middle one depends on the HTTP method:

- **Always safe**: `ConnectError`, `ConnectTimeout`, `PoolTimeout`. The request never reached the
  server.
- **Idempotent methods only**: `ReadTimeout`, `RemoteProtocolError`. The request *was* sent and
  only the response went missing, so the server may have applied it. Fine to retry a GET or PUT,
  not a POST.
- **Never blind-retried**: `WriteTimeout`, `WriteError`. These fire mid-body.

`assets/aioutil/retry.py` narrows the set by method automatically, with
`RetryPolicy(retry_non_idempotent=True)` as the explicit opt-out.

## Retries: what httpx does and does not do

The docs are precise and widely misread:

> Requests will be retried the given number of times in case an `httpx.ConnectError` or an
> `httpx.ConnectTimeout` occurs, allowing smoother operation under flaky networks.
>
> If you need other forms of retry behaviors, such as handling read/write errors or reacting to
> `503 Service Unavailable`, consider general-purpose tools such as [tenacity]…

So `AsyncHTTPTransport(retries=3)` covers **connection establishment only**. It never retries a
read timeout, a mid-flight protocol error, or any HTTP status code. If your reliability story is
"we set retries=3", you have no reliability story for the failures that actually happen at scale.

What a real policy needs:

1. **Exponential backoff with full jitter.** `sleep = random(0, min(cap, base * 2**attempt))`,
   the formula from Marc Brooker's AWS post. Fixed delays resynchronize every failing client into
   a herd that fails together on every subsequent attempt.
2. **Honor `Retry-After`** when the server sends it. A 429 with `Retry-After: 30` is the server
   telling you the answer; computing your own backoff ignores it. Cap it: a `Retry-After: 3600`
   should end the retry loop, not park the task for an hour.
3. **A retryable set that excludes write-side failures** (above), and for POST/PATCH also excludes
   500/502/503/504 responses: the server may have acted before failing.
4. **Never retry through a cancellation.** `except asyncio.CancelledError: raise` before the
   retryable branch, or your retry loop defeats every timeout wrapping it.
5. **A cap on total attempts, not just delay.** Unbounded retry against a hard-down dependency is
   a self-inflicted outage.

## Streaming

```python
async with client.stream("GET", url) as response:
    response.raise_for_status()
    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
        ...
```

Iterators: `aiter_bytes(chunk_size=None)`, `aiter_lines()`, `aiter_text()`, and `aiter_raw()`
(skips content decoding: gzip, br, zstd). Touching `response.content` or `.text` on a streamed
response you have not read raises `httpx.ResponseNotRead`:
*"Attempted to access streaming response content, without having called `read()`."*

The body is only available inside the `async with` block. Returning the response object out of it
and reading later gives you a closed stream.

Server-sent events and token-by-token LLM output are the common case for `aiter_lines()`. Note
that `read` timeout applies between chunks, so a model that thinks for 40 seconds before its first
token needs `read` above 40, not above the total generation time.

## Server-sent events and WebSockets (httpx2 only)

Hand-parsing `data: ` lines out of `aiter_lines()` is the standard workaround on httpx 0.28.
httpx2 has a real SSE client, which matters because SSE has edge cases (multi-line `data:`,
`event:`, `id:`, comment lines, `retry:`) that ad-hoc parsers get wrong:

```python
async with client.sse("/v1/messages", method="POST", json=payload) as events:
    async for event in events:          # ServerSentEvent
        if event.event == "content_block_delta":
            print(event.json()["delta"]["text"], end="")   # json() is a method, not a property
```

`ServerSentEvent` exposes `data`, `event`, `id`, `retry`, and `json()`. `client.sse(...)` takes the
same arguments as `request()` plus `max_event_size` (default 1 MiB). WebSockets are
`client.websocket(url)`, yielding an `AsyncWebSocketSession` with keepalive pings on by default
(20s interval, 20s timeout).

## Connection limits

```python
httpx.Limits(max_connections=20,           # hard ceiling on concurrent connections
             max_keepalive_connections=20, # how many stay open between requests
             keepalive_expiry=30.0)        # seconds an idle connection survives
```

Keep `max_connections` ≥ your semaphore limit. If the semaphore allows 50 concurrent calls and the
pool allows 20, the extra 30 queue inside httpx and eventually raise `PoolTimeout`: the same
work, but the wait now looks like a failure.

## HTTP/2

`pip install "httpx[http2]"` then `AsyncClient(http2=True)`. HTTP/2 multiplexes many requests over
one TCP connection, so a fan-out of 100 requests to one host stops needing 100 connections. That
is exactly where it pays; for requests scattered across many hosts it changes little. Both ends
must support it: check `response.http_version` rather than assuming.

## Testing

```python
transport = httpx.ASGITransport(app=app)
async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
    r = await client.get("/health")
```

The old `AsyncClient(app=app)` shortcut was deprecated in 0.27.0 and **removed in 0.28.0**: code
using it fails on any current version. `WSGITransport` is the sync/Flask equivalent.

For unit-testing your own retry and error handling without a network, `httpx.MockTransport` takes a
handler function and lets you script a failure sequence:

```python
calls = itertools.count()
def handler(request):
    n = next(calls)
    if n == 0:
        return httpx.Response(503, headers={"retry-after": "0"})
    if n == 1:
        raise httpx.ConnectTimeout("boom", request=request)
    return httpx.Response(200, json={"attempt": n})

client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=1.0)
```

Real output from `assets/aioutil/retry.py` against exactly that handler:

```
request_with_retry: status=200 body={'attempt': 2} after 0.019s
RetriesExhausted: attempts=[500, 500, 500]
non-retryable 404 returned directly: 404
```

## anyio underneath

httpx's networking goes through httpcore, which uses anyio (only the ASGI test transport touches
`asyncio` directly), so `AsyncClient` runs unmodified under `trio.run()` or
`anyio.run(..., backend="trio")`. From the docs: *"HTTPX supports
either asyncio or trio as an async environment. It will auto-detect which of those two to use."*
The 1.0 pre-releases drop the anyio dependency (1.0.dev6 requires only `truststore` and
`typing-extensions`), so whether trio support survives the rewrite is unknown.

## Sources

- Clients and connection pooling: https://www.python-httpx.org/advanced/clients/
- Timeouts: https://www.python-httpx.org/advanced/timeouts/
- Resource limits: https://www.python-httpx.org/advanced/resource-limits/
- Transports, retries, ASGITransport, MockTransport: https://www.python-httpx.org/advanced/transports/
- Exceptions: https://www.python-httpx.org/exceptions/
- Async support: https://www.python-httpx.org/async/
- HTTP/2: https://www.python-httpx.org/http2/
- CHANGELOG (0.27.0 deprecation, 0.28.0 removal of `app=`): https://github.com/encode/httpx/blob/master/CHANGELOG.md
- 1.0 status: https://www.encode.io/httpnext/ and https://pypi.org/project/httpx/1.0.dev6/
- Issue tracker closure, 2026-02-27: https://github.com/encode/httpx/discussions/3784
- httpx2 on PyPI (version, trove classifier, dependencies): https://pypi.org/project/httpx2/
- httpx2 CHANGELOG (the fork statement, `alias_httpx`): https://github.com/pydantic/httpx2/blob/main/src/httpx2/CHANGELOG.md
- httpx2 migration guide (alias rules, trust store, User-Agent): https://pydantic.dev/docs/httpx2/get-started/migration/
- Starlette release notes (testclient httpx2 support, 1.2.0): https://github.com/Kludex/starlette/blob/master/docs/release-notes.md
- FastAPI dependency metadata (httpx2 in extras from 0.143.4): https://pypi.org/pypi/fastapi/0.143.4/json
- Marc Brooker, *Exponential Backoff And Jitter*, AWS Architecture Blog: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
