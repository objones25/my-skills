# Services, batched API calls, and mixed pipelines

The three shapes async code actually takes in practice.

---

# 1. FastAPI / Starlette services

## The `def` vs `async def` rule

FastAPI's own guidance, verbatim:

> If you are using a third party library that communicates with something (a database, an API, the
> file system, etc.) and doesn't have support for using `await`… then declare your path operation
> functions as normally, with just `def`.
>
> When you declare a path operation function with normal `def` instead of `async def`, it is run in
> an external threadpool that is then awaited, instead of being called directly (as it would block
> the server).
>
> If you just don't know, use normal `def`.

So there are exactly three correct handler shapes, and one wrong one:

```python
@app.get("/a")                       # correct: all work is awaitable
async def a():
    return await client.get("/x")

@app.get("/b")                       # correct: all work is blocking -> threadpool
def b():
    return sync_db.query(...)

@app.get("/c")                       # correct: mixed -> explicit handoff
async def c():
    rows = await asyncio.to_thread(sync_db.query, ...)
    extra = await client.get("/x")
    return combine(rows, extra)

@app.get("/d")                       # WRONG
async def d():
    return sync_db.query(...)        # blocks EVERY in-flight request, not just this one
```

Handler `d` is the defining bug of async web services. It returns the right answer, passes tests
under one concurrent user, and collapses under load. Nothing raises. `ruff`'s `ASYNC2xx` rules and
`scripts/audit_async.py` both catch the common forms.

FastAPI's docs never spell out "this halts every other request" in those words. They state the
converse guarantee (that `def` handlers are threadpooled *because* calling them directly "would
block the server"). The severity follows from the single-threaded loop, not from a doc sentence.

## Lifespan, not per-request clients

```python
class State(TypedDict):
    http: httpx.AsyncClient

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    async with build_client(base_url=UPSTREAM) as client:
        yield {"http": client}

app = FastAPI(lifespan=lifespan)

@app.get("/proxy")
async def proxy(request: Request):
    client = request.state.http      # the same client for every request
```

`@app.on_event("startup")` is the deprecated spelling: *"If you provide a `lifespan` parameter,
`startup` and `shutdown` event handlers will no longer be called. It's all `lifespan` or all
events, not both."*

The same rule covers database pools, message-queue connections, and model handles: anything with
a setup cost belongs in lifespan.

## The threadpool ceilings (there are two)

Blocking work in a FastAPI process runs in one of two independent thread pools:

| Pool | Who uses it | Default size | How to raise it |
|---|---|---|---|
| anyio's default `CapacityLimiter` | plain `def` handlers, sync dependencies, `run_in_threadpool`, `anyio.to_thread.run_sync` | **40 tokens** | `anyio.to_thread.current_default_thread_limiter().total_tokens = N` |
| the event loop's default `ThreadPoolExecutor` | `asyncio.to_thread`, `loop.run_in_executor(None, ...)` | `min(32, cpu_count + 4)`: **6 on a 2-vCPU container** | `loop.set_default_executor(ThreadPoolExecutor(max_workers=N))` |

Measured on Starlette 1.9.0 / anyio 4.15.1: 60 blocked `asyncio.to_thread` calls left the anyio
limiter at `borrowed_tokens == 0`, ran on threads named `asyncio_N`, and only 6 ran at once.

So a service with sync DB calls in `def` handlers caps at 40 concurrent requests per process, and
request 41 queues silently. The same service written as `async def` + `asyncio.to_thread` caps at
the executor size instead, which on a small container is far lower, and raising the anyio limiter
does nothing for it.

```python
import anyio.to_thread
from concurrent.futures import ThreadPoolExecutor

anyio.to_thread.current_default_thread_limiter().total_tokens = 100
asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=40))
```

Set both in `lifespan` (the anyio call raises `NoEventLoopError` outside a running loop). Size them
against your database's own connection limit; 100 threads fighting over a 20-connection pool just
moves the queue.

## Timeouts at the edge

A handler should fail faster than its caller gives up:

```python
try:
    async with asyncio.timeout(5.0):
        response = await client.get(path)
except TimeoutError:
    raise HTTPException(status_code=504, detail="upstream timed out") from None
except httpx.HTTPError as exc:
    raise HTTPException(status_code=502, detail=f"upstream failed: {exc!r}") from exc
```

Without the outer timeout, the client's own timeout governs, and a 60s client timeout in a
handler whose caller quits at 5s means 55 seconds of a worker slot held for a response nobody will
read. Under load that is how a slow dependency takes down a healthy service.

`assets/examples/fastapi_service.py` is the runnable version of all of the above.

---

# 2. Batching LLM and API calls

The canonical async workload: N independent requests, a remote rate limit, and results you want
back in order.

```python
from aioutil import build_client, bounded_map_settled, partition, request_with_retry, RetryPolicy

async def classify_all(texts: list[str]) -> tuple[list[str], list]:
    async with build_client(base_url=API, timeout=SLOW_TIMEOUT, max_connections=8,
                            headers={"authorization": f"Bearer {key}"}) as client:

        async def one(text: str) -> str:
            r = await request_with_retry(
                client, "POST", "/v1/messages",
                json={"model": MODEL, "messages": [{"role": "user", "content": text}]},
                policy=RetryPolicy(attempts=5, base_delay=1.0, max_delay=60.0),
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]

        settled = await bounded_map_settled(one, texts, limit=8)
    return partition(settled)
```

The decisions inside that, each of which has a wrong default:

**Concurrency limit.** Comes from the provider's documented requests-per-minute, not from
`len(texts)`. If the limit is 50 RPM and each call takes 4 seconds, `50/60 × 4 ≈ 3.3`, so a
concurrency of 3 or 4 saturates it. Higher just generates 429s that you then pay to retry.

**Settled, not all-or-nothing.** One malformed input in a batch of 2,000 should not discard 1,999
completed calls that each cost money. `bounded_map_settled` returns an `Ok`/`Err` per item with
the input index attached, so a retry pass over just the failures is a one-liner.

**A long `read` timeout, a short `connect` timeout.** Generation legitimately takes minutes;
connecting does not. `SLOW_TIMEOUT` in `assets/aioutil/client.py` is `connect=5, read=600`.

**Retries that honor `Retry-After`.** Providers send it on 429. Computing your own backoff when
the server told you the number is how you stay rate-limited.

**Pool ≥ semaphore.** `max_connections=8` matches `limit=8`. Mismatched, the extra calls queue in
httpx and surface as `PoolTimeout`.

For token-bucket rate limiting rather than a concurrency cap ("60 requests per minute" as opposed
to "8 at a time"), a semaphore is the wrong primitive. You want a limiter that refills on a clock;
`aiolimiter`'s `AsyncLimiter(60, 60)` is the usual choice, used as `async with limiter:` inside the
same guarded function.

## Streaming responses

```python
async with client.stream("POST", "/v1/messages", json=payload) as response:
    response.raise_for_status()
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            yield json.loads(line[6:])
```

Remember `read` is the gap between chunks. A model that thinks for 40 seconds before its first
token needs `read > 40`, not `read >` total generation time.

---

# 3. Mixed I/O + CPU pipelines

The shape of most data and ML preprocessing work: fetch (I/O-bound), then transform (CPU-bound).
Using one tool for both is why "I added async and it got no faster."

`assets/examples/good_pipeline.py`, run against a local server:

```
downloaded 4/5 in 0.11s
hashed     4 in 0.19s
  52939228a3fbddd9.bin 51bec38da208486d
  ...
  FAILED [4] http://127.0.0.1:8899/missing.bin: HTTPStatusError: Client error '404 File not found'
```

Stage 1 is asyncio with a semaphore. Stage 2 is a process pool. Threads would give stage 2
nothing: under the GIL, N threads hashing take the same wall clock as one, plus overhead.

Two failure modes specific to the process-pool half:

**The `__main__` guard is not optional.** `ProcessPoolExecutor` re-imports the module in each
worker. Without the guard, 3.14 on Linux (default start method now `forkserver`) and every
macOS/Windows version (`spawn`) fail with:

```
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.
```

followed by `BrokenProcessPool` or `ConnectionResetError` in the parent. Run from stdin
(`python - <<EOF`) the worker instead fails with `FileNotFoundError: ... '<stdin>'`. On Linux with
3.13 or earlier (`fork`) the unguarded script runs fine, which is how the bug reaches a 3.14 box.

**Everything crossing the boundary is pickled.** The function must be module-level; arguments and
return values are copied. Sending a 2 GB array to a worker costs a 2 GB serialize plus a 2 GB
deserialize. Pass a path and let the worker read it.

## Overlapping the stages

The example above runs stage 1 to completion, then stage 2. For long runs, overlap them with a
bounded queue so downloading continues while hashing happens:

```python
queue: asyncio.Queue[Path | None] = asyncio.Queue(maxsize=32)

async def producer():
    async with build_client() as client:
        async def one(url):
            await queue.put(await stream_to_disk(client, url))
        await bounded_map(one, urls, limit=8)
    await queue.put(None)                      # sentinel

async def consumer(pool, loop):
    while (path := await queue.get()) is not None:
        yield await loop.run_in_executor(pool, checksum, path)
```

`maxsize=32` is the backpressure: once 32 downloaded files are waiting to be hashed, downloading
pauses. Without it, a fast network and a slow CPU stage fills the disk and then memory. On 3.13+,
`Queue.shutdown()` replaces the sentinel-value dance.

## Sources

- FastAPI, *Concurrency and async / await*: https://fastapi.tiangolo.com/async/
- FastAPI, *Lifespan Events*: https://fastapi.tiangolo.com/advanced/events/
- Starlette, *Lifespan*: https://starlette.dev/lifespan/
- Starlette, *Threadpool*: https://starlette.dev/threadpool/
- anyio, *Working with threads* (the 40-token limiter): https://anyio.readthedocs.io/en/stable/threads.html
- Python docs, `asyncio.to_thread` and `ThreadPoolExecutor` default size: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread · https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
- Python docs, *What's New in 3.14* (forkserver default): https://docs.python.org/3/whatsnew/3.14.html
- Python docs, `loop.run_in_executor` and the multiprocessing guard: https://docs.python.org/3/library/asyncio-eventloop.html
- Python docs, `asyncio.Queue.shutdown`: https://docs.python.org/3/library/asyncio-queue.html
