---
name: async-expert
description: Write, review, and debug concurrent Python (asyncio, httpx/httpx2, anyio, structured concurrency). Use whenever the user asks to make code concurrent or "faster with async", batch API or LLM calls, add retries, timeouts or rate limiting to network code, build or fix an async pipeline, write a FastAPI service that touches the network or a database, or debug code that is slow, hanging, deadlocked, or silently dropping tasks. Also use proactively when writing any code that awaits, when the user mentions asyncio, the event loop, TaskGroup, gather, semaphore, httpx, httpx2, aiohttp, anyio, trio, to_thread, or ProcessPoolExecutor, and when reviewing code that mixes blocking calls with async ones.
---

# Async Expert

Concurrency in Python fails quietly. Blocking code inside an `async def` still returns the right
answer: it just serializes the whole program while looking concurrent. A fire-and-forget task
gets garbage collected halfway through and nothing is logged until the process exits. A
`gather()` over 5,000 URLs opens 5,000 sockets and reports the resulting timeouts as if the
remote host were down. None of these raise. This skill exists to make them loud.

Verified against **Python 3.14** (asyncio claims also run on 3.11 to 3.13), **httpx 0.28.1**,
**httpx2 2.16.0**, **anyio 4.15.1**, **FastAPI 0.160.0 / Starlette 1.9.0**, **ruff 0.17.1**
(October 2026). Version matters: `asyncio.get_event_loop()` raises `RuntimeError` on 3.14 instead
of creating a loop, `TaskGroup` and `asyncio.timeout` (3.11) changed what correct code looks like,
and the HTTP client to reach for changed name (rules 21 to 28).

## Definitions used throughout

Terms that carry weight here, defined once.

| Term | Meaning |
|---|---|
| **Concurrency** | Many operations in flight at once. One thread, interleaved at `await` points. |
| **Parallelism** | Many operations executing at the same instant, on different cores. Needs processes in CPython. |
| **Event loop** | The scheduler. A `while` loop that runs ready tasks until each one hits an `await` that blocks, then runs the next. One thread. |
| **Coroutine function** | A function defined `async def`. Calling it runs *no code*: it returns a coroutine object. |
| **Coroutine object** | What the call returns. Inert until awaited or wrapped in a Task. |
| **Task** | A coroutine handed to the loop for scheduling. `asyncio.create_task(coro)` makes one. This is where concurrency starts. |
| **Awaitable** | Anything usable with `await`: a coroutine, a Task, or a Future. |
| **Future** | A low-level "result will exist later" box. You almost never create one directly. |
| **Checkpoint / yield point** | An `await` that actually suspends, giving the loop a chance to run something else. Code with no checkpoints is a blocked loop. |
| **I/O-bound** | Time is spent waiting on something external: network, disk, database. Async helps. |
| **CPU-bound** | Time is spent computing. Async does not help; processes do. |
| **Blocking call** | A synchronous function that waits. Inside `async def`, it stops *every* task, not just its own. |
| **Fan-out** | Starting N operations concurrently from one place. |
| **Backpressure** | A mechanism that slows producers when consumers fall behind. A bounded queue is one. |
| **Structured concurrency** | Every task's lifetime is bounded by a lexical scope, so a block cannot exit while its children run. `TaskGroup` and anyio task groups implement this. |
| **Cancel scope** | anyio/trio's region that can be cancelled as a unit, with a deadline. `asyncio.timeout()` is the narrow asyncio equivalent. |
| **ExceptionGroup** | PEP 654 container holding several exceptions raised concurrently. Caught with `except*`. |
| **GIL** | The lock making only one thread run Python bytecode at a time. Why threads do not speed up CPU-bound work. |

## Workflow

**1. Measure before converting anything.**
"Make it async" is a guess until you know where the time goes. Profile first: `scalene`,
`py-spy`, or `cProfile`. Time spent in system calls and waiting is I/O-bound and async will help;
time spent in Python bytecode is CPU-bound and async will not. Converting CPU-bound code to
asyncio makes it slower and harder to read. `references/debugging.md` has the profiling recipes.

**2. Pick the right tool from the profile.**

| What the profile shows | Tool |
|---|---|
| Waiting on network/disk, and an async library exists | asyncio + that library |
| Waiting on network/disk, only a blocking library exists | `asyncio.to_thread()` |
| Computing, in Python | `ProcessPoolExecutor` via `loop.run_in_executor()` |
| Computing, in a C extension that releases the GIL (NumPy, PyTorch) | threads work; measure both |

**3. Read the code before rewriting it.**
Find the actual I/O boundaries. Never assume a function is blocking because of its name: open it.
A wrapper that already returns a coroutine does not need `to_thread`, and wrapping it breaks it.

**4. Write against the rules.**
Condensed below; full rationale and sources in `references/rules.md`. For the loop, tasks, and
cancellation read `references/asyncio-core.md`; for HTTP read `references/httpx.md`; for anyio and
cancel scopes read `references/structured-concurrency.md`; for FastAPI, LLM batching, and
pipelines read `references/services-and-pipelines.md`.

**5. Run the audit, then the linter, then the code.**

```bash
python scripts/audit_async.py src/
ruff check --config assets/ruff-async.toml src/
python -X dev app.py
pytest -W error::RuntimeWarning -W error::pytest.PytestUnraisableExceptionWarning
```

In order: the audit (static shape mistakes, exit 1 on any finding), ruff's ASYNC rules (blocking
library calls), a dev-mode run (logs slow callbacks and shows where an un-awaited coroutine was
created), and the test suite with "coroutine was never awaited" promoted to a failure. A bare
`python -W error::RuntimeWarning app.py` does **not** fail on an un-awaited coroutine: the warning
fires inside a finalizer, so CPython prints "Exception ignored" and exits 0. Only pytest's
unraisable-exception hook turns it into a failing test.

The two static tools overlap deliberately and neither is a superset. `ruff`'s `ASYNC2xx` knows
more blocking library calls; `audit_async.py` knows the shape mistakes: unbounded fan-out,
`gather` with no failure policy, swallowed cancellation, client-per-iteration.

**6. Prove the concurrency is real.**
Print wall-clock time for the whole run and compare it to the sum of the individual operations.
If N operations of 1s each take N seconds, nothing is concurrent: you have async syntax wrapped
around serial execution, which is the most common outcome of a first conversion. Say in your
summary what the before and after numbers were. This step is what separates working async code
from code that merely compiles.

**7. Prove failure and cancellation work.**
Kill one item in the batch and confirm the rest behave the way you chose (all-or-nothing, or
settled). Ctrl-C the process mid-run and confirm it exits promptly instead of hanging: a hang
means something swallowed `CancelledError`.

## The hard gates

Enforced by `scripts/audit_async.py`, `ruff`'s ASYNC rules, or the runtime. Codes in
parentheses are what the audit prints.

**The loop is one thread**

1. No blocking call inside `async def`: `time.sleep`, `requests`, sync `httpx`, `open()`,
   `subprocess.run`, `sqlite3`, `input()` (`BLOCKING_*`, `ASYNC2xx`). One blocking call stalls
   every task in the process, not just the one that made it.
2. Blocking I/O you cannot replace goes to a thread: `await asyncio.to_thread(fn, *args)`.
   It is I/O-bound work only: under the GIL, threading CPU-bound work buys nothing.
3. CPU-bound work goes to processes, via `loop.run_in_executor(ProcessPoolExecutor(), fn, x)`.
   The function must be module-level and picklable, and the script needs an
   `if __name__ == "__main__":` guard or the workers re-import and crash.
4. No `threading.Lock`/`Event` inside `async def` (`BLOCKING_LOCK`). Use `asyncio.Lock`/`Event`;
   the threading versions park the loop thread.
5. Nothing between two `await`s should take longer than ~100ms. That is the threshold asyncio
   debug mode logs at, and it is a reasonable definition of "you blocked the loop."

**Tasks and fan-out**

6. `asyncio.TaskGroup` is the default for running things concurrently. `gather` is the exception,
   used when you specifically want failures collected rather than propagated.
7. Every `gather` states `return_exceptions=` explicitly (`BARE_GATHER`). The default `False` does
   not cancel siblings: it leaves them running orphaned while the exception propagates.
8. Never discard a `create_task` result (`FLOATING_TASK`, `RUF006`). Docs: *"The event loop only
   keeps weak references to tasks. A task that isn't referenced elsewhere may get garbage
   collected at any time, even before it's done."* `TaskGroup` holds strong references for you.
9. Calling a coroutine function without `await` or `create_task` runs nothing (`UNAWAITED_COROUTINE`).
   mypy reports it as `unused-coroutine`; in tests, run pytest with
   `-W error::RuntimeWarning -W error::pytest.PytestUnraisableExceptionWarning` so it fails the test.
10. `asyncio.wait()` takes tasks, not coroutines: passing coroutines has been forbidden since
    3.11 (`WAIT_ON_COROUTINES`).

**Bounds**

11. Every fan-out has a numeric concurrency limit (`UNBOUNDED_FANOUT`). "However many items I
    happen to have" is not a limit. Use `asyncio.Semaphore(n)`, or `assets/aioutil/bounded.py`.
12. The limit is chosen from the constraint that binds first: the remote host's documented rate
    limit, your connection pool size, or your file-descriptor ceiling. Write down which one.
13. Semaphore limit ≤ HTTP client `max_connections`. If the semaphore is larger, the queueing
    moves into the connection pool and surfaces as `PoolTimeout` instead of as a wait.
14. Producer/consumer uses `asyncio.Queue(maxsize=n)`. An unbounded queue is a memory leak with a
    slow consumer.

**Timeouts and cancellation**

15. Every network `await` sits inside a timeout. `async with asyncio.timeout(5):` wraps a whole
    block; `wait_for` wraps a single awaitable and is now implemented on top of `timeout()`.
16. A timeout you cannot reach is not a timeout: a client-level timeout of 60s inside a request
    handler the caller abandons after 5s protects nobody.
17. `except asyncio.CancelledError` must re-raise (`SWALLOWED_CANCEL`). Swallowing it means the
    task can never be cancelled, so every timeout above it becomes a hang.
18. `CancelledError` inherits from `BaseException` (since 3.8), so `except Exception` correctly
    lets it through. Bare `except:` and `except BaseException:` do not: those are the bug.
19. `asyncio.shield()` only for cleanup that must finish (releasing a lock, flushing a buffer).
    Shielding ordinary work turns Ctrl-C into a hang.
20. On failure, `TaskGroup` cancels siblings and raises an `ExceptionGroup` (catch with `except*`);
    `gather(return_exceptions=False)` cancels nothing and raises the first exception bare;
    `gather(return_exceptions=True)` raises nothing and returns exceptions as values. Pick
    deliberately: `references/asyncio-core.md` shows the timing of each with real output.

**HTTP**

Rules 21 to 28 apply identically to `httpx` 0.28.1 and the Pydantic-maintained `httpx2` fork: the
`AsyncClient` signature and every default are identical between them (verified by diff). For a new
project install `httpx2`: `httpx` has had no stable release since December 2024, Starlette's
`TestClient` warns on plain httpx, and FastAPI's `standard` extra pulls in httpx2 since 0.143.4.
Read the first section of `references/httpx.md` before adding either as a dependency.

21. One client for the process, not one per request (`CLIENT_IN_LOOP`). A new client is a new
    connection pool, which is a new TLS handshake for every single call.
22. Set `timeout=` explicitly on the client (`NO_CLIENT_TIMEOUT`). httpx's default is 5s for
    connect, read, write, and pool alike: wrong in both directions for most workloads.
23. `follow_redirects` defaults to **False** in httpx, unlike `requests`. Set it deliberately.
24. `AsyncHTTPTransport(retries=N)` retries connection *establishment* only: `ConnectError` and
    `ConnectTimeout`. It never retries a read timeout or a 503. Your retry policy is your job.
25. Retries use exponential backoff with **full jitter** and honor `Retry-After`. Fixed-delay
    retries synchronize every failing client into a herd.
26. Never replay a non-idempotent request (POST, PATCH) after the server may have acted on it: a
    read timeout, a write-side failure, or a 500/502/503/504 (a 5xx can come from a proxy after the
    origin acted). For those methods, replay only on 408, 425 and 429, or add an idempotency key.
27. Stream anything that could be large: `async with client.stream(...)` plus `aiter_bytes()`.
    `response.content` on an unread streamed response raises `ResponseNotRead`.
28. `raise_for_status()` raises `HTTPStatusError`. Transport failures raise `RequestError`
    subclasses. `httpx.HTTPError` is the base that catches both.

**Services**

29. In a `async def` handler, blocking code stalls every in-flight request. Either make the
    handler a plain `def` (Starlette runs it in a threadpool) or hand the blocking call to
    `asyncio.to_thread`. Do not do both.
30. The shared client is created in `lifespan`, not per request. `@app.on_event("startup")` is
    the deprecated spelling.
31. Two thread pools, sized separately. anyio's default limiter (40 tokens) is shared by every
    `def` handler, sync dependency, and `anyio.to_thread.run_sync` call; raise it with
    `anyio.to_thread.current_default_thread_limiter().total_tokens = N`. `asyncio.to_thread` does
    not use it: it runs on the loop's default executor, `min(32, cpu_count + 4)` threads (6 on a
    2-vCPU box); raise that with `loop.set_default_executor(ThreadPoolExecutor(N))`.

**Entry point**

32. `asyncio.run(main())`. Not `get_event_loop()` (raises `RuntimeError` on 3.14), not
    `new_event_loop()`, not `run_until_complete()` (`LEGACY_LOOP`).
33. Run with `asyncio.run(main(), debug=True)` or `PYTHONASYNCIODEBUG=1` during development.
    It logs callbacks over 100ms, which is exactly the "who blocked the loop" question.

## Choosing what to make concurrent

Rank by this, not by what looks slow:

1. **Independent network calls in a loop.** The single highest-value conversion. N sequential
   round trips become one round trip's worth of wall clock, bounded by your semaphore.
2. **Fan-out with a slow tail.** 100 calls where 3 are slow: concurrency hides the tail entirely.
3. **A server's request path.** Every blocking call removed from a handler raises the ceiling on
   concurrent requests for the whole process.
4. **Pipelines with independent stages.** Download while decoding, decode while uploading: a
   bounded `asyncio.Queue` between stages.

Not worth it: a single request (concurrency of one is just overhead), CPU-bound loops (use
processes), anything where the sequential version already meets the requirement, and code whose
next step depends on the previous result (that dependency is real and no amount of `await`
removes it).

## Reference files

Read as needed rather than all at once.

| File | When to read |
|---|---|
| `references/rules.md` | The full rule set with rationale and a primary source per rule. Read when justifying a rule or resolving a conflict. |
| `references/asyncio-core.md` | Event loop mechanics, coroutine vs task, `gather` vs `TaskGroup` with real measured output, cancellation, timeouts, threads and processes, what changed in 3.12–3.14. |
| `references/httpx.md` | Clients and pooling, the four timeouts, exception hierarchy, streaming, retries, HTTP/2, testing with `ASGITransport`, and the ecosystem's current state. |
| `references/structured-concurrency.md` | anyio task groups, cancel scopes, level vs edge cancellation, memory object streams, and how anyio differs from `asyncio.TaskGroup`. |
| `references/services-and-pipelines.md` | FastAPI handlers and lifespan, threadpool sizing, batching LLM/API calls with rate limits, and mixed IO+CPU pipelines. |
| `references/debugging.md` | Deciding IO- vs CPU-bound with a profiler, asyncio debug mode, `python -m asyncio ps/pstree`, finding hangs, the runtime blocking detectors, and a symptom-to-cause table. |

## Assets

| File | Use |
|---|---|
| `assets/aioutil/client.py` | `build_client()`: an `AsyncClient` with every setting that matters stated explicitly, plus two named timeout profiles. |
| `assets/aioutil/bounded.py` | `bounded_map` (all-or-nothing), `bounded_map_settled` (per-item verdicts), `cpu_map` (process pool). The concurrency limit is a required part of the API. |
| `assets/aioutil/retry.py` | `request_with_retry` / `with_retry`: full-jitter backoff, `Retry-After` (capped at `max_delay`), and retryable sets that narrow by HTTP method for both exceptions and 5xx statuses. |
| `assets/examples/good_pipeline.py` | Bounded concurrent download → parallel CPU stage. The reference shape for a data pipeline. |
| `assets/examples/fastapi_service.py` | Lifespan-shared client, bounded fan-out handler, and the three ways to handle blocking work in a handler. |
| `assets/examples/bad_example.py` | Every mistake the audit catches, one per numbered comment. The audit's fixture. |
| `assets/ruff-async.toml` | Ruff config enabling the ASYNC ruleset, with each rule code annotated. |
| `assets/ci.yml` | GitHub Actions workflow: audit, then lint, then tests. |

## Scripts

`scripts/audit_async.py` is a stdlib-only AST scan. Exit 0 means clean, 1 means findings, 2 means
a target path does not exist, so it drops straight into CI or a pre-commit hook.

```bash
python scripts/audit_async.py src/
python scripts/audit_async.py src/ --format json
python scripts/audit_async.py src/ --ignore SEQUENTIAL_AWAIT
python scripts/audit_async.py --list-codes
```

`scripts/test_audit_async.py` (stdlib) and `scripts/test_aioutil.py` (needs httpx) are the
regression tests for the script and the helpers. Run them after changing either.

Two checks are heuristic and may need ignoring in specific files: `SEQUENTIAL_AWAIT` (an `await`
in a `for` loop is correct when the iterations genuinely depend on each other) and
`UNBOUNDED_FANOUT` (quiet if any `Semaphore` or `CapacityLimiter` appears in the module).

## Sources

Each rule's primary source is cited next to it in `references/rules.md`; each reference file ends
with its own source list.
