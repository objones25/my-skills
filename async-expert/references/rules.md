# The rules, with rationale and sources

Numbering matches the "hard gates" section of `SKILL.md`. Each rule states what it prevents and
where the claim comes from. `[audit]` marks rules the audit script enforces; `[ruff]` marks ones
ruff catches; `[judgment]` marks ones no tool can check.

Contents: [The loop is one thread](#the-loop-is-one-thread) (1 to 5) ·
[Tasks and fan-out](#tasks-and-fan-out) (6 to 10) · [Bounds](#bounds) (11 to 14) ·
[Timeouts and cancellation](#timeouts-and-cancellation) (15 to 20) · [HTTP](#http) (21 to 28) ·
[Services](#services) (29 to 31) · [Entry point](#entry-point) (32 to 33)

---

## The loop is one thread

### 1. No blocking call inside `async def` `[audit: BLOCKING_*]` `[ruff: ASYNC2xx]`

The event loop runs on one thread. A synchronous call that waits does not suspend one task, it
freezes the scheduler: every other task in the process stops until it returns. This is the
difference between async code that works and async code that merely compiles.

The common offenders: `time.sleep`, `requests.*`, sync `httpx.*`, `open()`, `subprocess.run`,
`os.system`, `sqlite3.connect`, `psycopg2.connect`, `input()`, `pathlib.Path.read_text`.

Measured cost, from `asyncio.run(main(), debug=True)` on a `time.sleep(0.4)` in a coroutine:
`took 0.400 seconds`. That is 400ms of dead air for every concurrent request, not just the one.

Source: https://docs.python.org/3/library/asyncio-dev.html · https://docs.astral.sh/ruff/rules/

### 2. Blocking I/O goes to a thread

`await asyncio.to_thread(fn, *args)`. Added in 3.9; propagates the current `contextvars.Context`;
documented as *"primarily intended to be used for executing IO-bound functions."*

The docs are explicit about the limit: *"Due to the GIL, `asyncio.to_thread()` can typically only
be used to make IO-bound functions non-blocking. However, for extension modules that release the
GIL … it can also be used for CPU-bound functions."* So NumPy and PyTorch operations are worth
measuring in a thread; pure-Python loops are not.

Source: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread

### 3. CPU-bound work goes to processes

`loop.run_in_executor(ProcessPoolExecutor(), fn, x)`. Two documented constraints:

- The callable must be picklable: module-level, not nested, not a lambda.
- The script needs `if __name__ == "__main__":`. The docs: *"the entry point guard … is required …
  due to the peculiarities of multiprocessing, which is used by ProcessPoolExecutor."* Without it,
  workers re-import and re-execute your top-level code. On 3.14 Linux (where the default start
  method changed from `fork` to `forkserver`) and on macOS/Windows (`spawn`), the run fails with
  `RuntimeError: An attempt has been made to start a new process before the current process has
  finished its bootstrapping phase`, followed by `BrokenProcessPool` or a `ConnectionResetError`.
  On 3.13 and earlier Linux (`fork`) the same script happens to work, which is how the bug ships.

Source: https://docs.python.org/3/library/asyncio-eventloop.html

### 4. No `threading` primitives inside `async def` `[audit: BLOCKING_LOCK]`

`threading.Lock.acquire()` parks the OS thread, which is the event loop's thread. Use
`asyncio.Lock`, `asyncio.Event`, `asyncio.Semaphore`, whose `await` suspends the task instead.

Source: https://docs.python.org/3/library/asyncio-sync.html

### 5. Nothing between two awaits should exceed ~100ms `[judgment]`

Not a hard number so much as the one the runtime already picked: asyncio debug mode logs
callbacks exceeding `loop.slow_callback_duration`, *"Default value is 100 milliseconds."* Adopting
it as your own threshold means the tooling agrees with your rule for free.

Source: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.slow_callback_duration

---

## Tasks and fan-out

### 6. `TaskGroup` is the default `[judgment]`

Python 3.11's own release notes: *"For new code this is recommended over using `create_task()` and
`gather()` directly."* It holds strong references to its children, cancels siblings on failure,
awaits everything before the block exits, and propagates errors as an `ExceptionGroup`. Rules 8,
9, and 20 are all things `TaskGroup` gets right by construction.

Source: https://docs.python.org/3/whatsnew/3.11.html

### 7. `gather` always states `return_exceptions=` `[audit: BARE_GATHER]`

The default is `False`, and the default's behavior is the surprising one: *"the first raised
exception is immediately propagated to the task that awaits on `gather()`. Other awaitables in the
aws sequence **won't be cancelled** and will continue to run."*

Measured (`references/asyncio-core.md`): the failing task raises at 0.10s, the sibling keeps
running and completes at 0.50s, unobserved. Writing the keyword explicitly forces the author to
notice which of the two behaviors they are asking for.

Source: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather

### 8. Never discard a `create_task` result `[audit: FLOATING_TASK]` `[ruff: RUF006]`

Verbatim from the docs:

> Save a reference to the result of this function, to avoid a task disappearing mid-execution. The
> event loop only keeps weak references to tasks. A task that isn't referenced elsewhere may get
> garbage collected at any time, even before it's done.

The docs also name the fix: *"use `asyncio.TaskGroup` which keeps a strong reference to each task,
awaits them and propagates their exceptions."*

Source: https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task

### 9. A coroutine call without `await` runs nothing `[audit: UNAWAITED_COROUTINE]`

`fetch(url)` returns an inert coroutine object. CPython warns at garbage-collection time
(`RuntimeWarning: coroutine 'fetch' was never awaited`), which is far from the mistake.

Promoting that warning to an error does not stop the program. The warning is raised inside the
coroutine's finalizer, so `python -W error::RuntimeWarning app.py` prints "Exception ignored while
finalizing coroutine" and still exits 0 (measured on 3.11 to 3.15). What does catch it:

- mypy's `unused-coroutine` error code, at type-check time.
- pytest with `-W error::RuntimeWarning -W error::pytest.PytestUnraisableExceptionWarning`: the
  unraisable hook turns it into a failed test (measured on pytest 9.1.1).
- `python -X dev` (or `PYTHONASYNCIODEBUG=1`), which adds where the coroutine was created.

Source: https://docs.python.org/3/library/asyncio-dev.html · https://docs.pytest.org/en/stable/how-to/capture-warnings.html

### 10. `asyncio.wait()` takes tasks, not coroutines `[audit: WAIT_ON_COROUTINES]`

*"Changed in version 3.11: Passing coroutine objects to `wait()` directly is forbidden."* Wrap
them in tasks first, or use `TaskGroup`/`as_completed`.

Source: https://docs.python.org/3/library/asyncio-task.html#asyncio.wait

---

## Bounds

### 11. Every fan-out has a numeric limit `[audit: UNBOUNDED_FANOUT]`

`gather(*[fetch(u) for u in urls])` over 5,000 URLs attempts 5,000 concurrent connections. What
actually happens: the connection pool queues most of them, the remote rate-limiter rejects a
batch, or you hit the file-descriptor ceiling. The result is a wall of timeouts that reads like the
remote host being down, which sends the investigation in the wrong direction.

No stdlib documentation prescribes a limiting primitive for this: `asyncio.Semaphore` and a
bounded `asyncio.Queue` are both conventional, and the choice is left to the author. Concurrency
cap → `Semaphore`. Requests-per-unit-time → a token-bucket limiter such as `aiolimiter`; a
semaphore cannot express "60 per minute."

Source: https://docs.python.org/3/library/asyncio-sync.html · `assets/aioutil/bounded.py`

### 12. The limit comes from the binding constraint `[judgment]`

Provider rate limit, connection pool size, or file descriptors: whichever is smallest. Write
which one it was in a comment, because the number looks arbitrary six months later and someone
will "optimize" it upward.

### 13. Semaphore limit ≤ client `max_connections` `[judgment]`

httpx defaults to `max_connections=100`. A semaphore above that moves the queueing inside httpx,
where waiting past the `pool` timeout raises `PoolTimeout`: the same wait, now reported as a
failure.

Source: https://www.python-httpx.org/advanced/resource-limits/

### 14. Producer/consumer uses a bounded queue `[judgment]`

`asyncio.Queue(maxsize=n)`. `await queue.put(x)` suspending when full *is* the backpressure. An
unbounded queue with a slow consumer is a memory leak that presents as a working pipeline until
the process is OOM-killed. anyio's memory object streams default to `max_buffer_size=0`
(rendezvous), which is the stricter and better default. `Queue.shutdown()` (3.13) replaces
sentinel values for clean termination.

Source: https://docs.python.org/3/library/asyncio-queue.html · https://anyio.readthedocs.io/en/stable/streams.html

---

## Timeouts and cancellation

### 15. Every network await is inside a timeout `[judgment]`

`async with asyncio.timeout(5.0):` bounds a whole block. Since 3.12 `wait_for` is implemented on
top of `timeout()`; the context-manager form is preferred for new code because it composes with
`TaskGroup` and can wrap several awaits plus their cleanup.

Source: https://docs.python.org/3/library/asyncio-task.html#asyncio.timeout · https://docs.python.org/3/whatsnew/3.11.html

### 16. A timeout must be shorter than the caller's patience `[judgment]`

Nesting composes by earliest deadline, not innermost: a 30s outer timeout around a 60s inner one
means the inner one never fires. Measured: an `asyncio.timeout(0.1)` wrapping an
`asyncio.timeout(5.0)` fires at 0.10s. Use `timeout_at()` to propagate one absolute deadline
through several hops rather than restarting the clock at each.

### 17. `except CancelledError` must re-raise `[audit: SWALLOWED_CANCEL]`

Docs: *"This exception can be caught to perform custom operations when asyncio Tasks are
cancelled. In almost all situations the exception must be re-raised."* A task that swallows it
cannot be cancelled, so every timeout above it silently becomes a hang and Ctrl-C stops working.

Source: https://docs.python.org/3/library/asyncio-exceptions.html

### 18. `CancelledError` is a `BaseException` `[judgment]`

*"Changed in version 3.8: CancelledError is now a subclass of BaseException rather than
Exception."* This is why `except Exception:` is safe and bare `except:` / `except BaseException:`
are not. A settling wrapper that catches per-item failures must catch `Exception`, and re-raise
`CancelledError` explicitly if it catches anything broader.

Source: https://docs.python.org/3/library/asyncio-exceptions.html

### 19. `shield()` only for cleanup `[judgment]`

`asyncio.shield` protects an awaitable from the *caller's* cancellation. Shielding ordinary work
turns Ctrl-C and every enclosing timeout into a hang. Shield a buffer flush or a lock release, not
a request. Note the shielded object still needs a strong reference: the same GC warning as
`create_task` appears on `shield`.

Source: https://docs.python.org/3/library/asyncio-task.html#asyncio.shield

### 20. Know which failure policy you chose `[judgment]`

| | siblings on failure | raises | when |
|---|---|---|---|
| `TaskGroup` | cancelled | `ExceptionGroup` | at first failure |
| `gather(return_exceptions=False)` | keep running | first exception, bare | at first failure |
| `gather(return_exceptions=True)` | run to completion | nothing | when all finish |

`TaskGroup` docs: *"The first time any of the tasks belonging to the group fails with an exception
other than `asyncio.CancelledError`, the remaining tasks in the group are cancelled."*

Source: https://docs.python.org/3/library/asyncio-task.html#task-groups

---

## HTTP

*Rules 21 to 28 hold for both `httpx` 0.28.1 and `httpx2` 2.16.0 (the Pydantic fork). Diffing the two
installed packages shows an identical `AsyncClient` signature and identical defaults; httpx2 adds
SSE, WebSockets, and `alias_httpx()`. New projects should take httpx2: see `references/httpx.md`.
Sources: https://pypi.org/project/httpx2/ · https://github.com/pydantic/httpx2/blob/main/src/httpx2/CHANGELOG.md*

### 21. One client per process `[audit: CLIENT_IN_LOOP]`

httpx docs: *"When you make requests using the top-level API … HTTPX has to establish a new
connection for every single request."* versus a `Client`, which *"will reuse the underlying TCP
connection."* On HTTPS that is a saved TLS handshake per call.

Source: https://www.python-httpx.org/advanced/clients/

### 22. Set `timeout=` explicitly `[audit: NO_CLIENT_TIMEOUT]`

Read from the installed 0.28.1: `DEFAULT_TIMEOUT_CONFIG = Timeout(timeout=5.0)`, so connect, read,
write and pool are all 5 seconds. Too short for LLM generation, too long for a health check. `read` is
the gap between chunks, not total duration, which cuts both ways: streaming a huge file is fine,
but a server dribbling one byte per second never trips it.

Source: https://www.python-httpx.org/advanced/timeouts/

### 23. `follow_redirects` defaults to `False` `[judgment]`

Verified in the installed signature. `requests` follows redirects by default; ported code silently
starts returning 301s with empty bodies.

Source: https://www.python-httpx.org/quickstart/

### 24. Transport `retries=` covers connection setup only `[judgment]`

*"Requests will be retried the given number of times in case an `httpx.ConnectError` or an
`httpx.ConnectTimeout` occurs… If you need other forms of retry behaviors, such as handling
read/write errors or reacting to 503 Service Unavailable, consider general-purpose tools such as
tenacity."*

Source: https://www.python-httpx.org/advanced/transports/

### 25. Backoff is exponential with full jitter `[judgment]`

`sleep = random(0, min(cap, base * 2**attempt))`: Marc Brooker's formula. Its simulations show
jittered strategies substantially outperform plain exponential backoff; full jitter did the least
total work of the variants tested. Without jitter, clients that failed together retry together,
and the herd re-forms at every backoff level. Honor `Retry-After` when the server sends it: a 429
carrying `Retry-After: 30` is the server giving you the answer.

Source: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

### 26. The retryable set depends on the HTTP method `[judgment]`

This applies to status codes as much as to exceptions. A 502 or 503 after a POST does not mean the
handler did nothing (a gateway can time out after the upstream committed), so 500/502/503/504 are
retried for idempotent methods only. 408, 425 and 429 are the statuses where the server says it
did not act (RFC 9110 15.5.9, RFC 8470, RFC 6585), so they are safe to resend for any method.
urllib3's `Retry` makes the same call: its default `allowed_methods` excludes POST and PATCH.

Three tiers, and conflating them is how a retry policy double-charges a card:

| Tier | Exceptions | Safe because |
|---|---|---|
| Always retryable | `ConnectError`, `ConnectTimeout`, `PoolTimeout` | The request never reached the server. |
| Idempotent methods only | `ReadTimeout`, `RemoteProtocolError` | The request *was* sent. Only the response went missing: the server may have applied it. |
| Never retried blindly | `WriteTimeout`, `WriteError` | Fired mid-body; part of a non-idempotent request may already be applied. |

Idempotent methods are GET, HEAD, OPTIONS, TRACE, PUT and DELETE (RFC 9110 §9.2.2; PUT and DELETE
are idempotent though not safe). POST and PATCH are neither.

`assets/aioutil/retry.py` narrows both sets by method automatically. Measured:

```
GET  retryable set: ['ConnectError', 'ConnectTimeout', 'PoolTimeout', 'ReadTimeout', 'RemoteProtocolError']
POST retryable set: ['ConnectError', 'ConnectTimeout', 'PoolTimeout']
GET  -> retried 3 times then RetriesExhausted
POST -> ReadTimeout raised on first attempt, NOT retried
POST -> 502 returned on first attempt, NOT replayed; 429 retried
```

`RetryPolicy(retry_non_idempotent=True)` is the opt-out, for endpoints that take an idempotency
key or are genuinely no-ops when repeated.

Source: https://www.python-httpx.org/exceptions/ · https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2 · https://www.rfc-editor.org/rfc/rfc8470 · https://www.rfc-editor.org/rfc/rfc6585

### 27. Stream anything that could be large `[judgment]`

`async with client.stream(...)` plus `aiter_bytes()`. Touching `.content` on an unread streamed
response raises `ResponseNotRead`: *"Attempted to access streaming response content, without
having called read()."* The body is only available inside the `async with` block.

Source: https://www.python-httpx.org/quickstart/

### 28. Know what `raise_for_status()` raises `[judgment]`

`HTTPStatusError`, and only that. Transport failures are `RequestError` subclasses.
`httpx.HTTPError` is the common base. `StreamError` sits outside `HTTPError` entirely: those are
API-misuse errors, not network failures.

Source: https://www.python-httpx.org/exceptions/

---

## Services

### 29. Blocking work never runs directly in an `async def` handler `[ruff: ASYNC2xx]`

FastAPI: *"When you declare a path operation function with normal `def` instead of `async def`, it
is run in an external threadpool that is then awaited, instead of being called directly (as it
would block the server)."* And: *"If you just don't know, use normal `def`."*

Note FastAPI's docs state the guarantee for `def` rather than the consequence for `async def`. The
severity (one blocking call stalls every in-flight request) follows from the single-threaded
loop (rule 1), not from a sentence on that page.

Source: https://fastapi.tiangolo.com/async/

### 30. The shared client lives in `lifespan` `[judgment]`

*"The recommended way to handle the startup and shutdown is using the `lifespan` parameter… If you
provide a `lifespan` parameter, `startup` and `shutdown` event handlers will no longer be called.
It's all `lifespan` or all events, not both."*

Source: https://fastapi.tiangolo.com/advanced/events/ · https://starlette.dev/lifespan/

### 31. Two thread pools, sized separately `[judgment]`

Starlette: *"The default thread pool size is only 40 tokens. This means that only 40 threads can
run at the same time."* That is anyio's default `CapacityLimiter`, shared by plain `def` handlers,
sync dependencies, `fastapi.concurrency.run_in_threadpool`, and `anyio.to_thread.run_sync`.
Verified on anyio 4.15.1: `total_tokens == 40`. Raise it with
`anyio.to_thread.current_default_thread_limiter().total_tokens = N` (inside the running loop, for
example in `lifespan`; outside one it raises `NoEventLoopError`).

`asyncio.to_thread` is not part of that pool. It is `loop.run_in_executor(None, ...)`, which uses
the event loop's default `ThreadPoolExecutor`: `min(32, (os.process_cpu_count() or 1) + 4)`
workers. Measured with Starlette 1.9.0: 60 blocked `asyncio.to_thread` calls left
`borrowed_tokens == 0` on the anyio limiter, ran on threads named `asyncio_N`, and only 6 ran at
once on a 2-vCPU machine. Raising the anyio limiter does nothing for them; raise this pool with
`loop.set_default_executor(ThreadPoolExecutor(max_workers=N))`.

Size both against the database's own connection limit.

Source: https://starlette.dev/threadpool/ · https://anyio.readthedocs.io/en/stable/threads.html · https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread · https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor

---

## Entry point

### 32. `asyncio.run()`, never `get_event_loop()` `[audit: LEGACY_LOOP]`

On 3.14, `asyncio.get_event_loop()` raises `RuntimeError` when no loop is running instead of
creating one, and the whole policy system is deprecated for removal in 3.16. Child watchers were
removed in 3.14. Inside a coroutine, use `asyncio.get_running_loop()`.

Source: https://docs.python.org/3/whatsnew/3.14.html · https://docs.python.org/3/library/asyncio-eventloop.html

### 33. Develop with debug mode on `[judgment]`

`asyncio.run(main(), debug=True)` or `PYTHONASYNCIODEBUG=1`. Logs callbacks over 100ms, makes
non-threadsafe cross-thread API calls raise, and improves un-awaited-coroutine diagnostics. It
costs performance: development and staging only.

Source: https://docs.python.org/3/library/asyncio-dev.html
