# asyncio core

Everything in this file was run on Python 3.14. Output blocks are real program output,
not illustrations.

## The model in one paragraph

An event loop is a `while` loop holding a queue of ready tasks. It runs one task until that task
hits an `await` that cannot complete immediately; the task suspends and returns control to the
loop, which picks the next ready task. When the thing a suspended task was waiting on completes
(a socket becomes readable, a timer fires), the loop marks that task ready again and eventually
resumes it from exactly where it left off. All of this is one thread. Nothing runs in parallel.
The speedup comes entirely from not sitting idle during waits.

The consequence follows directly: **any code that does not `await` runs to completion before
anything else gets a turn.** A `time.sleep(1)` inside a coroutine is not "one task sleeping";
it is the entire program frozen for a second.

## Coroutine function, coroutine object, task

Three different things, and conflating them is the most common beginner bug.

```python
async def fetch(n):          # coroutine FUNCTION
    await asyncio.sleep(1)
    return n

coro = fetch(1)              # coroutine OBJECT — no code has run yet
task = asyncio.create_task(coro)   # TASK — now the loop is scheduling it
result = await task          # 1
```

`fetch(1)` on its own runs nothing. It is inert. The audit calls this `UNAWAITED_COROUTINE`, and
CPython emits `RuntimeWarning: coroutine 'fetch' was never awaited` at garbage-collection time,
long after the fact, and easy to miss in a busy log. `-W error::RuntimeWarning` does not fix that
for a script: the warning is raised inside a finalizer, so CPython prints "Exception ignored" and
the program carries on and exits 0. Catch it with mypy (`unused-coroutine`), with pytest run as
`-W error::RuntimeWarning -W error::pytest.PytestUnraisableExceptionWarning`, or by reading the
`-X dev` output, which also shows where the coroutine was created.

`await coro` both schedules and runs to completion. So a loop of `await fetch(u)` is serial:

```python
for url in urls:            # SEQUENTIAL_AWAIT
    results.append(await fetch(url))   # N round trips, one at a time
```

Concurrency starts when several tasks exist on the loop *before* you await any of them. That is
what `create_task`, `TaskGroup.create_task`, and `gather` all do.

## gather vs TaskGroup: measured

The three failure policies differ in what happens to the siblings. Here is the actual output
(`demo.py`, Python 3.14) with a task that fails at 0.1s and one that finishes at 0.5s:

```
gather(return_exceptions=False)
    caught ValueError: fast-fail exploded at 0.10s
    slow: finished at 0.50s                       <-- sibling NOT cancelled, still running

gather(return_exceptions=True)
    slow: finished at 0.50s
    results = [ValueError('fast-fail exploded'), 'slow']    <-- exceptions become values

TaskGroup
    slow: CANCELLED at 0.10s                      <-- sibling cancelled immediately
    caught ExceptionGroup: (ValueError('fast-fail exploded'),) at 0.10s
```

Read the first block carefully. `gather` with the default `return_exceptions=False` propagated the
error at 0.10s **and left the other task running**. The docs say so explicitly: *"Other
awaitables in the aws sequence won't be cancelled and will continue to run."* If your enclosing
function returns at that point, that orphan keeps holding a connection with nobody to receive its
result or its exception.

| | siblings on failure | what is raised | when |
|---|---|---|---|
| `TaskGroup` | cancelled | `ExceptionGroup` (`except*`) | at first failure |
| `gather(return_exceptions=False)` | **keep running** | the first exception, bare | at first failure |
| `gather(return_exceptions=True)` | run to completion | nothing | when all finish |

Default to `TaskGroup`. Reach for `gather(return_exceptions=True)` when partial results have
value: scraping where some URLs are dead, enriching a dataset. Never use bare `gather()`.

Catching an `ExceptionGroup` uses `except*`, which runs its block once per matching exception type
and lets non-matching ones propagate:

```python
try:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(fetch(1))
        tg.create_task(fetch(2))
except* httpx.HTTPError as eg:
    for exc in eg.exceptions:
        log.warning("http failure: %r", exc)
except* ValueError as eg:
    ...
```

## The disappearing task

From the `asyncio.create_task` docs, verbatim:

> **Important**: Save a reference to the result of this function, to avoid a task disappearing
> mid-execution. The event loop only keeps weak references to tasks. A task that isn't referenced
> elsewhere may get garbage collected at any time, even before it's done.

So this is a real bug, not a style preference:

```python
asyncio.create_task(send_metrics())   # FLOATING_TASK / RUF006
```

The two fixes, in order of preference:

```python
# 1. TaskGroup — holds strong references, awaits them, propagates their exceptions
async with asyncio.TaskGroup() as tg:
    tg.create_task(send_metrics())

# 2. A module-level set, when the task genuinely must outlive the block
_background: set[asyncio.Task] = set()

def spawn(coro):
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)
    return task
```

Option 2 still never awaits the task, so a failure inside it produces only a
`Task exception was never retrieved` message at GC time. Add an explicit done-callback that logs,
or prefer option 1.

## Cancellation

`CancelledError` has inherited from `BaseException` since 3.8. That is deliberate: it means a
routine `except Exception:` does not eat it. What does eat it:

```python
try:
    await work()
except asyncio.CancelledError:     # SWALLOWED_CANCEL
    log.info("stopping")           # <-- no re-raise
```

A task that swallows cancellation cannot be cancelled. Every `asyncio.timeout` above it becomes a
hang, and Ctrl-C stops working. The correct shape always re-raises:

```python
try:
    await work()
except asyncio.CancelledError:
    await flush_buffer()           # cleanup
    raise                          # then let it propagate
```

If the cleanup itself awaits and might be cancelled, wrap only the cleanup in `asyncio.shield`.
Shielding the main work instead is how "Ctrl-C does nothing" happens.

`Task.cancelling()` and `Task.uncancel()` (3.11) exist mainly so `TaskGroup` can distinguish its
own cancellation from an outer one. You rarely call them directly.

## Timeouts

`asyncio.timeout()` (3.11) is a context manager, so it can bound a whole block including several
awaits, cleanup, and retries:

```python
async with asyncio.timeout(5.0):
    conn = await pool.acquire()
    rows = await conn.fetch(query)
```

On expiry it cancels everything inside the block and raises `TimeoutError`. Since 3.12,
`asyncio.wait_for` is implemented on top of it; `wait_for` remains fine for a single awaitable.

Two things to get right:

- **Timeouts compose by nesting, and the innermost deadline does not win; the earliest does.**
  A 30s outer timeout around a 60s inner one means the inner one never fires.
- **A timeout is only useful if it is shorter than the caller's patience.** A 60s client timeout
  inside a request handler whose caller gives up at 5s protects nothing.

`asyncio.timeout_at(loop.time() + n)` sets an absolute deadline, which is what you want when
propagating one deadline through several calls rather than restarting the clock at each hop.

## Bounding concurrency

`asyncio.Semaphore(n)` is the tool. The pattern that keeps the limit and the fan-out together:

```python
sem = asyncio.Semaphore(10)

async def guarded(item):
    async with sem:
        return await work(item)

async with asyncio.TaskGroup() as tg:
    tasks = [tg.create_task(guarded(i)) for i in items]
results = [t.result() for t in tasks]
```

Note the `.result()` read happens *after* the `async with` block; inside it, the tasks are not
finished yet. `assets/aioutil/bounded.py` packages this with results in input order.

For producer/consumer, `asyncio.Queue(maxsize=n)` gives backpressure: `await queue.put(x)`
suspends the producer once the queue is full, which is the whole point. An unbounded queue with a
slow consumer is a memory leak that looks like a working pipeline. `Queue.shutdown()` (3.13)
gives a clean way to stop consumers without sentinel values.

## Threads and processes

```python
value = await asyncio.to_thread(blocking_fn, arg)          # I/O-bound blocking library
```

`to_thread` (3.9) propagates the current `contextvars.Context` and is documented as *"primarily
intended to be used for executing IO-bound functions."* Under the GIL it does nothing for
CPU-bound Python, because the threads take turns on the same core. It *does* help for extension code
that releases the GIL (NumPy, PyTorch, `hashlib` on large inputs), which is worth measuring
rather than assuming.

CPU-bound work needs processes:

```python
loop = asyncio.get_running_loop()
with ProcessPoolExecutor() as pool:
    results = await asyncio.gather(
        *[loop.run_in_executor(pool, cpu_fn, x) for x in items],
        return_exceptions=False,
    )
```

That shape has one trap: if an item fails (or the caller times out), leaving the `with` block calls
`pool.shutdown(wait=True)` on the loop thread, which blocks every task until the remaining items
finish. Measured: one item failing at once, two others taking 2s, froze the loop for 2s; an
`asyncio.timeout(0.2)` around it returned after 2.17s. `assets/aioutil/bounded.py:cpu_map` shuts the
pool down with `wait=False, cancel_futures=True` on failure instead.

Two constraints the docs call out, both of which bite in practice:

- The function must be picklable, so it must be defined at module level, not nested or a lambda.
- The script needs `if __name__ == "__main__":`. Workers re-import the module; without the guard
  they re-run your top-level code. On 3.14 Linux (default start method now `forkserver`) and on
  macOS/Windows (`spawn`) that fails with `RuntimeError: An attempt has been made to start a new
  process before the current process has finished its bootstrapping phase`; run from stdin it is
  `FileNotFoundError: ... '<stdin>'`. Either way the parent then sees `BrokenProcessPool` or a
  `ConnectionResetError`. On 3.13 and earlier Linux (`fork`) the unguarded script happens to work.

Note `asyncio.get_running_loop()`, not `get_event_loop()`. The former is correct inside a
coroutine; the latter raises `RuntimeError` on 3.14 when no loop is running.

## What changed, by version

| Version | Change that affects how you write code |
|---|---|
| 3.8 | `CancelledError` becomes a `BaseException` subclass. |
| 3.9 | `asyncio.to_thread()` added. |
| 3.11 | `TaskGroup`, `asyncio.timeout()`/`timeout_at()`, `Runner`, `Barrier`, `Task.uncancel()`. `asyncio.wait()` no longer accepts coroutines. `@asyncio.coroutine` removed. |
| 3.12 | Eager task factory (`asyncio.eager_task_factory`), which runs a coroutine synchronously until it first blocks, skipping the scheduling round trip for tasks that finish immediately. `loop_factory` on `asyncio.run()`. |
| 3.13 | `Queue.shutdown()` + `QueueShutDown`. `as_completed()` usable as both a sync and async iterator. |
| 3.14 | **`get_event_loop()` raises `RuntimeError` when no loop is running.** Default multiprocessing start method on Linux is `forkserver`, not `fork`. Child watchers removed. `asyncio.capture_call_graph()` / `print_call_graph()` and the `python -m asyncio ps` / `pstree` CLI. First-class free-threading support: per-thread task lists, so multiple event loops in different threads scale roughly linearly. The whole policy system is deprecated for removal in 3.16. |

The eager task factory is worth knowing for LLM/API batching where many calls hit a cache and
return without ever awaiting, because it removes one scheduling hop per task:

```python
loop = asyncio.get_running_loop()
loop.set_task_factory(asyncio.eager_task_factory)
```

It changes execution order: the coroutine body starts running at `create_task` time rather than at
the next loop iteration. Measured on 3.14, appending a marker before and inside the task body:

```
execution order: ['after-create-lazy', 'body-lazy', 'body-eager', 'after-create-eager']
                  ^ default: body runs later    ^ eager: body runs before create_task returns
```

Turn it on deliberately and re-run your tests; code that relied on "nothing happens until I
await" changes behavior.

## Sources

- Coroutines and Tasks: https://docs.python.org/3/library/asyncio-task.html
- Developing with asyncio: https://docs.python.org/3/library/asyncio-dev.html
- Synchronization primitives: https://docs.python.org/3/library/asyncio-sync.html
- Queues: https://docs.python.org/3/library/asyncio-queue.html
- Event loop (`run_in_executor`, `slow_callback_duration`): https://docs.python.org/3/library/asyncio-eventloop.html
- Exceptions: https://docs.python.org/3/library/asyncio-exceptions.html
- What's New in 3.11: https://docs.python.org/3/whatsnew/3.11.html
- What's New in 3.12: https://docs.python.org/3/whatsnew/3.12.html
- What's New in 3.13: https://docs.python.org/3/whatsnew/3.13.html
- What's New in 3.14: https://docs.python.org/3/whatsnew/3.14.html
- Kumar Aditya, *Scaling asyncio on Free-Threaded Python*, Quansight Labs: https://labs.quansight.org/blog/scaling-asyncio-on-free-threaded-python
