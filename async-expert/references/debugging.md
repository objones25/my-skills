# Debugging async code

All output below is real, from Python 3.14.

## First: is this even I/O-bound?

Everything else in this skill is wasted if the answer is no. Profile before converting.

**scalene** splits time into Python, native, and system, which is exactly the I/O-vs-CPU
distinction you need and which `cProfile` does not give you:

```bash
uv add --dev scalene
scalene run --cpu-only your_script.py
scalene view --cli
```

That is the scalene 2.x interface (2.0.0 shipped 2025-12-20; verified on 2.3.0): `run` writes
`scalene-profile.json`, `view` renders it in the browser or, with `--cli`, the terminal. The 1.x
form `scalene --html --outfile profile.html script.py` now just prints the usage text and profiles
nothing.

Read the result this way:

| Column dominant in a function | Meaning | Tool |
|---|---|---|
| **System time** | Waiting on syscalls: network, disk | asyncio (or threads if the library is sync-only) |
| **Python time** | Executing bytecode | `ProcessPoolExecutor` |
| **Native time** | Inside a C extension | Depends: measure whether it releases the GIL |

**py-spy** is the tool for a process that is already running and already misbehaving, with no code
changes, no restart:

```bash
py-spy dump --pid 12345          # what every thread is doing right now
py-spy top --pid 12345           # live sampling
```

`py-spy dump` on a hung service usually ends the investigation in one command: you see the exact
blocking call sitting on the event-loop thread.

Naming conventions are a weak signal but a free one: `fetch`, `get`, `download`, `query`, `load`
lean I/O-bound; `compute`, `process`, `encode`, `transform`, `render` lean CPU-bound. Confirm with
the profiler; do not decide with the name.

## Debug mode: who blocked the loop

```python
asyncio.run(main(), debug=True)
```

or `PYTHONASYNCIODEBUG=1`, or `python -X dev`. What it turns on:

- Callbacks that run longer than `loop.slow_callback_duration` (**default 0.1 seconds**) are logged.
- Non-threadsafe APIs called from the wrong thread raise instead of corrupting state.
- Un-awaited coroutines and never-retrieved task exceptions get better tracebacks.

The slow-callback log is the direct answer to "something is blocking the loop, what":

```
Executing <Task finished name='Task-1' coro=<main() done, defined at /tmp/slowcb.py:4>
result=None created at .../runners.py:109> took 0.400 seconds
```

That was a `time.sleep(0.4)` inside an `async def`. Lower the threshold when hunting smaller
stalls:

```python
loop = asyncio.get_running_loop()
loop.slow_callback_duration = 0.02      # 20ms
```

Debug mode costs real performance. Development and staging, not production.

## Finding a hang: the await graph

New in 3.14 and the best tool for "it's stuck and I don't know where." From inside the process:

```python
import asyncio
asyncio.print_call_graph()          # current task
asyncio.print_call_graph(some_task) # a specific task
asyncio.capture_call_graph()        # same data, as an object
```

Real output, three nested coroutines inside a `TaskGroup`:

```
* Task(name='worker-1', id=0x7f2d11870230)
  + Call stack:
  |   File '.../asyncio/graph.py', line 276, in print_call_graph()
  |   File '/tmp/cg.py', line 3, in async leaf()
  |   File '/tmp/cg.py', line 6, in async middle()
  + Awaited by:
    * Task(name='Task-1', id=0x7f2d11870050)
      + Call stack:
      |   File '.../asyncio/taskgroups.py', line 121, in async TaskGroup._aexit()
      |   File '/tmp/cg.py', line 8, in async main()
```

The `Awaited by:` section is what a normal traceback cannot show you: which task is blocked on
which. That is the shape of a deadlock.

There is also an out-of-process CLI:

```bash
python -m asyncio ps      <PID>     # flat table of tasks and their awaiters
python -m asyncio pstree  <PID>     # the await tree; errors out on cycles
```

Real output against a hung process on uv's CPython 3.14.6 (also works on 3.14.0):

```
tid   task id         task name  coroutine stack                                  awaiter chain                                    awaiter name
1526  0x7f15b14a4410  Task-1     TaskGroup._aexit -> TaskGroup.__aexit__ -> main
1526  0x7f15b14a45f0  worker-1   sleep -> leaf -> middle                          TaskGroup._aexit -> TaskGroup.__aexit__ -> main  Task-1
```

It reads the target's memory from outside, so it needs the same permission a debugger needs
(same user, or root, and a ptrace policy that allows it; in containers that can mean
`--cap-add SYS_PTRACE`). Test it against your deployment before an incident.

The in-process fallback is a signal handler that dumps every task. It has to iterate
`asyncio.all_tasks()`: a plain `signal.signal(SIGUSR1, lambda *_: asyncio.print_call_graph())`
prints nothing when the loop is idle waiting on I/O (the usual state of a hung service), because
there is no current task at that moment. Measured: the plain handler printed one empty line;
this one printed both tasks with their awaiter chain.

```python
import asyncio, signal

def install_task_dump(sig=signal.SIGUSR1):
    loop = asyncio.get_running_loop()
    def dump(*_):
        for task in asyncio.all_tasks(loop):
            asyncio.print_call_graph(task)
    signal.signal(sig, dump)
```

Call `install_task_dump()` once at the top of `main()`, then `kill -USR1 <pid>`. Use
`signal.signal`, not `loop.add_signal_handler`: the loop's handler only runs when the loop gets
control back, so if a sync call has blocked the loop it prints nothing until the block ends.
Measured on 3.14.6 with the loop stuck in `time.sleep(2)`: `signal.signal` dumped the tasks during
the block, `add_signal_handler` only after it. If the loop thread is stuck in C code, add
`faulthandler.register(signal.SIGUSR2, all_threads=True)` for thread stacks.

Always pass `name=` to `create_task`. Every one of these tools prints task names, and
`Task-47` tells you nothing.

## Static detection

Two tools, deliberately overlapping, neither a superset of the other.

```bash
ruff check --config assets/ruff-async.toml src/
python scripts/audit_async.py src/
```

`ruff`'s ASYNC ruleset (its port of `flake8-async`) knows blocking library calls:

| Code | Catches |
|---|---|
| ASYNC100 | timeout context with no `await` inside: the timeout can never fire |
| ASYNC110 | `while flag: await sleep()` busy-wait instead of an `Event` |
| ASYNC119 | `yield` inside a context manager in an async generator: cleanup can be skipped (preview-only: needs `--preview`, otherwise selecting it does nothing) |
| ASYNC210 | blocking HTTP call (urllib/requests) inside `async def` |
| ASYNC212 | sync `httpx` call inside `async def` |
| ASYNC221 | `subprocess.run` / `os.system` inside `async def` |
| ASYNC230 | builtin `open()` inside `async def` |
| ASYNC240 | blocking `pathlib` methods inside `async def` |
| ASYNC251 | `time.sleep()` inside `async def` |
| RUF006 | `create_task` result not stored |

`audit_async.py` knows the shape mistakes ruff does not model: unbounded fan-out, `gather` with no
failure policy, swallowed cancellation, a client rebuilt per iteration, sequential awaits in a
loop. Run `--list-codes` for the full set.

## Runtime detection

`blockbuster` patches common blocking stdlib calls so that invoking one from inside a running
event loop raises `BlockingError` instead of silently stalling. It catches what static analysis
cannot: a blocking call three libraries deep.

```python
import pytest
from blockbuster import blockbuster_ctx

@pytest.fixture(autouse=True)
def blockbuster():
    with blockbuster_ctx() as bb:
        yield bb
```

Best used as an autouse fixture in the test suite rather than in production.

## The un-awaited coroutine

`RuntimeWarning: coroutine 'fetch' was never awaited` fires at garbage-collection time, far from
the call site and easy to lose in a log. Making it an error with `python -W error::RuntimeWarning
app.py` does not help: the warning is raised inside the coroutine's finalizer, so CPython prints
"Exception ignored while finalizing coroutine" and the program continues and exits 0 (measured on
3.11 through 3.15). Use one of these instead:

```bash
mypy src/
pytest -W error::RuntimeWarning -W error::pytest.PytestUnraisableExceptionWarning
python -X dev app.py
```

mypy reports it statically as `unused-coroutine`. The pytest pair turns it into a failing test.
`-X dev` keeps it a warning but adds a "Coroutine created at" traceback pointing at the call.

## Reading the common failure signatures

| Symptom | Usual cause |
|---|---|
| N operations of 1s each take N seconds | Nothing was scheduled concurrently. Coroutines awaited directly in a loop instead of created as tasks. |
| Ctrl-C does nothing; process hangs on shutdown | Something swallowed `CancelledError`, or `shield` wraps the main work. |
| `PoolTimeout` under load | Semaphore limit exceeds `max_connections`. |
| Works with 1 user, collapses at 40 | anyio's 40-token limiter, saturated by `def` handlers or sync dependencies. |
| Collapses at a handful of concurrent requests on a small container | `asyncio.to_thread` calls queued on the loop's default executor, `min(32, cpus + 4)` threads. |
| `Task exception was never retrieved` at exit | A fire-and-forget task failed. Nobody awaited it. |
| Task silently stops partway with no error | `FLOATING_TASK`: garbage collected mid-flight. |
| `BrokenProcessPool` immediately | Missing `if __name__ == "__main__":`, or an unpicklable function. |
| Adding async made it slower | The work is CPU-bound. Profile; move it to processes. |
| Timeouts fire but the work keeps going | Something caught `TimeoutError`/`CancelledError` and continued. |

## Sources

- Python docs, *Developing with asyncio*: https://docs.python.org/3/library/asyncio-dev.html
- Python docs, `loop.slow_callback_duration`: https://docs.python.org/3/library/asyncio-eventloop.html
- Python docs, *What's New in 3.14* (call graph, `python -m asyncio ps`): https://docs.python.org/3/whatsnew/3.14.html
- Ruff, flake8-async rules: https://docs.astral.sh/ruff/rules/
- scalene (2.x `run`/`view` CLI): https://github.com/plasma-umass/scalene
- py-spy: https://github.com/benfred/py-spy
- blockbuster: https://github.com/cbornet/blockbuster
- pytest, unraisable exceptions and `-W`: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
- mypy, `unused-coroutine`: https://mypy.readthedocs.io/en/stable/error_code_list.html
