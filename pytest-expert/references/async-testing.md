# Async testing

Async tests fail in ways sync tests cannot: a test that hangs forever, a coroutine that
was never awaited, an object created on one event loop and awaited on another, and
errors that arrive bundled in an `ExceptionGroup`. Each section shows the failure from a
real run and the fix. Patterns are runnable in `assets/test_async_template.py`.

Verified November 2026: pytest 9.2.0, pytest-asyncio 1.4.1, anyio 4.16.0, pytest-timeout
2.4.0, Python 3.13. Plugin options were read from `pytest --help`, signatures with
`inspect`, and pytest-asyncio behavior against its v1.4.0 docs (Context7).

## Contents

- [Terms](#terms)
- [Pick one plugin and configure it](#pick-one-plugin-and-configure-it)
- [Event loop scope](#event-loop-scope)
- [Failures you will meet](#failures-you-will-meet)
- [Time: inject it, never wait for it](#time-inject-it-never-wait-for-it)
- [Cancellation and timeouts](#cancellation-and-timeouts)
- [Doubles for async code](#doubles-for-async-code)
- [Sources](#sources)

## Terms

| Term | Meaning |
|---|---|
| **Event loop** | The scheduler that runs coroutines. A coroutine runs only when a loop drives it. |
| **Coroutine** | What calling an `async def` returns. Nothing executes until it is awaited. |
| **Loop scope** | How long one event loop lives in a test run: per test (`function`), per module, per session. |
| **Strict / auto mode** | pytest-asyncio settings. Strict runs only tests marked `@pytest.mark.asyncio`; auto claims every `async def` test. |
| **Task group** | `asyncio.TaskGroup` (or anyio's): runs child tasks and, if any fail, cancels the rest and raises an `ExceptionGroup`. |
| **Cancellation** | Asking a task to stop: `CancelledError` is raised at its current `await`, so its `finally` blocks run. |

## Pick one plugin and configure it

pytest itself does not run coroutines. With no plugin claiming a test, pytest 9.2 fails
it loudly:

```
async def functions are not natively supported.
You need to install a suitable plugin for your async framework, for example:
  - anyio
  - pytest-asyncio
  ...
FAILED tests/test_unmarked.py::test_forgot_the_mark - Failed: async def funct...
```

That is also what strict mode does to an `async def` test without the mark: it fails
instead of passing silently.

| Plugin | Use when | Marks |
|---|---|---|
| pytest-asyncio | the code is asyncio-only | `@pytest.mark.asyncio`, `@pytest_asyncio.fixture` |
| anyio's built-in plugin (ships with anyio, nothing to install) | the code uses anyio, or must run on both asyncio and trio | `@pytest.mark.anyio` plus an `anyio_backend` fixture |

Both can live in one suite. Keep pytest-asyncio in strict mode so it only claims tests
marked for it; the template runs both kinds side by side. anyio's docs say its own
`anyio_mode = "auto"` "does not work" when pytest-asyncio is also in auto mode, because
the two conflict; mark anyio tests with `@pytest.mark.anyio` instead.

pytest-asyncio 1.x needs explicit config. If `asyncio_default_fixture_loop_scope` is
unset it emits a `PytestDeprecationWarning` from `pytest_configure`, and with
`filterwarnings = ["error"]` that becomes an `INTERNALERROR` before any test runs
(reproduced with 1.4.1). Set all three keys, but only where the plugin is installed:
under `strict = true`, `strict_config` rejects them otherwise (`ERROR: Unknown config
option: asyncio_default_fixture_loop_scope`). For a shared config that must work
whether or not the plugin is installed (pytest-asyncio often arrives as a transitive
dependency), add a targeted filter instead; `assets/pytest-config.toml` and the wallet
example ship it, and both pass with and without the plugin (verified, pytest-asyncio
1.4.1):

```toml
filterwarnings = [
    "error",
    'ignore:The configuration option "asyncio_default_fixture_loop_scope" is unset:pytest.PytestDeprecationWarning',
]
```

```toml
asyncio_mode = "strict"
asyncio_default_fixture_loop_scope = "function"
asyncio_default_test_loop_scope = "function"
```

anyio's plugin has one ini option, `anyio_mode` (`"strict"` or `"auto"`, per `pytest
--help`). It picks up tests marked `@pytest.mark.anyio` or that request the
`anyio_backend` fixture. That fixture's default runs each test on every supported
backend; override it in a top-level `conftest.py` to choose:

```python
@pytest.fixture(params=["asyncio"])       # add "trio" when it is installed and supported
def anyio_backend(request):
    return request.param
```

## Event loop scope

By default every test and async fixture gets a fresh loop (`function`). That is the
right default: state cannot leak through the loop. Widen it only for an expensive async
resource, and then widen it for the fixture *and* the tests that await its objects.

A session-scoped async fixture on the default function loop fails at setup:

```
ScopeMismatch: You tried to access the function scoped fixture _function_scoped_runner
with a session scoped request object.
ERROR tests/test_scope.py::test_uses_pool - Failed: ScopeMismatch: You tried ...
```

The fix is a loop at least as wide as the fixture's cache scope (the docs: "the event
loop scope must be larger or the same as the fixture's caching scope"):

```python
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pool(): ...

@pytest.mark.asyncio(loop_scope="session")
async def test_query(pool): ...
```

The second half matters. A test left on its own function loop can receive the session
fixture's value, but awaiting anything bound to the session loop fails:

```
RuntimeError: Task <Task pending ...> got Future <Future pending> attached to a different loop
```

Database pools, HTTP clients, and locks created inside the fixture are all bound to the
loop that created them. Put every test that uses them on the same `loop_scope`, for a
whole module with `pytestmark = pytest.mark.asyncio(loop_scope="module")`.

## Failures you will meet

| Symptom | Cause | Fix |
|---|---|---|
| `async def functions are not natively supported` | no plugin claimed the test (unmarked in strict mode, or no plugin) | add the mark, or install and configure a plugin |
| `INTERNALERROR ... "asyncio_default_fixture_loop_scope" is unset` | pytest-asyncio 1.x plus `filterwarnings = error` | set the three keys above, or add the targeted filter |
| `RuntimeWarning: coroutine 'save' was never awaited`, reported as `PytestUnraisableExceptionWarning` | a missing `await`; the call created a coroutine and dropped it | add `await`. With warnings as errors the test fails, which is why that filter matters |
| `ScopeMismatch ... _function_scoped_runner` | wide fixture, narrow loop | `loop_scope` at least the fixture's `scope` |
| `got Future ... attached to a different loop` | an object from one loop awaited on another | same `loop_scope` on fixture and tests |
| the test never finishes | awaiting something nothing will complete | a deadline inside the test, plus pytest-timeout or `faulthandler_timeout` as a backstop |
| `ExceptionGroup` where you expected `ValueError` | a task group bundles child errors | `pytest.RaisesGroup` (rules.md, rule 33) |

The missing-`await` row is worth stressing: without `filterwarnings = ["error"]` the
test above *passes*, with the save never having happened.

## Time: inject it, never wait for it

`asyncio.sleep(2)` in a test costs two seconds per run and still races under load. The
audit flags `time.sleep` (`SLEEP`) and allows only `asyncio.sleep(0)`, which yields to
the loop without waiting. Make the code take its sleep (or clock) as a parameter:

```python
async def fetch_with_retry(fetch, *, attempts=3, base_delay=0.5, sleep=asyncio.sleep): ...
```

The test passes a fake that records delays:

```python
assert await fetch_with_retry(flaky_fetch, sleep=fake_sleep) == "ok"
assert fake_sleep.delays == [0.5, 1.0]
```

That pins the backoff schedule exactly, in microseconds. Monkeypatching
`asyncio.sleep` instead also stubs every other caller of it during the test, including
library code and your own deadline loops; the parameter is the narrower seam.

## Cancellation and timeouts

Two separate jobs:

1. **The code under test handles cancellation.** Cancel the task, await it inside
   `pytest.raises(asyncio.CancelledError)`, then assert the cleanup happened (the
   template's `test_cancelling_the_worker_runs_its_finally_block`). Swallowing
   `CancelledError` in production code is a bug this test catches: the `pytest.raises`
   fails.
2. **A hung test fails instead of hanging CI.** Put a deadline inside the test around
   any wait (`async with asyncio.timeout(1.0):`, or `anyio.fail_after(1.0)`, which
   raises `TimeoutError`), and keep a suite-wide backstop. pytest-timeout turned an
   `await event.wait()` that nothing set into `Failed: Timeout (>2.0s) from
   pytest-timeout.` in 2.2 s. Without the plugin, the built-in `faulthandler_timeout`
   (already in `assets/pytest-config.toml`) printed `Timeout (0:00:01)!` and every
   thread's stack, then the run kept hanging until it was killed: it diagnoses, it
   does not fail.

Waiting until something happens: loop on `await asyncio.sleep(0)` *inside* a deadline,
never on a fixed `sleep(0.1)`. The template's worker test does this; the loop needs an
`# audit: ignore[BRANCHING]` pragma with a reason because it is a wait, not a case loop.

## Doubles for async code

`mock.create_autospec(Client, instance=True)` makes `async def` methods into
`AsyncMock`s that keep the real signature: `client.send.assert_awaited_once_with("ada",
"hello")` checks the await, and calling `client.send("ada")` with a missing argument
raises `TypeError: missing a required argument: 'message'` (verified). A plain
`AsyncMock()` accepts any call, so the audit's `PATCH_NO_SPEC` rule applies here too.
Use `assert_awaited*`, not `assert_called*`: a coroutine that was created but never
awaited counts as called.

## Sources

- pytest-asyncio v1.4.0 docs (Context7, `/pytest-dev/pytest-asyncio/v1.4.0`): markers and `loop_scope`, decorators ("loop scope must be larger or the same as the fixture's caching scope"), modes.
- anyio testing docs (opened 2026-11-02): https://anyio.readthedocs.io/en/stable/testing.html : marker, default `anyio_backend` runs all backends, conflict with pytest-asyncio auto mode. `anyio_mode` values read from `pytest --help` with anyio 4.16.0.
- pytest 9.2.0 output for unclaimed async tests, read from a run.
- pytest-timeout 2.4.0 and Python 3.13 `asyncio.timeout`, `TaskGroup`: behavior read from runs, not docs.
