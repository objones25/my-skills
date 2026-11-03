# 04: "The async suite hangs in CI until the job times out after six hours"

*Illustrative system.* An asyncio service tested with pytest-asyncio 1.4.1, strict mode.
One test waits for an event that the code under test was supposed to set.

## Verdict

A missing deadline turns a failed expectation into a hang. Put a deadline in the test,
add pytest-timeout as a suite-wide backstop, and then fix what the deadline exposes.

## Candidate causes

1. The awaited thing never completes (an event nobody sets, a queue nobody fills).
2. An object created on one event loop awaited from another (shared fixtures with
   mismatched `loop_scope`).
3. A blocking call (sync I/O, `time.sleep`) on the event loop.

Separate them by making the hang fail fast and reading the error.

## The evidence

With pytest-timeout as a backstop (`--timeout=2`), the hang becomes a failure:

```
E           Failed: Timeout (>2.0s) from pytest-timeout.
1 failed in 2.23s
```

The built-in `faulthandler_timeout` alone is not enough: it printed `Timeout
(0:00:01)!` and every thread's stack, then the run kept hanging until killed.

A second test in the same suite failed differently once it stopped hanging:

```
RuntimeError: Task <Task pending ...> got Future <Future pending> attached to a different loop
```

That is cause 2: a session-scoped fixture (`loop_scope="session"`) handed a future to a
test running on its own function-scoped loop.

## The fix

1. Deadlines inside tests around every wait: `async with asyncio.timeout(1.0):`
   (or `anyio.fail_after`).
2. `timeout = 120` with pytest-timeout installed, so no future hang costs six hours.
3. Tests that await the session fixture's objects get the same scope:
   `@pytest.mark.asyncio(loop_scope="session")`.
4. The event that was never set was the real bug; the deadline turned it into a
   readable failure.

All of these patterns are runnable in `assets/test_async_template.py`; the failure
table is in `async-testing.md`.
