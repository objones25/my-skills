# Verified behavior

Every surprising claim in this skill, with the command that established it and the real
output. Re-run in October 2026 on **Python 3.11.17, 3.12.13, 3.13.16, 3.14.6** with
**wandb 0.31.0** (released 2026-10-16) and, for comparison, **wandb 0.28.2**. Where the two
wandb versions differ, the section says so. Reproduce any of it before trusting it in a
decision that matters.

Contents: [QueueHandler destroys structured exceptions](#1-queuehandler-destroys-structured-exceptions) ·
[Context filters behind a queue see nothing](#2-context-filters-behind-a-queue-see-nothing) ·
[Propagation duplicates output](#3-propagation-duplicates-output) ·
[W&B drops out-of-order steps](#4-wb-drops-out-of-order-steps-without-raising) ·
[W&B summary lags one commit](#5-wb-summary-lags-one-commit-when-you-pass-step) ·
[define_metric changes the summary's type](#6-define_metricsummarymin-changes-the-summarys-type) ·
[Config values are immutable](#7-config-values-are-immutable-new-keys-are-not) ·
[The global run breaks after finish](#8-the-global-run-breaks-after-finish) ·
[Offline runs cannot resume](#9-offline-runs-cannot-resume-locally) ·
[Offline artifacts and run.url](#10-offline-artifacts-and-runurl) ·
[What the context manager actually changes](#11-what-the-context-manager-actually-changes) ·
[No network: init raises instead of degrading](#12-no-network-init-raises-instead-of-degrading) ·
[ContextVars, threads, and tasks](#13-contextvars-threads-and-tasks) ·
[dictConfig never starts the queue listener](#14-dictconfig-never-starts-the-queue-listener)

## 1. QueueHandler destroys structured exceptions

`logging.handlers.QueueHandler.prepare()` calls `self.format(record)` and then sets `msg`
to that string and `args`, `exc_info`, `exc_text`, `stack_info` to None. That is
deliberate, so records can be pickled across a process boundary. Inside one process it silently degrades
JSON output. With a stock QueueHandler in front of the JSON formatter:

```json
{
  "level": "ERROR",
  "message": "payment rejected\nTraceback (most recent call last):\n  File \"service_example.py\", line 49, in handle_request\n    charge_id = charge_card(...)\n  ...\nValueError: amount exceeds per-transaction limit",
  "module": "service_example"
}
```

The traceback is inside `message` and there is no `exception` object at all. With
`obslog.setup.ContextQueueHandler` (whose `prepare()` returns the record untouched):

```json
{
  "level": "ERROR",
  "message": "payment rejected",
  "exception": {
    "type": "ValueError",
    "message": "amount exceeds per-transaction limit",
    "traceback": "Traceback (most recent call last):\n  File \"service_example.py\", line 49, ..."
  }
}
```

Locked in by `test_stock_queue_handler_would_lose_the_structure` and
`test_queue_handler_preserves_exc_info_and_context`.

## 2. Context filters behind a queue see nothing

A filter runs in the thread of the handler it is attached to. Behind a queue, the sinks run
in the listener's thread, where the request-id ContextVar is empty. Same program, filters
on the sinks:

```json
{"level": "INFO", "message": "request received", "service": "checkout-api"}
```

Filters moved onto the QueueHandler (which runs in the caller's thread):

```json
{"level": "INFO", "message": "request received", "service": "checkout-api", "request_id": "29a6002c26fc", "path": "/charge"}
```

`build_config()` places them accordingly, and only when a queue is actually in use.

## 3. Propagation duplicates output

```python
h = logging.StreamHandler(sys.stdout)
root = logging.getLogger(); root.addHandler(h); root.setLevel(logging.INFO)
child = logging.getLogger("app.db"); child.addHandler(h)   # handler on both
child.info("connected")
```

```
app.db -> connected
app.db -> connected
--- with propagate=False ---
app.db -> connected
--- logger level vetoes handler level ---
app.quiet -> seen              # the INFO line never reached the INFO handler
```

Two lessons: a handler on a child *and* on root emits twice, and a logger's own level kills
a record before any handler is consulted.

## 4. W&B drops out-of-order steps without raising

```python
run.log({"loss": 1.0}, step=0)
run.log({"loss": 0.5}, step=5)
run.log({"loss": 99.0}, step=1)    # lower than the current step
run.log({"loss": 0.25}, step=6)
```

```
after step0: step= 0 summary= None
after step5: step= 5 summary= 1
after step1: step= 5 summary= 1     <- run.step did not move; the write vanished
after step6: step= 6 summary= 0.5   <- the committed value is step 5's, not 99
```

Identical numbers on 0.28.2, 0.29.0, 0.30.0 and 0.31.0. What changed is the terminal
signal. On 0.31.0 the offline run prints:

```
wandb: WARNING Tried to log to step 1 that is less than the current step 5. Steps must be
monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric
to log data out of order.
```

0.27.0 through 0.30.0 printed nothing in offline mode (the 0.31.0 changelog lists it as a
fixed regression from 0.27.0). Either way there is no exception and the data is gone, and
a warning on a busy console is easy to miss. This is rule WB006's reason for existing.

## 5. W&B summary lags one commit when you pass `step=`

Visible in the same output: after logging step 5, `run.summary["loss"]` still reads the
step-0 value, because with explicit steps a step commits only when a later step begins.
Without `step=` (the form this skill prescribes) every `run.log` call commits at once:

```
summary m after first log: 1
summary m after second log: 2
summary m after third (lower): 0.5     <- last value, not the best one
```

`commit=False` behaves like the explicit-step case (`summary` is None until the next
committing call). Either way, track best-so-far in your own variable and write it to the
summary explicitly rather than reading `run.summary` mid-run to make a decision.

## 6. `define_metric(summary="min")` changes the summary's type

```python
with start_run(project="p", config={}, summary_goals={"val/loss": "min"}) as run:
    for value in (0.5, 0.2, 0.9):
        run.log({"val/loss": value})
    print(type(run.summary["val/loss"]).__name__, dict(run.summary["val/loss"]))
```

```
SummarySubDict {'min': 0.2}
```

The value is a sub-dict, not a float. Code doing `if run.summary["val/loss"] < best:`
raises a TypeError, in a place nobody looks. Setting an explicit key
(`run.summary["best_val_loss"] = best`) avoids it.

## 7. Config values are immutable (new keys are not)

```python
run.config["lr"] = 2e-3
```

```
ConfigError: Attempted to change value of key "lr" from 0.001 to 0.002
If you really want to do this, pass allow_val_change=True to config.update()
```

`run.config.update({"lr": 2e-3})` raises the same error. `run.config.update({"lr": 2e-3},
allow_val_change=True)` is the escape hatch. Adding a key that did not exist
(`run.config["new_key"] = 5`) is allowed. The `wandb.init` docstring notes that
`allow_val_change` defaults to True in notebooks, so this guard is off there.

## 8. The global run breaks after finish

```
wandb.log worked with implicit global run; run.step = 1
second run id    : gjwtr0mb
wandb.log goes to: gjwtr0mb | expected: gjwtr0mb
after finish, wandb.run = None
wandb.log after finish raised: Error You must call wandb.init() before wandb.log()
```

Module-level `wandb.log` follows whichever run is current, and raises once none is. A
second `wandb.init()` while a run is still open does not start a new run in a script: the
default `reinit="default"` means `"return_previous"` outside notebooks, so it returns the
open run and silently ignores the new `config` (same on 0.28.2 and 0.31.0):

```
second init returned same run: True 6g2ylhgr 6g2ylhgr config phase: train
```

`reinit="create_new"` gives a second, independent run but does **not** update `wandb.run`,
so module-level `wandb.log` keeps writing to the first one. Rule WB004: hold the object
returned by `init`.

## 9. Offline runs cannot resume locally

```python
run = wandb.init(project="probe", id=previous_id, resume="allow")   # WANDB_MODE=offline
```

```
resume allow offline: id r0ozq5wq same? True resumed False step 0      # 0.31.0
resume must  offline: id r0ozq5wq same? True resumed False step 0      # 0.31.0
```

The run keeps the id you passed, but `run.resumed` is False and `run.step` restarts at 0:
the local process has no server-side history to continue from. 0.31.0 added offline
support for `resume="must"` at *sync* time (`wandb sync` attaches the data to the existing
run, and fails if there is none), but inside the training process nothing changes. Take
the resume point (epoch, global step) from your checkpoint, never from `run.step`, when a
run may be offline.

## 10. Offline artifacts and `run.url`

```
wandb: WARNING URL not available in offline run
INFO train: run k22rqw6u started (url=None)
...
ArtifactNotLoggedError: 'Artifact.aliases' used prior to logging artifact or while in
offline mode. Call Artifact.wait() before accessing logged artifact properties.
```

`run.url` is `None` offline, so formatting it into a log line yields "None"; and reading a
logged artifact's `aliases`/`version` raises until the upload completes. Both are guarded
in `obslog/wandb_run.py`. Unchanged on 0.31.0.

## 11. What the context manager actually changes

Offline runs, wandb 0.31.0 (0.28.2 identical), state read back with
`wandb leet inspect --summary`:

| Script | Process exit | Recorded state |
|---|---|---|
| `run = wandb.init(...)`, then an uncaught `raise` | 1 | failed (exit code 1) |
| `with wandb.init(...) as run:`, uncaught `raise` inside | 1 | failed (exit code 1) |
| `run = wandb.init(...)`, `raise` caught by the caller, script continues | 0 | **finished (exit code 0)** |
| `with wandb.init(...)`, `raise` caught by the caller | 0 | failed (exit code 1) |
| no `finish()`, clean exit | 0 | finished (exit code 0) |
| `os.kill(os.getpid(), SIGKILL)` or unhandled SIGTERM, with or without `with` | 137 / 143 | no exit record (crashed) |
| SIGTERM handler that calls `sys.exit(128 + signum)`, inside `with` | 143 | failed (exit code 1) |

So an *uncaught* crash is marked failed either way, because `wandb.init` installs its own
`sys.excepthook` and `atexit` hook (`wandb/sdk/lib/exit_hooks.py`), and the metrics logged
before the crash were flushed in every case except the signal kills. What the context
manager buys is that an exception *caught further up* (a sweep agent's function, a retry
loop, a notebook cell, a train-then-eval script) closes the run as failed at the point it
happened instead of leaving it open to be recorded as finished later. A hard kill is
handled by neither; the server marks it crashed when heartbeats stop
(<https://docs.coreweave.com/products/wandb/runs/run-states.md>).

## 12. No network: init raises instead of degrading

With a key configured but the server unreachable (`WANDB_BASE_URL=http://127.0.0.1:9`),
`wandb.init()` retries for `init_timeout` (default 90 s) and then raises:

```
CommError Timed out while retrying: Post "http://127.0.0.1:9/graphql": dial tcp 127.0.0.1:9: connect: connection refused
```

With no key at all it raises immediately with `UsageError` ("No API key configured. Use
`wandb login` to log in."). Neither falls back to offline on its own. `obslog.start_run`
catches the `CommError` and retries with `mode="offline"`, and `resolve_mode()` picks
offline up front when `wandb.login(prompt=False, verify=False)` returns False.

Detecting "is there a key" by grepping `~/.netrc` for `api.wandb.ai` broke in 0.31.0: the
default server moved to `https://api.forge.coreweave.com`, and a fresh `wandb login` now
writes `machine api.forge.coreweave.com`. Old `api.wandb.ai` entries are still read.

## 13. ContextVars, threads, and tasks

`obslog.context.log_context` stores the context in a `ContextVar`. Where it is visible,
run on 3.11 to 3.14 (default builds) with `request_id="req-1"` bound in the caller:

| Where the log call runs | Sees `request_id`? |
|---|---|
| asyncio task created *after* binding | yes (tasks copy the creator's context) |
| asyncio task created *before* binding | no |
| a child task that binds its own value | its own value; the parent is unaffected |
| `asyncio.to_thread(fn)` | yes (it copies the context explicitly) |
| `loop.run_in_executor(None, fn)` | **no** |
| `threading.Thread(target=fn)` / `ThreadPoolExecutor.submit(fn)` | **no** |
| `pool.submit(contextvars.copy_context().run, fn)` | yes |

On the free-threaded 3.14 build (`python3.14t`), or with `-X thread_inherit_context=1`,
new threads start with a copy of the caller's context, and the two "no" thread rows become
"yes". On a default build, wrap the callable in `copy_context().run` when work fans out to
a pool, or bind the id again inside the worker (as `service_example.py` does). Context is
also absent in a `QueueListener`'s thread, which is why section 2 puts the filter on the
`QueueHandler`.

## 14. dictConfig never starts the queue listener

```python
logging.config.dictConfig(build_config(level="INFO"))   # Django LOGGING=, gunicorn logconfig_dict
logging.getLogger("app").warning("can you see me?")
```

On 3.12+ with the original `ContextQueueHandler`, nothing printed and the record stayed in
the queue (`queue size still pending: 1`), because `dictConfig` builds the
`QueueListener` but does not start it, and only `configure()` started it.
`ContextQueueHandler.emit()` now starts its listener on first use, and
`test_build_config_works_without_configure` locks that in. On 3.11 there is no queue in
the config, so the record printed either way.

## Reproducing all of it

```bash
python -m pip install wandb pytest
PYTHONPATH=assets python -m pytest assets/tests -q      # historical: 42 passed (40 + 2 skipped on 3.11), machine not recorded; 43 passed on 2026-11-03, macOS arm64, Python 3.13
PYTHONPATH=assets LOG_FORMAT=json  python assets/examples/service_example.py
PYTHONPATH=assets WANDB_MODE=offline python assets/examples/train_example.py --epochs 3
python scripts/audit_observability.py assets/examples/bad_example.py   # 17 findings
```
