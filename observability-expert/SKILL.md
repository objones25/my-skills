---
name: observability-expert
description: Observability for Python code - stdlib logging setup, structured JSON logs, request/correlation IDs, and Weights & Biases experiment tracking. Use when the user asks to add or configure logging, replace prints, emit JSON logs, tag records with a request or run id, debug missing or duplicated log output, instrument a training loop, or use W&B (wandb.init, run.log, config, summary, artifacts, sweeps, offline or resumed runs). Also use it proactively when writing a training script, a service entry point, or any code whose failures will need diagnosing later. Ships a 17-check audit script (exit 1 on any finding) and drop-in logging + W&B modules.
---

# Observability Expert

Instrumentation that survives the moment you actually need it: 3am, a failed run, a log
file that is either empty or 400MB of `print`. The default state of a Python codebase is
invisible: prints that vanish under a process manager, exceptions logged without
tracebacks, W&B runs with metrics but no hyperparameters. This skill makes that state
loud instead of quiet.

Verified against **Python 3.11.17, 3.12.13, 3.13.16 and 3.14.6** and **wandb 0.31.0**
(also 0.28.2), October 2026. The behavior claims were executed, not recalled; the
surprising ones are in `references/verified-behavior.md` with their output. Read that
file before telling a user how W&B behaves on an older or newer release.

## Definitions used throughout

| Term | Meaning |
|---|---|
| **Logger** | The object you call (`logger.info(...)`). Decides *whether* a message survives, by level and filters. Named; dotted names form a tree (`a.b`'s parent is `a`). |
| **Handler** | Decides *where* a surviving record goes: stdout, a file, a queue, a network socket. Has its own level and formatter, independent of the logger's. |
| **Formatter** | Decides what a record *looks like* as text. One handler, one formatter. |
| **Filter** | A hook returning True (keep) or False (drop) that may also *mutate* the record on the way through. The stdlib's designated place to attach context. |
| **LogRecord** | The object built per log call: message, level, logger name, timestamp, source line, thread, plus anything you passed via `extra={...}`. |
| **Propagation** | After a logger's own handlers run, the record is passed to its parent, up to root. On by default. The cause of most duplicate-line bugs. |
| **`dictConfig`** | `logging.config.dictConfig(d)`: configures the whole logging tree from one dict. The supported way to set up more than one destination. |
| **Structured log** | A record emitted as machine-parseable fields (JSON) rather than a sentence, so `request_id` is a field you filter on rather than a substring you grep. |
| **Correlation ID** | An id attached to every record produced while handling one request or job, so a single unit of work can be reassembled from interleaved output. |
| **`ContextVar`** | `contextvars.ContextVar`: a variable whose value belongs to the current *context*. Each thread starts with its own empty context, and each asyncio task runs in a **copy** of the context it was created in, so a task created inside a request sees that request's id and changes it makes stay private. How correlation IDs travel without being passed as arguments. It does **not** follow work into a thread pool on its own (see `references/stdlib-logging.md`). |
| **W&B run** | One execution of a script, in W&B's model. Owns a config (inputs), a history (time series), a summary (final values), and artifacts. |
| **W&B config** | The run's hyperparameters. Written at `init`. Changing the value of an existing key raises `ConfigError` unless you call `run.config.update(..., allow_val_change=True)`; adding new keys is allowed. (In notebooks `allow_val_change` defaults to True.) |
| **W&B summary** | One value per metric: the run's headline number. Defaults to the last logged value, which is not the same as the best one. |
| **W&B artifact** | A versioned file or directory (checkpoint, dataset) bound to the run that produced it, addressable later by alias (`model:best`). |
| **Sweep** | A W&B-orchestrated hyperparameter search: a YAML/dict search space and an agent that starts one run per configuration. |

## The division of labor

Logs, metrics, and runs answer different questions. Sending everything to one of them is
the most common structural mistake, and it is expensive in both directions: per-step
metrics in a log file are unreadable, and prose events in W&B are unsearchable.

| Question | Where it belongs | Shape |
|---|---|---|
| What happened, in what order, and what went wrong? | stdlib `logging` | timestamped records, one per event, level-graded |
| How did a number move over time? | W&B history (`run.log`) | metric name -> value per step |
| What produced this result? | W&B config + artifacts | hyperparameters, data version, checkpoint |
| Which run was best? | W&B summary | one value per metric per run |

A training script needs both: the log says "run started, resumed from step 4000, hit NaN
at 4200, aborted"; W&B says the loss curve was flat for 200 steps before that. Neither
one alone tells you what happened.

## Workflow

**1. Read the code before instrumenting it.** Find the entry points (`main`, `train.py`,
the ASGI app), the existing logging setup if any, and every `print`. Instrumentation
added without knowing where the process starts ends up configured twice, or never.

**2. Install one configuration point.** Copy `assets/obslog/` into the project and call
`configure()` once in each entry point, before anything else logs. Read
`references/stdlib-logging.md` for what the config does and why each choice is made.
Library code (anything importable that is not an entry point) only creates loggers;
configuration belongs to the entry point.

**3. Replace prints with module loggers.** `logger = logging.getLogger(__name__)` at the
top of each module, `%`-style arguments at each call site. Levels: DEBUG for the detail
you would only want while debugging, INFO for lifecycle events worth keeping, WARNING for
degraded-but-working, ERROR for a failed operation, CRITICAL for a process that cannot
continue.

**4. Add correlation.** A service binds a request id per request; a training script binds
the run id and epoch. `assets/obslog/context.py` does this with a ContextVar and a filter,
so no function signature changes. See `references/services.md`.

**5. Instrument the experiment side.** For anything with a training loop, wire W&B per
`references/wandb.md`: whole config at `init`, `define_metric` for x-axes, artifacts for
checkpoints, a context manager so any exception inside the block marks the run failed.
`assets/obslog/wandb_run.py` is the wrapper; `assets/examples/train_example.py` is a
complete working loop.

**6. Run the audit and the tests.**

```bash
python scripts/audit_observability.py src/            # exit 1 on any finding
python scripts/audit_observability.py --list-codes .  # what each code means
PYTHONPATH=assets python -m pytest assets/tests -q    # the plumbing's own tests
```

The work is done when the audit exits 0 (or every remaining finding is reported to the
user with a reason), and you have run the program and seen each record you added appear
once, at the level you chose, with the fields you expected. Instrumentation is the one
kind of code whose bugs are invisible in every test that does not look at the output.

## Hard rules

Each has a code the audit reports. The reasoning and sources are in `references/rules.md`;
`assets/examples/bad_example.py` violates every one of them on purpose, as the audit's test
fixture.

| Code | Rule |
|---|---|
| LOG001 | Pass values as arguments (`logger.info("order %s", oid)`), not f-strings or `.format()`. The message is then built only if the level passes. |
| LOG002 | Log through `logging.getLogger(__name__)`, not the module-level `logging.info(...)`, which is the root logger and merges your messages with everyone else's. |
| LOG003 | Name every logger; `logging.getLogger()` with no name returns root. |
| LOG004 | No `print` for diagnostics. It has no level, no timestamp, no routing, and it disappears under most process supervisors. |
| LOG005 | Only entry points configure logging. A library that calls `basicConfig`/`dictConfig` on import silently overrides its host application. |
| LOG006 | Inside `except`, log with `logger.exception(...)` or `exc_info=True`. A traceback cannot be reconstructed later; the message can. |
| LOG007 | An `except` block must log, re-raise, or return a fallback; one that does none of these deletes the failure. |
| LOG008 | Keep secrets (passwords, tokens, keys, full card numbers) out of log calls and out of W&B config, which is readable by everyone with project access. |
| LOG009 | `logger.warn()` is deprecated; use `.warning()`. |
| LOG010 | Set `disable_existing_loggers: False` in `dictConfig`. The default `True` disables every existing logger that the config does not name (or name an ancestor of), which is usually every library imported before the call. |
| LOG011 | Log the exception itself (`exc_info`), not `str(exc)`, which discards the traceback and the exception type. |
| WB001 | Always pass `project=` to `wandb.init()`. Without it W&B guesses from the git repo or script name and falls back to "uncategorized", so the same experiment scatters by working directory. The first positional argument is `entity`, not `project`. |
| WB002 | Always pass the whole `config=`. Metrics without the hyperparameters that produced them are not an experiment record. |
| WB003 | Use `with wandb.init(...) as run:`. Any exception inside the block then marks the run **failed** right there. Without it, an uncaught crash is still marked failed (wandb hooks `sys.excepthook`), but an exception caught further up (a sweep agent, a retry loop, a notebook) leaves the run open and it ends up **finished**, exit 0. |
| WB004 | Hold the run object instead of module-level `wandb.log`/`wandb.summary`/`wandb.run`. Those target an implicit global run, raise after `finish()`, and a second `wandb.init()` while a run is open returns the *same* run (its config ignored) unless `reinit` says otherwise. |
| WB005 | No W&B credential in source. Use `wandb login` or the `WANDB_API_KEY` environment variable. |
| WB006 | Log your counter as a metric and make it the x-axis with `define_metric`, instead of passing `step=` to `run.log`. A step lower than the current one is dropped with no exception (wandb 0.31 prints a terminal warning; 0.27 to 0.30 printed nothing offline). |

Deliberate exceptions are marked in the source, where they are visible in review:
`# obs: allow LOG003 (reaching root on purpose)`. Test suites that exercise the
violations need these; application code rarely should.

## Reference files

Read the one that matches the work:

- `references/rules.md`: read when a user pushes back on a rule or asks why; every rule
  with rationale, the failure it prevents, and sources.
- `references/stdlib-logging.md`: read before writing any logging config, and whenever
  output is missing, duplicated, or loses its context (threads, queues, processes).
- `references/wandb.md`: read before writing any `wandb` call: run lifecycle, config,
  x-axes, summary, artifacts, sweeps, offline and resumed runs, distributed runs.
- `references/services.md`: read when the code handles requests or jobs: request ids,
  access logs, FastAPI/uvicorn/gunicorn/Django/Celery wiring, volume control.
- `references/ml-training.md`: read when writing or reviewing a training loop: what to
  log, how often, and what goes to the log versus the run.
- `references/verified-behavior.md`: the surprising behaviors, with the commands and real
  output that established them, and which versions they apply to.

## What ships

```
observability-expert/
├── SKILL.md                       this file
├── INSTALL.md                     how to install and wire it up
├── references/                    the six documents above
├── assets/
│   ├── obslog/                    the drop-in package
│   │   ├── setup.py               dictConfig builder, queue wiring, excepthook
│   │   ├── json_formatter.py      JSON-lines formatter with structured exceptions
│   │   ├── context.py             ContextVar + filter for correlation IDs
│   │   ├── redact.py              secret redaction as a safety net
│   │   └── wandb_run.py           run lifecycle, artifacts, offline fallback
│   ├── examples/
│   │   ├── train_example.py       instrumented training loop (runs offline)
│   │   ├── service_example.py     request-scoped logging under concurrency
│   │   └── bad_example.py         every rule violated once, on purpose
│   └── tests/test_obslog.py       43 tests (43 passed 2026-11-03, macOS arm64, py3.13)
└── scripts/audit_observability.py 17-check AST audit; exit 1 on any finding
```
