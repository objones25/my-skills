# The rules, with reasons

Every rule the audit enforces, what it prevents, and where the claim comes from. Read this
when a rule seems arbitrary or a user pushes back on one: an unjustified rule gets worked
around, and a worked-around rule is worse than no rule.

Contents: [Logging rules](#logging-rules) · [W&B rules](#wb-rules) ·
[Judgment calls the audit cannot make](#judgment-calls-the-audit-cannot-make) · [Sources](#sources)

## Logging rules

### LOG001: pass values as arguments, not f-strings

```python
logger.debug("order %s: %d items", order_id, len(items))   # deferred
logger.debug(f"order {order_id}: {len(items)} items")      # built every time
```

`Logger.debug` stores `msg` and `args` on the record and only calls `msg % args` if a
handler actually emits it. An f-string is evaluated by the interpreter *before* the call,
so a DEBUG line in a hot loop pays full formatting cost even when the level is INFO.
Coralogix measured a million calls of `.format()`-built versus `%`-deferred messages:
2197 ms vs 1111 ms when the level was off (about half the time saved), 11061 ms vs
10149 ms when it was on (about 8%). The exact number matters less than the shape: the
work is unconditional and the benefit is conditional.

The second reason matters more at scale: `record.msg` stays a stable template. Log
backends group by that template, so a million distinct f-string outputs become one event
type with a million instances instead of a million unique messages.

Exception: `logging.warning("...")` with no interpolation at all is fine; there is nothing
to defer.

### LOG002 / LOG003: never the root logger in application code

`logging.info(...)` is a convenience wrapper around the *root* logger. Two costs: your
records become indistinguishable from every library's records at the same level, and you
lose the per-module control that logger names buy (`logging.getLogger("myapp.db").setLevel(DEBUG)`
while everything else stays at INFO).

`logging.getLogger(__name__)` gives a logger named for the module, which is also the value
of `%(name)s` in your format string, so the source of a line is visible without reading
code. One logger per module is enough; per-class or per-function loggers add noise without
adding control.

Root still matters: it is where the *handlers* live (see LOG005). The rule is about who
you call, not where output is configured.

### LOG004: no print for diagnostics

`print` writes to stdout with no level, no timestamp, no logger name, and no way to turn
it off or route it elsewhere. Under a process supervisor, in a container, or in a Jupyter
kernel, it may not be captured at all. And it cannot be silenced by a consumer of your
code, which is why a library that prints is a library people vendor and patch.

`print` is legitimate as a program's *output*: a CLI's results, a script's report. The
audit treats a module with an `if __name__ == "__main__":` block as an entry point where
printing is allowed, and flags it everywhere else.

### LOG005: only entry points configure logging

Configuration is process-global. A library that calls `basicConfig()` or `dictConfig()` on
import reconfigures the application that imported it: silently, at import time, in an
order that depends on import order. The stdlib's own guidance for library authors is to
create loggers and log to them, and to leave configuration to the application. If nothing
configures logging, the stdlib default emits WARNING and above to stderr, which is a
reasonable fallback for a library's users.

Corollary: `basicConfig()` is a no-op if the root logger already has handlers, and after
the first call, unless `force=True`. Two libraries both calling it means one of them
silently loses.

### LOG006 / LOG011: keep the traceback

```python
except ValueError:
    logger.exception("payment rejected")            # traceback attached
    logger.error("payment rejected", exc_info=True) # identical
    logger.error(str(exc))                          # traceback destroyed
```

`logger.exception()` is `logger.error()` with `exc_info=True`, valid only inside an
exception handler. The message text is the part you can always reconstruct from the code;
the traceback (the actual line that raised, and the call chain) exists only at the
moment of the exception. Logging `str(exc)` throws away the one irreplaceable thing and
keeps the reproducible one.

With the JSON formatter in this skill, `exc_info` also becomes a structured field
(`exception.type`, `exception.message`, `exception.traceback`), so a backend can count
errors by exception class.

### LOG007: an except block must log or re-raise

```python
except KeyError:
    pass          # the failure never happened, as far as anyone can tell
```

This is the highest-value check in the audit. It produces the bug report that says
"nothing happens when I click the button" with an empty log to match. If a failure is
genuinely expected and harmless, that is a `logger.debug` line explaining why, not
silence. Returning a fallback value counts as handling it, and so does the
optional-dependency idiom (`except ImportError: torch = None`); the audit accepts both.
A bare `continue` in a loop does not count.

### LOG008: no secrets, anywhere near a log

Logs are copied: shipped to aggregators, kept in backups, pasted into tickets, read by
whoever has dashboard access. A W&B run config is worse: it is exported by the public
API and visible to everyone with project access. Treat both as public.

Twitter's 2018 incident (plaintext passwords written to an internal log, ~330M users asked
to reset) is the canonical example: the logging was internal, the log was not the
attacker's target, and the exposure was real anyway.

`assets/obslog/redact.py` redacts secret-looking fields as a safety net, and
`wandb_run.redact_config` does the same for run configs. Both, and the audit, match names
word by word (`hf_token`, `apiKey`, `db_password`), so ML names that merely contain a
secret-ish substring (`max_tokens`, `tokenizer`, `tokens_per_sec`, `author`, `passes`)
are left alone. Neither is a substitute for not passing the value.

### LOG009: `warn` is deprecated

`Logger.warn` has emitted a `DeprecationWarning` since Python 3.3 in favor of
`Logger.warning`, and still exists (with the warning) through 3.15. Mechanical fix, no
argument.

### LOG010: `disable_existing_loggers: False`

`dictConfig` defaults this to `True`, which disables every logger that exists at
configuration time unless the new config names it or one of its ancestors (verified on
3.11 to 3.15: with `"loggers": {"named": ...}`, `named.child` stays enabled and `lib` is
disabled). Since libraries create their loggers at import, and imports happen before
`configure()`, the default silences exactly the third-party diagnostics you will want
during the next incident. The audit flags the key when it is absent or explicitly `True`.

## W&B rules

### WB001: always name the project

`wandb.init()` without `project=` guesses one: the git repository's name, or the script's,
and "uncategorized" only when neither works (verified on 0.31.0: a script in a repo named
`myrepo` logged to project `myrepo`; the same script outside a repo logged to
`uncategorized`). Either way the destination depends on where the script happened to be
run from, so the same experiment scatters across projects. The project is also the unit
of access control and of the runs API (`api.runs("entity/project")`). Note that the first
*positional* argument of `wandb.init` is `entity`, not `project`.

### WB002: log the whole config, not a subset

```python
with wandb.init(project="lm", config=asdict(cfg)) as run: ...
```

A run's value is `(inputs, outputs)`. Metrics alone are outputs with the inputs missing:
you can see that run 47 was better without being able to say why, and you cannot filter
runs by a hyperparameter that was never recorded. Pass the entire config object --
`vars(args)`, `dataclasses.asdict(cfg)`, `OmegaConf.to_container(cfg)`, rather than the
three fields you currently care about. The field you leave out is the one that explains the
anomaly.

Config values are immutable after `init`. Assigning a new value to an existing key raises
`ConfigError: Attempted to change value of key "lr" from 0.001 to 0.002`; use
`run.config.update({...}, allow_val_change=True)` when a value is genuinely refined later
(e.g. a resolved `auto` batch size). Adding a new key is allowed, and in notebooks
`allow_val_change` defaults to True.

### WB003: use the context manager

```python
with wandb.init(project="lm", config=cfg) as run:
    ...
```

On a normal exit this calls `finish()`. On an exception leaving the block it calls
`finish(exit_code=1)`, which marks the run **failed** at that point.

What happens without it, verified on 0.28.2 and 0.31.0 (`verified-behavior.md` section 11):
an *uncaught* exception still marks the run failed, because `wandb.init` installs its own
`sys.excepthook`, and the metrics already logged are flushed. The failure the context
manager prevents is an exception that is *caught further up*: a sweep agent calling your
function, a retry loop, a train-then-eval script, a notebook. The process carries on, the
run stays open, and it is recorded as **finished** with exit code 0, so a failed trial
looks like a successful one. A `SIGKILL` or OOM kill is outside both mechanisms; W&B marks
that run **crashed** when its heartbeats stop.

In a notebook, where a cell exit is not a process exit, use the `with` block inside the
cell, or hold `run = wandb.init(...)` and call `run.finish()` yourself.

### WB004: hold the run object

`wandb.log({...})` targets an implicit global run (`wandb.run`). After `finish()`,
`wandb.run` is None and the next `wandb.log` raises
`You must call wandb.init() before wandb.log()`. With two runs in one process (a sweep
worker, a train-then-eval script) the global points at whichever is current, silently.
Two more traps, verified on 0.28.2 and 0.31.0: a second `wandb.init()` while a run is open
returns that *same* run in a script and ignores the new `config` (`reinit` defaults to
`"return_previous"` outside notebooks), and `reinit="create_new"` creates a second run
without moving `wandb.run`, so `wandb.log` keeps writing to the first. `run.log(...)` on
the object returned by `init` has none of these failure modes. The same applies to
`wandb.summary` (use `run.summary`) and `wandb.log_artifact` (use `run.log_artifact`).

### WB005: credentials out of source

`wandb login` writes the key to `~/.netrc`; `WANDB_API_KEY` covers CI. A key in a repo is
a key in every clone, every fork, and every model card that copies the training script.

### WB006: do not pass `step=`

W&B's step counter is monotonic. A `run.log(..., step=N)` where N is *lower* than the
current step is discarded: the data is gone, `run.step` does not move, and nothing is
raised. wandb 0.31.0 prints `WARNING Tried to log to step 1 that is less than the current
step 5 ... this data will be ignored`; 0.27.0 through 0.30.0 printed nothing in offline
mode. Mixing explicit steps with implicit ones is how a resumed run, a multi-process
trainer, or a validation loop that logs on its own counter quietly loses half its points.

The supported alternative is to log the counter as a metric and declare it as an axis:

```python
run.define_metric("train/step")
run.define_metric("train/*", step_metric="train/step")
run.define_metric("epoch")
run.define_metric("val/*", step_metric="epoch")

run.log({"train/loss": loss, "train/step": global_step})   # no step= anywhere
run.log({"val/loss": val_loss, "epoch": epoch})
```

Each metric now carries its own x-axis, and the internal step is left alone.

## Judgment calls the audit cannot make

Rules a static checker cannot enforce, but which decide whether the instrumentation is
actually useful:

- **Level discipline.** INFO is for events a healthy system produces at a rate a human can
  read: started, connected, finished, resumed. If your INFO stream is unreadable at normal
  load, the excess belongs at DEBUG. If a WARNING fires on every request, it is not a
  warning, it is a metric.
- **One record per event, not per line of code.** Three consecutive `logger.info` calls in
  one function usually means one record with three `extra` fields.
- **Log at boundaries.** Entering and leaving the process: requests, jobs, external API
  calls, database transactions, file writes. Internal control flow is DEBUG at most.
- **Log the decision, not the data.** "selected shard 3 of 8 (least loaded)" beats dumping
  the load table. If you need the table, it is a DEBUG record with the table in `extra`.
- **Metrics are not logs.** A per-step scalar belongs in W&B (or a metrics backend). A
  per-step log line at 1000 steps/second is a denial of service against your own disk.
- **Sampling over silencing.** For a high-volume event you still want evidence of, log
  every Nth occurrence with the count, rather than dropping the record entirely.

## Sources

- CPython documentation: `Doc/howto/logging.rst`, `Doc/howto/logging-cookbook.rst`,
  `Doc/library/logging.config.rst`, `Doc/library/logging.handlers.rst`: the level
  semantics, propagation, `dictConfig` schema, `disable_existing_loggers` default,
  QueueHandler/QueueListener, and the library-authors guidance.
  <https://docs.python.org/3/howto/logging.html>
- Coralogix, "Python Logging Best Practices": lazy formatting cost measurement (it
  benchmarks `.format()`, not f-strings, and its headline "9-50%" range rounds up its own
  8% active-level figure), `LoggerAdapter`/`setLogRecordFactory` for context, module-level
  loggers, queue-based decoupling. It does not cover `contextvars`.
  <https://coralogix.com/blog/python-logging-best-practices-tips/>
  (Vendor blog for a log-aggregation product; the technical claims are stdlib behavior and
  check out, but treat the "ship it to us" framing accordingly.)
- Dash0, "Logging in Python": `contextvars`-based context, correlation IDs in middleware,
  filters for enrichment, JSON formatting libraries.
  <https://www.dash0.com/guides/logging-in-python> (Same caveat: observability vendor.)
- W&B SDK source and docstrings, `wandb` 0.31.0 and 0.28.2, read from the installed
  package: `wandb_init.py` (init parameters, modes, `reinit`, `resume`), `wandb_run.py`
  (`log`, `define_metric`, `finish`, `__exit__`), `sdk/lib/exit_hooks.py` (the excepthook
  that marks uncaught crashes failed), `internal/handler.py` in 0.28.2 (the monotonic-step
  drop).
- W&B changelog, <https://github.com/wandb/wandb/blob/main/CHANGELOG.md>: 0.29.0 (removed
  `summary="best"`, `wandb.Image` normalization), 0.30.0 (`wandb.login(prompt=False)`),
  0.31.0 (Forge default server, offline step warnings restored, offline `resume="must"`).
- W&B run states: <https://docs.coreweave.com/products/wandb/runs/run-states.md>.
- Twitter, May 2018: "passwords were written to an internal log before completing the
  hashing process"; all 330 million users advised to change passwords.
  <https://www.theregister.com/2018/05/03/twitter_no_big_deal_but_everyone_needs_to_change_their_password_right_now/>
- Behavior verified by execution: see `verified-behavior.md`.
