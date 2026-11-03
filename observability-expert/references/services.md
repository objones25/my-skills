# Instrumenting a service

Read this when the code handles requests, jobs, or messages: anything where many units of
work are in flight and a log line without an id is unusable.

Contents: [The three moves](#the-three-moves) · [Framework wiring](#framework-wiring) ·
[What to log per request](#what-to-log-per-request) · [Volume control](#volume-control) ·
[Health of the logging itself](#health-of-the-logging-itself)

## The three moves

`assets/examples/service_example.py` is the whole pattern in 80 lines, runnable with no
framework:

1. **Configure once, in the entry point.** `configure(service="checkout-api",
   version="1.4.2", env="prod", fmt="json")`. `service`/`version`/`env` are stamped on
   every record, which is what a log backend groups by.
2. **Bind a request id at the boundary.** `with log_context(request_id=..., path=...):`
   around the handler. Every record emitted inside (including from libraries and from
   functions five frames down) carries it, as long as the work stays in this thread or
   in asyncio tasks created inside the block. Work handed to a thread pool needs
   `contextvars.copy_context().run` (see `stdlib-logging.md`).
3. **Emit one access record per request** with status, duration, and the fields you would
   group by, at a level that reflects the outcome.

Real output, three concurrent requests, one of them failing:

```
INFO  __main__  service_example:47 | request received
DEBUG __main__  service_example:30 | charging 1200 cents
INFO  __main__  service_example:56 | payment captured
INFO  access    service_example:62 | POST /charge -> 200 in 40.7ms
ERROR __main__  service_example:53 | payment rejected
Traceback (most recent call last):
  ...
ValueError: amount exceeds per-transaction limit
WARNING access  service_example:62 | POST /charge -> 400 in 44.5ms
```

and the same thing as JSON, where the interleaving stops mattering because `request_id`
reassembles each request:

```json
{"timestamp": "2026-08-26T22:19:34.304+00:00", "level": "ERROR", "logger": "__main__", "message": "payment rejected", "module": "service_example", "line": 53, "service": "checkout-api", "version": "1.4.2", "env": "dev", "exception": {"type": "ValueError", "message": "amount exceeds per-transaction limit", "traceback": "Traceback (most recent call last):\n  ..."}, "request_id": "f31ecef005b8", "path": "/charge"}
```

## Framework wiring

**Accept an inbound id.** If the caller sends `X-Request-ID` (or `traceparent`), reuse it
rather than generating a new one; that is what makes a single id span services. Generate
one only when the header is absent, and echo it back on the response so a user can quote it
in a bug report.

**FastAPI / Starlette**: middleware, since it wraps both the handler and the response:

```python
@app.middleware("http")
async def logging_middleware(request, call_next):
    request_id = request.headers.get("x-request-id") or new_request_id()
    started = time.perf_counter()
    with log_context(request_id=request_id, path=request.url.path, method=request.method):
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        access_logger.log(
            logging.WARNING if response.status_code >= 500 else logging.INFO,
            "%s %s -> %d in %.1fms",
            request.method, request.url.path, response.status_code, duration_ms,
            extra={"status": response.status_code, "duration_ms": round(duration_ms, 1)},
        )
        response.headers["x-request-id"] = request_id
        return response
```

ContextVars work here because each request runs in its own asyncio task; a value set in
the middleware is visible in the handler and invisible to concurrent requests. A sync
(`def`) endpoint runs in Starlette's threadpool via `anyio.to_thread.run_sync`, which
also copies the context, but anything you push to your own `ThreadPoolExecutor` or
`loop.run_in_executor` does not.

**uvicorn / gunicorn** install their own handlers on the `uvicorn`, `uvicorn.access`,
and `gunicorn.error`/`gunicorn.access` loggers, with `propagate=False` (verified with
uvicorn 0.54.0 and gunicorn 26.2.0). So their lines come out in their own format, and
never reach your JSON handlers on root. If you instead flip `propagate` back on without
removing their handlers, every line prints twice, once in each format. When uvicorn is
started programmatically, `uvicorn.run(app, log_config=None)` leaves logging alone;
otherwise undo theirs after `configure()`:

```python
for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn.error", "gunicorn.access"):
    logger = logging.getLogger(name)
    logger.handlers.clear()      # drop their handlers
    logger.propagate = True      # let the records reach root, where yours live
```

Run uvicorn with `--no-access-log` if you emit your own access record, so you get one per
request rather than two. For gunicorn, put `logconfig_dict = build_config(...)` in
`gunicorn.conf.py` (it is a config-file setting with no CLI flag; the CLI has
`--log-config` and `--log-config-json`).

**Django**: `LOGGING = build_config(...)` in settings, with
`LOGGING_CONFIG = "logging.config.dictConfig"` (the default). Django applies it during
setup, so `configure()` is not called separately. Middleware handles the id binding.

**Celery / RQ**: bind `task_id` and `task_name` in a `task_prerun` signal handler and
reset in `task_postrun`. Celery configures the root logger itself by default; connecting
any receiver to `setup_logging` stops that ("Celery won't configure the loggers if this
signal is connected", Celery signals docs), so connect one that calls `configure()`.

## What to log per request

| Event | Level | Fields |
|---|---|---|
| Request received | INFO (DEBUG if very high volume) | method, path, request_id |
| Outbound call to another service | INFO | target, duration_ms, status |
| Business decision | INFO | the decision and the reason, not the inputs |
| Handled failure (4xx) | WARNING | status, reason code |
| Unhandled failure (5xx) | ERROR with `exc_info` | the traceback |
| Response sent | INFO / WARNING by status | status, duration_ms |

Duration goes in a field, not only in the message, so it can be aggregated without parsing.
User identifiers are fine; user *data* is not (rule LOG008): log `user_id`, never the
email, the address, or the card.

## Volume control

Logging is a cost centre, and per-request logs are the first thing to blow a budget.

- **Level, per subsystem.** `build_config(library_levels={"botocore": "WARNING", ...})`
  quiets the libraries that log per network call. This is the biggest single win.
- **DEBUG off in production, but switchable.** `LOG_LEVEL=DEBUG` on one instance, or a
  runtime endpoint that calls `logging.getLogger("myapp.db").setLevel(DEBUG)` for five
  minutes, beats redeploying to see one value.
- **Sample the repetitive.** For an event that fires thousands of times a second, log every
  Nth with the count included, so the evidence survives without the volume.
- **Aggregate instead.** If you are logging a line to count something, you want a metric,
  not a log.
- **Rate-limit warnings.** A `WARNING` that fires per request during an outage produces
  gigabytes exactly when the system is already struggling. A filter that drops duplicates
  within a window, or a `logger.warning(..., extra={"count": n})` once per window, keeps
  the signal.

## Health of the logging itself

- **Uncaught exceptions.** `install_excepthook()` routes them through logging, so the last
  thing a crashing process does is emit a structured record instead of a raw stderr
  traceback your pipeline never sees. In threads, `threading.excepthook` needs the same
  treatment; in asyncio, `loop.set_exception_handler`.
- **Warnings.** `configure()` calls `logging.captureWarnings(True)`, so `DeprecationWarning`
  and friends arrive as `py.warnings` records instead of bare stderr text.
- **Flush on shutdown.** With a queue, `atexit` stops the listener (already wired). Under
  SIGTERM in a container, handle the signal and exit cleanly rather than being killed
  mid-queue.
- **Never let logging raise.** `logging` swallows formatting errors into
  `logging.raiseExceptions` output rather than propagating them; keep it that way in
  production. If you write a custom handler, its `emit` must not raise.
