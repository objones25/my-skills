"""One place that configures logging for the whole process.

Design rules baked in here (the reasoning is in references/rules.md):

- Every handler lives on the **root** logger. Module loggers get none, so a
  record is emitted exactly once and third-party libraries are formatted like
  your own code.
- `disable_existing_loggers` is False, so loggers created by already-imported
  libraries keep working.
- Console output goes to **stdout**, not stderr. Under a process supervisor or
  a container runtime, stdout is the log stream; stderr is where crashes go.
- Slow handlers (files, network) run behind a `QueueHandler`, so a log call
  costs an enqueue rather than an I/O wait on the calling thread.

Call `configure()` exactly once, as early as possible in your entry point, and
never from library code.

    from obslog.setup import configure
    configure(level="INFO", fmt="json", service="checkout-api")
"""

from __future__ import annotations

import atexit
import logging
import logging.config
import logging.handlers
import os
import queue
import sys
import threading
from typing import Any, Literal

Format = Literal["console", "json"]

_configured = False


class ContextQueueHandler(logging.handlers.QueueHandler):
    """A QueueHandler that enqueues the record itself, not a formatted string.

    The stock `QueueHandler.prepare()` calls `self.format(record)` and then sets
    `msg` to that string and `args`, `exc_info`, `exc_text` and `stack_info` to
    None, because a queue may cross a process boundary and those fields are not
    picklable. Two consequences, both silent:

    - a JSON formatter downstream receives a record whose message already has
      the traceback glued onto the end, so the structured `exception` field it
      would have produced never appears;
    - anything that reads the original fields downstream sees None.

    For a queue inside one process -- which is what `dictConfig` builds -- none
    of that stripping is needed, so this override returns the record untouched
    and lets the real handler's formatter do the formatting.

    Trade-off worth knowing: the record is enqueued by reference, so a mutable
    object passed to the log call reflects later mutations if the listener
    formats it after the fact. Log values, not live objects. For a
    multiprocessing queue, keep the stock behavior instead -- pickling is the
    constraint there, not formatting.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record

    def emit(self, record: logging.LogRecord) -> None:
        # dictConfig builds the listener but never starts it. configure()
        # starts it, but Django (`LOGGING = build_config(...)`) and gunicorn
        # (`logconfig_dict`) call dictConfig themselves and never reach
        # configure() -- without this, every record would sit in the queue
        # and nothing would ever be printed.
        listener = getattr(self, "listener", None)
        if listener is not None and not getattr(listener, "_obslog_started", False):
            _start_listener(listener)
        super().emit(record)


# Libraries that log at INFO on every network call or figure render. Left at
# INFO they drown your own records; below WARNING you lose real problems.
DEFAULT_LIBRARY_LEVELS: dict[str, str] = {
    "urllib3": "WARNING",
    "botocore": "WARNING",
    "boto3": "WARNING",
    "s3transfer": "WARNING",
    "matplotlib": "WARNING",
    "PIL": "WARNING",
    "asyncio": "WARNING",
    "filelock": "WARNING",
    "httpx": "WARNING",
    "wandb": "INFO",
}

CONSOLE_FORMAT = (
    "%(asctime)s %(levelname)-8s %(name)-24s %(module)s:%(lineno)d | %(message)s"
)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def build_config(
    *,
    level: str = "INFO",
    fmt: Format = "console",
    service: str | None = None,
    version: str | None = None,
    env: str | None = None,
    log_file: str | None = None,
    file_level: str = "DEBUG",
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    use_queue: bool = True,
    library_levels: dict[str, str] | None = None,
    scan_messages_for_secrets: bool = False,
) -> dict[str, Any]:
    """Return the dictConfig dict. Separated from `configure` so it can be
    inspected, diffed, or dumped to YAML in a test."""
    static: dict[str, Any] = {}
    if service:
        static["service"] = service
    if version:
        static["version"] = version
    if env:
        static["env"] = env

    filters: dict[str, Any] = {
        "context": {"()": "obslog.context.ContextFilter"},
        "redact": {
            "()": "obslog.redact.RedactingFilter",
            "scan_message": scan_messages_for_secrets,
        },
    }

    formatters: dict[str, Any] = {
        "console": {"format": CONSOLE_FORMAT, "datefmt": DATE_FORMAT},
        "json": {
            # "()" (a factory) rather than "class", because a custom formatter
            # takes constructor kwargs that dictConfig's "class" path does not
            # forward.
            "()": "obslog.json_formatter.JSONFormatter",
            "static": static,
        },
    }

    queued = bool(use_queue) and sys.version_info >= (3, 12)

    # Where the enrichment filters go depends on whether a queue is in play.
    # A filter runs wherever its handler runs: on a real handler behind a queue
    # that is the listener's background thread, where the ContextVar holding the
    # request id is empty. Putting them on the QueueHandler keeps them in the
    # thread that made the log call, which is the only place the context exists.
    enrich = ["context", "redact"]
    sink_filters: list[str] = [] if queued else enrich

    handlers: dict[str, Any] = {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": fmt,
            "filters": sink_filters,
            "stream": "ext://sys.stdout",
        }
    }
    sinks = ["stdout"]

    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": file_level,
            "formatter": "json",
            "filters": sink_filters,
            "filename": log_file,
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8",
        }
        sinks.append("file")

    root_handlers = sinks
    if queued:
        # 3.12+ wires the QueueListener automatically from this entry.
        handlers["queue"] = {
            "class": "obslog.setup.ContextQueueHandler",
            "handlers": sinks,
            "filters": enrich,
            "respect_handler_level": True,
        }
        root_handlers = ["queue"]

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": filters,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            name: {"level": lvl}
            for name, lvl in (library_levels or DEFAULT_LIBRARY_LEVELS).items()
        },
        # Root at DEBUG lets each handler decide its own cutoff; a root level
        # above a handler's level would silently veto that handler.
        "root": {"level": "DEBUG", "handlers": root_handlers},
    }


def configure(
    *,
    level: str | None = None,
    fmt: Format | None = None,
    service: str | None = None,
    version: str | None = None,
    env: str | None = None,
    log_file: str | None = None,
    use_queue: bool | None = None,
    force: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Configure process-wide logging. Env vars fill in anything not passed.

    Env: LOG_LEVEL, LOG_FORMAT (console|json), LOG_FILE, LOG_QUEUE (0|1),
    SERVICE_NAME, SERVICE_VERSION, ENVIRONMENT.

    Returns the config dict that was applied. Calling twice is a no-op unless
    `force=True` -- double configuration is the usual cause of duplicated lines.
    """
    global _configured
    if _configured and not force:
        return {}

    resolved_fmt: Format = fmt or os.getenv("LOG_FORMAT", "console")  # type: ignore[assignment]
    if resolved_fmt not in ("console", "json"):
        raise ValueError(f"LOG_FORMAT must be 'console' or 'json', got {resolved_fmt!r}")

    config = build_config(
        level=(level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        fmt=resolved_fmt,
        service=service or os.getenv("SERVICE_NAME"),
        version=version or os.getenv("SERVICE_VERSION"),
        env=env or os.getenv("ENVIRONMENT"),
        log_file=log_file or os.getenv("LOG_FILE"),
        use_queue=(
            use_queue
            if use_queue is not None
            else os.getenv("LOG_QUEUE", "1") not in ("0", "false", "False")
        ),
        **kwargs,
    )
    logging.config.dictConfig(config)
    _start_queue_listener()

    if config["root"]["handlers"] != ["queue"] and sys.version_info < (3, 12):
        logging.getLogger(__name__).debug(
            "queue-based logging needs Python 3.12+ for dictConfig auto-wiring; "
            "handlers are running inline. Use configure_queue_manually() to get "
            "the same behavior on 3.9-3.11."
        )

    # Route warnings.warn() through logging so deprecation warnings land in the
    # same stream as everything else instead of on bare stderr.
    logging.captureWarnings(True)

    _configured = True
    return config


def _start_queue_listener() -> None:
    """Start the listener thread for a dictConfig-created QueueHandler (3.12+).

    dictConfig builds the listener but does not start it -- starting a thread as
    a side effect of parsing config would be surprising. `getHandlerByName` is
    the documented way to reach a handler you named in the config dict.
    """
    if sys.version_info < (3, 12):
        return
    handler = logging.getHandlerByName("queue")
    listener = getattr(handler, "listener", None)
    if listener is None:
        return
    _start_listener(listener)


_listener_lock = threading.Lock()


def _start_listener(listener: logging.handlers.QueueListener) -> None:
    """Start a listener at most once, so a record logged after the atexit stop
    does not try to spawn a thread during interpreter shutdown."""
    with _listener_lock:
        if getattr(listener, "_obslog_started", False):
            return
        listener._obslog_started = True  # type: ignore[attr-defined]
        if getattr(listener, "_thread", None) is None:
            listener.start()
        # Without this, records still sitting in the queue at interpreter
        # shutdown are lost -- exactly when you most want the last few lines.
        atexit.register(_stop_once, listener)


def configure_queue_manually(
    handlers: list[logging.Handler],
    level: int = logging.DEBUG,
    filters: list[logging.Filter] | None = None,
) -> logging.handlers.QueueListener:
    """Python 3.9-3.11 path: build the queue plumbing by hand.

    Same shape as the 3.12+ auto-wiring: root gets one QueueHandler carrying the
    enrichment filters, and a background listener owns the real handlers.

        from obslog.context import ContextFilter
        from obslog.redact import RedactingFilter
        sink = logging.StreamHandler(sys.stdout)
        sink.setFormatter(JSONFormatter())
        configure_queue_manually([sink])
    """
    from obslog.context import ContextFilter
    from obslog.redact import RedactingFilter

    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    root = logging.getLogger()  # obs: allow LOG003 -- reaching root deliberately
    for existing in list(root.handlers):
        root.removeHandler(existing)
    queue_handler = ContextQueueHandler(log_queue)
    for flt in filters if filters is not None else [ContextFilter(), RedactingFilter()]:
        queue_handler.addFilter(flt)
    root.addHandler(queue_handler)
    root.setLevel(level)
    listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True
    )
    listener.start()
    atexit.register(_stop_once, listener)
    return listener


def _stop_once(listener: logging.handlers.QueueListener) -> None:
    """Stop a listener unless it is already stopped.

    `QueueListener.stop()` raises AttributeError when called twice, which at
    interpreter shutdown surfaces as a traceback after the program's last line.
    """
    if getattr(listener, "_thread", None) is not None:
        listener.stop()


def install_excepthook(logger_name: str = "unhandled") -> None:
    """Send uncaught exceptions through logging instead of raw stderr.

    Without this, a crash prints a traceback that your JSON pipeline never sees,
    so the one event you most want in the log backend is the one that misses it.
    """
    logger = logging.getLogger(logger_name)

    def handler(exc_type, exc_value, exc_tb):  # type: ignore[no-untyped-def]
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = handler
