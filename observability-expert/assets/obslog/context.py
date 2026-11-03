"""Ambient context for log records: request IDs, run IDs, anything per-task.

The problem this solves: a log line that says "payment failed" is useless in a
server handling 200 concurrent requests, because you cannot tell which request
it belongs to. Threading a `request_id` argument through every function is the
naive fix and it does not survive contact with a real call stack.

`contextvars.ContextVar` is the stdlib answer. It stores a value that is visible
to everything running in the current *execution context* -- the current thread,
and the current asyncio task, each of which gets its own copy. Setting it inside
one request never leaks into another, including across `await`.

A `logging.Filter` then copies whatever is in the ContextVar onto every
LogRecord, so formatters can render it and JSON output carries it. Filters are
the stdlib's designated place for "mutate the record on the way through": their
`filter()` returns True to keep the record, and anything it sets on the record
first is visible downstream.

Usage:

    from obslog.context import log_context, bind_log_context

    with log_context(request_id="abc123", user_id=7):
        logger.info("charging card")     # record.request_id == "abc123"

    bind_log_context(run_id="a1b2c3")    # no scope: set for the rest of this context
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# One ContextVar holding a dict, rather than one ContextVar per field, so callers
# can add fields the filter has never heard of. The default is a module-level
# empty dict that is never mutated -- every set() below installs a fresh copy.
_EMPTY: dict[str, Any] = {}
_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "obslog_context", default=_EMPTY
)


def get_log_context() -> dict[str, Any]:
    """Current context fields. Returns a copy; mutating it changes nothing."""
    return dict(_log_context.get())


def bind_log_context(**fields: Any) -> contextvars.Token[dict[str, Any]]:
    """Merge `fields` into the current context and return the reset token.

    The token is what `ContextVar.reset()` takes to restore the previous value.
    Prefer the `log_context` context manager, which handles the reset for you.
    """
    merged = {**_log_context.get(), **fields}
    return _log_context.set(merged)


@contextmanager
def log_context(**fields: Any) -> Iterator[dict[str, Any]]:
    """Scope `fields` onto every log record emitted inside the block."""
    token = bind_log_context(**fields)
    try:
        yield get_log_context()
    finally:
        _log_context.reset(token)


def new_request_id() -> str:
    """Short, collision-resistant id for one request or one job."""
    return uuid.uuid4().hex[:12]


class ContextFilter(logging.Filter):
    """Copy the ambient context dict onto each record as plain attributes.

    Attach this to *handlers*, not loggers, when you want it applied to
    everything including third-party libraries' records: a filter on a logger
    only sees records logged through that logger, while a filter on a handler
    sees every record that reaches the handler, whatever logger produced it.

    `defaults` fills in keys that a %-style format string references
    unconditionally. Without them, a format string containing
    `%(request_id)s` raises a formatting error on any record logged outside a
    request, and logging swallows it into a noisy stderr traceback.
    """

    def __init__(self, defaults: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.defaults = defaults or {}

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self.defaults.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        for key, value in _log_context.get().items():
            setattr(record, key, value)
        return True
