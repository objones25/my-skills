"""A JSON-lines formatter for the stdlib logging module.

The stdlib ships no JSON formatter, so this is the one piece every project ends
up writing. A `logging.Formatter` has exactly one job: turn a `LogRecord` (the
object logging builds for each call, carrying the message plus metadata like
time, level, module, line number, and any `extra={...}` fields) into a string.
Returning JSON instead of a human line is a subclass and one method.

Output is JSON Lines (`.jsonl`): one self-contained JSON object per line. The
file as a whole is not a single JSON document -- parse it line by line.

Why bother: free-text logs are searchable only by substring, and multi-line
tracebacks break line-oriented tooling entirely. With JSON, `request_id` is a
field you can filter and group by in any log backend, and a traceback is one
string inside one record instead of forty orphaned lines.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

# Attributes the logging module itself puts on every LogRecord. Anything on a
# record that is NOT in this set was put there by the caller via `extra={...}`
# (or by a filter), which is exactly what we want to promote into the JSON.
# Verified against Python 3.13 by inspecting a live record's __dict__.
BUILTIN_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",  # 3.12+: asyncio task name
        "thread",
        "threadName",
    }
)

DEFAULT_FIELDS: dict[str, str] = {
    "timestamp": "timestamp",
    "level": "levelname",
    "logger": "name",
    "message": "message",
    "module": "module",
    "line": "lineno",
}


class JSONFormatter(logging.Formatter):
    """Render each record as one JSON object.

    Args:
        fields: output key -> LogRecord attribute name. Two synthetic names are
            supported: "message" (the %-formatted message) and "timestamp"
            (UTC ISO-8601). Everything else is read off the record with getattr.
        static: fields added to every record -- service name, version, env.
            Log aggregators group by these, so set them once here rather than
            passing them at every call site.
    """

    def __init__(
        self,
        *,
        fields: dict[str, str] | None = None,
        static: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.fields = dict(fields) if fields is not None else dict(DEFAULT_FIELDS)
        self.static = dict(static or {})

    def format(self, record: logging.LogRecord) -> str:
        synthetic: dict[str, Any] = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(timespec="milliseconds"),
        }

        out: dict[str, Any] = {}
        for out_key, attr in self.fields.items():
            if attr in synthetic:
                out[out_key] = synthetic[attr]
            else:
                value = getattr(record, attr, None)
                if value is not None:
                    out[out_key] = value

        out.update(self.static)

        # Exception type and message as separate fields: a backend can then
        # group by exception class without regexing the traceback blob.
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, _ = record.exc_info
            out["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": self.formatException(record.exc_info),
            }
        elif record.exc_text:
            out["exception"] = {"traceback": record.exc_text}

        if record.stack_info:
            out["stack_info"] = self.formatStack(record.stack_info)

        # Caller-supplied context: everything `extra={...}` and every filter put
        # on the record. Skipped if it collides with a key already written.
        for key, value in record.__dict__.items():
            if key not in BUILTIN_RECORD_ATTRS and key not in out:
                out[key] = value

        # default=str keeps one unserializable object (a Path, a UUID, a model)
        # from taking down the log call. Losing type fidelity beats losing the
        # record.
        return json.dumps(out, default=str)
