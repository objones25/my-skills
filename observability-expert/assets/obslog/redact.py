"""Last-resort redaction of secrets on their way into a log record.

The real rule is "never pass a secret to a log call". This filter exists because
that rule is enforced by humans and humans paste dicts into log lines. It is a
safety net, not a license: it only sees what it can pattern-match, so anything
it catches is a bug to fix at the call site.

Two mechanisms:

1. Key-based: any `extra={...}` field (or context field) whose *name* looks
   secret gets its value replaced. Names are matched word by word, so
   `hf_token` and `apiKey` are redacted but `max_tokens` and `author` are not.
2. Value-based: the formatted message is scanned for a few high-confidence
   token shapes (Bearer tokens, `key=value` pairs with a secret-looking key,
   long base64ish blobs following `sk-`). Regex scanning costs time per record,
   so it is opt-in via `scan_message=True`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

REDACTED = "***REDACTED***"

# Field names are split into words (snake_case, kebab-case, dotted, camelCase)
# and matched word by word. A plain substring search redacts ML config keys
# such as `max_tokens`, `tokenizer`, `tokens_per_sec`, `author` and `passes`,
# which silently corrupts the experiment record it was meant to protect.
SECRET_WORDS = frozenset(
    {
        "password", "passwd", "pass", "pwd", "secret", "token", "apikey",
        "credential", "credentials", "authorization", "auth", "cookie",
        "ssn", "sessionid", "bearer",
    }
)
SECRET_BIGRAMS = frozenset(
    {
        ("api", "key"), ("access", "key"), ("private", "key"),
        ("secret", "key"), ("session", "id"), ("card", "number"),
    }
)
# `token` qualified like this is a vocabulary entry or a count, not a credential.
_TOKEN_NEIGHBORS = frozenset(
    {"pad", "eos", "bos", "unk", "sep", "cls", "mask", "id", "ids", "count", "len", "type"}
)
_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def name_words(name: str) -> list[str]:
    """`hfAPIToken` -> ['hf', 'api', 'token']; `max_tokens` -> ['max', 'tokens']."""
    return [w.lower() for w in _WORD_RE.findall(str(name))]


def is_secret_name(name: str, extra: tuple[str, ...] = ()) -> bool:
    """True if a field/variable name looks like it holds a credential."""
    if str(name).lower() in extra:
        return True
    words = name_words(name)
    for i, word in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else ""
        prev = words[i - 1] if i > 0 else ""
        if (word, nxt) in SECRET_BIGRAMS:
            return True
        if word not in SECRET_WORDS:
            continue
        if word == "token" and (prev in _TOKEN_NEIGHBORS or nxt in _TOKEN_NEIGHBORS):
            continue
        return True
    return False

MESSAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9._\-]{8,}"),
    re.compile(
        r"(?i)\b(pass(word|wd)?|secret|token|api[_-]?key|auth)\b"
        r"\s*[:=]\s*[\"']?[^\s\"',}]{4,}"
    ),
)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (REDACTED if is_secret_name(str(k)) else _redact_value(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(v) for v in value)
    return value


class RedactingFilter(logging.Filter):
    """Redact secret-looking fields on every record that passes through.

    Attach to handlers so it covers third-party libraries too. `extra_keys`
    adds project-specific field names (e.g. "member_id") to the key pattern.
    """

    def __init__(
        self,
        *,
        scan_message: bool = False,
        extra_keys: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.scan_message = scan_message
        self.extra_keys = tuple(k.lower() for k in extra_keys)

    def _is_secret_key(self, key: str) -> bool:
        return is_secret_name(key, self.extra_keys)

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in list(record.__dict__.items()):
            if key.startswith("_"):
                continue
            if self._is_secret_key(key):
                record.__dict__[key] = REDACTED
            elif isinstance(value, (dict, list, tuple)):
                record.__dict__[key] = _redact_value(value)

        if self.scan_message:
            # getMessage() applies the %-args, so this catches secrets that
            # arrived as positional args too. Writing the result back to `msg`
            # and clearing `args` keeps downstream formatters from re-expanding.
            message = record.getMessage()
            redacted = message
            for pattern in MESSAGE_PATTERNS:
                redacted = pattern.sub(REDACTED, redacted)
            if redacted != message:
                record.msg = redacted
                record.args = ()
        return True
