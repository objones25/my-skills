"""Security test patterns, runnable as-is. Copy the pattern, swap in your function.

Each pattern turns an attack class into a regression test: a fixed table of
known payloads (documents what you defend against) plus a property over
generated input (finds the payload you did not think of).

Run: pytest assets/test_security_template.py -p no:randomly
"""

from __future__ import annotations

import json
import pickle
import re
import sqlite3
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

pytestmark = pytest.mark.security


# --------------------------------------------------------------------------
# Toy system under test (replace with your own code)
# --------------------------------------------------------------------------
class RejectedInput(ValueError):
    pass


NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")
MAX_BODY = 4096


def safe_child(base: Path, name: str) -> Path:
    if not NAME.fullmatch(name):
        raise RejectedInput(f"bad name: {name!r}")
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise RejectedInput(f"escapes base: {name!r}")
    return path


def make_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE users (name TEXT PRIMARY KEY, role TEXT)")
    db.executemany("INSERT INTO users VALUES (?, ?)", [("alice", "admin"), ("bob", "viewer")])
    return db


def find_user(db: sqlite3.Connection, name: str):
    return db.execute("SELECT name, role FROM users WHERE name = ?", (name,)).fetchone()


def parse_body(raw: bytes) -> dict:
    if len(raw) > MAX_BODY:
        raise RejectedInput("body too large")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RejectedInput("body is not JSON") from exc
    if not isinstance(data, dict):
        raise RejectedInput("body must be an object")
    return data


PERMISSIONS = {("admin", "read"), ("admin", "write"), ("admin", "delete"), ("viewer", "read")}


def can(role: str, action: str) -> bool:
    return (role, action) in PERMISSIONS


# 1. Path traversal: payload table + generated names.
TRAVERSAL = ["../x", "..", "/etc/passwd", "a/../../x", "a\\..\\x", "x\n", "x\x00", "\uff0e\uff0e", "%2e%2e%2f"]


@pytest.mark.parametrize("name", TRAVERSAL, ids=repr)
def test_traversal_payloads_are_rejected(tmp_path, name):
    with pytest.raises(RejectedInput, match="bad name"):
        safe_child(tmp_path, name)


@given(st.text(max_size=50))
def test_no_generated_name_escapes_the_base(tmp_path_factory, name):
    base = tmp_path_factory.getbasetemp()
    try:
        path = safe_child(base, name)
    except RejectedInput:
        return
    assert path.is_relative_to(base.resolve())


# 2. SQL injection: payloads are data, so the table is untouched and nothing matches.
@pytest.fixture
def db():
    """Closed after each test: on Python 3.13+ an unclosed sqlite3 connection emits a
    ResourceWarning when collected, and under filterwarnings = "error" that fails
    whichever later test happens to be running."""
    conn = make_db()
    yield conn
    conn.close()


@pytest.mark.parametrize("payload", ["' OR '1'='1", "x'; DROP TABLE users; --", "%", "_"], ids=repr)
def test_injection_payloads_match_nothing(db, payload):
    assert find_user(db, payload) is None
    assert db.execute("SELECT count(*) FROM users").fetchone()[0] == 2


# 3. Regex anchors: `$` matches before a trailing newline; fullmatch does not.
def test_dollar_anchor_lets_a_trailing_newline_through():
    assert re.match(r"^[a-z]+$", "alice\n") is not None
    assert NAME.fullmatch("alice\n") is None


# 4. Resource limits: oversized input is rejected before any expensive work.
def test_oversized_body_is_rejected():
    with pytest.raises(RejectedInput, match="too large"):
        parse_body(b"{" + b" " * MAX_BODY + b"}")


# 5. Untrusted bytes are parsed as data, never deserialized as objects.
def test_pickle_payload_is_rejected_as_non_json():
    with pytest.raises(RejectedInput, match="not JSON"):
        parse_body(pickle.dumps({"role": "admin"}))


@given(st.binary(max_size=200))
def test_body_parser_raises_only_rejected_input(raw):
    try:
        data = parse_body(raw)
    except RejectedInput:
        return
    assert isinstance(data, dict)
    assert len(raw) <= MAX_BODY


# 6. Authorization matrix: every role x action pair is decided explicitly.
ROLES, ACTIONS = ("admin", "viewer", "anonymous"), ("read", "write", "delete")
EXPECTED = {
    ("admin", "read"): True, ("admin", "write"): True, ("admin", "delete"): True,
    ("viewer", "read"): True, ("viewer", "write"): False, ("viewer", "delete"): False,
    ("anonymous", "read"): False, ("anonymous", "write"): False, ("anonymous", "delete"): False,
}


def test_matrix_covers_every_pair():
    assert set(EXPECTED) == {(r, a) for r in ROLES for a in ACTIONS}


@pytest.mark.parametrize(("role", "action"), sorted(EXPECTED), ids=lambda v: str(v))
def test_authorization_matrix(role, action):
    assert can(role, action) is EXPECTED[(role, action)]
