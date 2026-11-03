#!/usr/bin/env python3
"""Regression test for audit_negative_space.py. Runs under pytest or bare python:

    python scripts/test_audit.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_negative_space import audit_file

CASES: list[tuple[str, str, str]] = [
    (
        "NSP001",
        """
def f(a, b):
    c = a + b
    d = c * 2
    return d
""",
        "function with no checks",
    ),
    (
        "NSP002",
        """
def f(x):
    require(x is not None, "x")
    require(x > 0, "x")
    assert (x > 0, "x must be positive")
    return x
""",
        "tuple assert never fires",
    ),
    (
        "NSP003",
        """
def f(queue):
    require(queue is not None, "queue")
    assert queue.pop() is not None
    return queue
""",
        "mutation inside a check",
    ),
    (
        "NSP004",
        """
def f(a, b):
    require(a > 0 and b > 0, "both positive")
    require(a != b, "distinct")
    return a + b
""",
        "compound check",
    ),
    (
        "NSP005",
        """
def f(client):
    require(client is not None, "client")
    require(hasattr(client, "get"), "client.get")
    while True:
        r = client.get()
        if r:
            return r
""",
        "unbounded while True",
    ),
    (
        "NSP006",
        """
def f(path):
    require(path, "path")
    require(isinstance(path, str), "path type")
    try:
        return open(path)
    except:
        raise
""",
        "bare except",
    ),
    (
        "NSP007",
        """
def f(path):
    require(path, "path")
    require(isinstance(path, str), "path type")
    try:
        return open(path)
    except OSError:
        pass
""",
        "swallowed exception",
    ),
    (
        "NSP005",
        """
import itertools

def f(source):
    require(source is not None, "source")
    require(callable(source), "source callable")
    for i in itertools.count():
        if source(i):
            return i
""",
        "for over itertools.count()",
    ),
    (
        "NSP005",
        """
def f(read):
    require(read is not None, "read")
    require(callable(read), "read callable")
    for chunk in iter(read, b""):
        print(chunk)
""",
        "for over iter(callable, sentinel)",
    ),
    (
        "NSP009",
        """
def f(x):
    require(x, "x")
    require(x > 0, "x positive")
    return x  # nsp: ignore NSP005
""",
        "suppression that matches nothing",
    ),
]

CLEAN = """
import logging

log = logging.getLogger(__name__)
MAX = 10


def split(n, frac):
    require(n >= 2, "n")
    require(0.0 < frac < 1.0, "frac")
    k = int(n * frac)
    require(1 <= k <= n - 1, "k")
    return list(range(k)), list(range(k, n))


def poll(source):
    require(source is not None, "source")
    for _ in bounded(range(MAX), MAX):
        try:
            item = source.next()
        except TimeoutError as exc:
            log.warning("poll timed out: %s", exc)
            continue
        if item is not None:
            return item
    require(True, "exhausted")
    return None


def event_loop(handler):
    require(handler is not None, "handler")
    require(callable(handler), "handler callable")
    while True:  # nsp: ignore NSP005
        handler()
"""


def run(source: str, *, strict: bool = False) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.py"
        path.write_text(source, encoding="utf-8")
        return [f.code for f in audit_file(path, min_assertions=2, strict=strict)]


def test_each_code_fires() -> None:
    for code, source, label in CASES:
        codes = run(source)
        assert code in codes, f"{code} ({label}) did not fire; got {codes}"


def test_clean_source_has_no_findings() -> None:
    codes = run(CLEAN)
    assert codes == [], f"expected clean, got {codes}"


def test_strict_flags_plain_assert() -> None:
    source = "def f(x):\n    require(x, 'x')\n    assert x > 0\n    return x\n"
    assert "NSP008" not in run(source, strict=False)
    assert "NSP008" in run(source, strict=True)


def test_suppression_comment() -> None:
    source = (
        "def f(x):\n"
        "    require(x, 'x')\n"
        "    assert (x > 0, 'msg')  # nsp: ignore\n"
        "    return x\n"
    )
    assert "NSP002" not in run(source)


def test_marker_inside_string_is_not_a_suppression() -> None:
    source = 'def f(x):\n    require(x, "x")\n    assert (x, "# nsp: ignore")\n    return x\n'
    codes = run(source)
    assert "NSP002" in codes, f"string literal acted as a suppression: {codes}"


def test_bounded_for_loops_are_clean() -> None:
    source = (
        "import itertools\n"
        "def f(xs):\n"
        "    require(xs, 'xs')\n"
        "    require(len(xs) < 9, 'few')\n"
        "    for _ in itertools.repeat(None, 3):\n"
        "        pass\n"
        "    for _ in itertools.islice(itertools.count(), 5):\n"
        "        pass\n"
        "    return xs\n"
    )
    codes = run(source)
    assert codes == [], f"bounded for-loops flagged: {codes}"


def test_expect_counts_as_a_check() -> None:
    source = (
        "def f(d):\n"
        "    v = expect(d.get('k'), 'k')\n"
        "    w = expect(d.get('j'), 'j')\n"
        "    return v, w\n"
    )
    codes = run(source)
    assert "NSP001" not in codes, f"expect() not counted as a check: {codes}"


def test_auditor_survives_unusual_but_valid_shapes() -> None:
    source = (
        "@deco\n"
        "@deco\n"
        "def twice(x):\n"
        "    require(x, 'x')\n"
        "    require(x > 0, 'x positive')\n"
        "    'a string statement mid-body'\n"
        "    return x\n"
        "\n"
        "def only_a_loop(xs):\n"
        "    for x in xs:\n"
        "        print(x)\n"
    )
    codes = run(source)
    assert codes == ["NSP001"], f"expected only only_a_loop flagged, got {codes}"


def test_test_files_are_exempt_from_density() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test_thing.py"
        path.write_text("def test_x():\n    y = 1\n    z = 2\n    print(y, z)\n", encoding="utf-8")
        assert audit_file(path, min_assertions=2, strict=True) == []


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
