"""mutate_check.py: survivors are reported for weak tests, all killed for strong ones."""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import SKILL_ROOT, load_script

SCRIPT = SKILL_ROOT / "scripts" / "mutate_check.py"
# ---------------------------------------------------------------------------
# Fixtures and helpers for the cases below. They use tiny sources and a plain
# `python check.py` test command (not pytest) so each run takes well under a second.
# ---------------------------------------------------------------------------

CHECK_PRELUDE = "import sys\nsys.dont_write_bytecode = True\ntry:\n    import shop\nexcept Exception:\n    sys.exit(2)\n"

PICK = "def pick(a, b):\n    return a if a > b else b\n\n\ndef twice(x):\n    return x * 2\n"
PICK_CHECK = "assert shop.pick(1, 2) == 2\nassert shop.pick(3, 2) == 3\nassert shop.twice(3) == 6\n"
TINY = "def ok(a, b):\n    a > b\n"  # exactly one mutation site: `>` -> `>=`

# Passes on the original source. On any mutant it records its pid and a grandchild's pid,
# then hangs, so the tests can check timeouts, signals and process-group cleanup.
HANG_ON_MUTANT = (
    "import os, pathlib, subprocess, sys, time\n"
    "if pathlib.Path('shop.py').read_text() == pathlib.Path('orig.txt').read_text():\n"
    "    sys.exit(0)\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "pathlib.Path('grandchild').write_text(str(child.pid))\n"
    "pathlib.Path('started').write_text(str(os.getpid()))\n"
    "print('=== short test summary info ===')\n"
    "print('FAILED tests/t.py::test_hang - timed out', flush=True)\n"
    "time.sleep(60)\n"
)


@pytest.fixture(scope="module")
def mc():
    return load_script("scripts/mutate_check.py")


@pytest.fixture
def proj(tmp_path):
    def make(source: str, check: str, *, prelude: bool = True) -> Path:
        (tmp_path / "shop.py").write_text(source)
        (tmp_path / "orig.txt").write_text(source)
        (tmp_path / "check.py").write_text((CHECK_PRELUDE if prelude else "") + check)
        return tmp_path

    return make


def cli(project, *args, command=None, source="shop.py", timeout=60):
    cmd = [sys.executable, str(SCRIPT), source, *args, "--", *(command or [sys.executable, "check.py"])]
    return subprocess.run(cmd, cwd=project, capture_output=True, text=True, timeout=timeout, check=False)


def cli_json(project, *args, **kw):
    proc = cli(project, "--format", "json", *args, **kw)
    return proc, json.loads(proc.stdout)


def mutants_of(mc, source, kind_prefix=None):
    """Unparsed text of every mutant of `source` whose site kind starts with kind_prefix."""
    tree = ast.parse(source)
    finder = mc.SiteFinder(None)
    finder.visit(tree)
    return [
        ast.unparse(mc.mutate(tree, node_id, kind))
        for node_id, _line, kind, _desc in finder.sites
        if kind_prefix is None or kind.startswith(kind_prefix)
    ]


def sites_of(mc, source, function=None):
    finder = mc.SiteFinder(function)
    finder.visit(ast.parse(source))
    return finder.sites


def unparsed(source):
    return ast.unparse(ast.parse(source))


# ---------------------------------------------------------------------------
# Every swap-table entry: operator in, exact mutated source out.
# ---------------------------------------------------------------------------

CMP_PAIRS = [("<", "<="), ("<=", "<"), (">", ">="), (">=", ">"), ("==", "!="), ("!=", "=="),
             ("is", "is not"), ("is not", "is"), ("in", "not in"), ("not in", "in")]
BINOP_PAIRS = [("+", "-"), ("-", "+"), ("*", "//"), ("//", "*"), ("/", "*"), ("%", "//"), ("**", "*")]


MATMUL_PAIRS = [("@", "*")]
SHIFT_PAIRS = [("<<", ">>"), (">>", "<<")]
BIT_PAIRS = [("|", "&"), ("&", "|"), ("^", "&")]
NEW_OP_CASES = ([("matmul", old, new) for old, new in MATMUL_PAIRS] + [("shift", old, new) for old, new in SHIFT_PAIRS]
                + [("bitop", old, new) for old, new in BIT_PAIRS])


def test_every_table_entry_is_pinned(mc):
    assert len(CMP_PAIRS) == len(mc.COMPARE_SWAPS)
    assert len(BINOP_PAIRS) == len(mc.BINOP_SWAPS)
    assert len(mc.BOOLOP_SWAPS) == 2
    assert mc.NAME_SWAPS == {"max": "min", "min": "max", "any": "all", "all": "any"}
    assert len(mc.MATMUL_SWAPS) == len(MATMUL_PAIRS)
    assert len(mc.SHIFT_SWAPS) == len(SHIFT_PAIRS)
    assert len(mc.BIT_SWAPS) == len(BIT_PAIRS)
    assert len(mc.OP_SWAPS) == len(BINOP_PAIRS) + len(MATMUL_PAIRS) + len(SHIFT_PAIRS) + len(BIT_PAIRS)
    assert mc.KIND_NAMES == ("cmp", "binop", "augassign", "boolop", "not", "bool", "int", "float", "name",
                             "return", "matmul", "shift", "bitop", "neg")


@pytest.mark.parametrize(("old", "new"), CMP_PAIRS)
def test_compare_swap(mc, old, new):
    got = mutants_of(mc, f"def f(a, b):\n    return a {old} b\n", "cmp")
    assert got == [unparsed(f"def f(a, b):\n    return a {new} b\n")]
    (site,) = [s for s in sites_of(mc, f"def f(a, b):\n    return a {old} b\n") if s[2] == "cmp:0"]
    assert site[3] == f"{old} -> {new}"


def test_chained_compare_mutates_one_operator_at_a_time(mc):
    got = mutants_of(mc, "def f(a, b, c):\n    return a < b < c\n", "cmp")
    assert got == [unparsed("def f(a, b, c):\n    return a <= b < c\n"),
                   unparsed("def f(a, b, c):\n    return a < b <= c\n")]


@pytest.mark.parametrize(("old", "new"), BINOP_PAIRS)
def test_binop_swap(mc, old, new):
    src = f"def f(a, b):\n    return a {old} b\n"
    assert mutants_of(mc, src, "binop") == [unparsed(f"def f(a, b):\n    return a {new} b\n")]
    (site,) = [s for s in sites_of(mc, src) if s[2] == "binop"]
    assert site[3] == f"{old} -> {new}"


@pytest.mark.parametrize(("old", "new"), BINOP_PAIRS)
def test_augassign_swap(mc, old, new):
    src = f"def f(a, b):\n    a {old}= b\n    return a\n"
    assert mutants_of(mc, src, "augassign") == [unparsed(f"def f(a, b):\n    a {new}= b\n    return a\n")]
    (site,) = [s for s in sites_of(mc, src) if s[2] == "augassign"]
    assert site[3] == f"{old}= -> {new}="


@pytest.mark.parametrize(("kind", "old", "new"), NEW_OP_CASES)
def test_new_operator_swap_binary_form(mc, kind, old, new):
    src = f"def f(a, b):\n    return a {old} b\n"
    assert mutants_of(mc, src, kind) == [unparsed(f"def f(a, b):\n    return a {new} b\n")]
    assert mutants_of(mc, src, "binop") == [] and mutants_of(mc, src, "augassign") == []
    (site,) = [s for s in sites_of(mc, src) if s[2] == kind]
    assert site[3] == f"{old} -> {new}"


@pytest.mark.parametrize(("kind", "old", "new"), NEW_OP_CASES)
def test_new_operator_swap_augmented_form(mc, kind, old, new):
    src = f"def f(a, b):\n    a {old}= b\n    return a\n"
    assert mutants_of(mc, src, kind) == [unparsed(f"def f(a, b):\n    a {new}= b\n    return a\n")]
    (site,) = [s for s in sites_of(mc, src) if s[2] == kind]
    assert site[3] == f"{old}= -> {new}="


def test_runtime_type_union_is_mutated_but_none_unions_are_not(mc):
    src = "def f(x):\n    return isinstance(x, int | str)\n"
    assert mutants_of(mc, src, "bitop") == [unparsed("def f(x):\n    return isinstance(x, int & str)\n")]


def test_set_union_is_mutated(mc):
    assert mutants_of(mc, "def f(a, b):\n    a |= b\n    return a\n", "bitop") == [
        unparsed("def f(a, b):\n    a &= b\n    return a\n")]


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("def f(a):\n    return -a\n", "def f(a):\n    return a\n"),
        ("def f(a):\n    return -(a + 1)\n", "def f(a):\n    return a + 1\n"),
        ("def f(a):\n    return [-a, 2]\n", "def f(a):\n    return [a, 2]\n"),
    ],
)
def test_unary_minus_is_dropped(mc, src, expected):
    assert mutants_of(mc, src, "neg") == [unparsed(expected)]
    (site,) = [s for s in sites_of(mc, src) if s[2] == "neg"]
    assert site[3] == "drop unary minus"


@pytest.mark.parametrize(("old", "new"), [("0.0", "1.0"), ("1.5", "2.5"), ("2.0", "3.0"), ("-0.5", "-1.5")])
def test_float_constant_gets_one_added(mc, old, new):
    got = mutants_of(mc, f"def f():\n    x = {old}\n    return x\n", "float")
    assert got == [unparsed(f"def f():\n    x = {new}\n    return x\n")]


def test_float_description_shows_both_values(mc):
    (site,) = [s for s in sites_of(mc, "def f():\n    x = 0.5\n    return x\n") if s[2] == "float"]
    assert site[3] == "0.5 -> 1.5"


@pytest.mark.parametrize(("old", "new"), [("and", "or"), ("or", "and")])
def test_boolop_swap(mc, old, new):
    src = f"def f(a, b):\n    return a {old} b\n"
    assert mutants_of(mc, src, "boolop") == [unparsed(f"def f(a, b):\n    return a {new} b\n")]
    (site,) = [s for s in sites_of(mc, src) if s[2] == "boolop"]
    assert site[3] == f"{old} -> {new}"


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("def f(a):\n    return not a\n", "def f(a):\n    return a\n"),
        ("def f(a):\n    if not a:\n        return 1\n", "def f(a):\n    if a:\n        return 1\n"),
        ("def f(a, b):\n    return (not a, b)\n", "def f(a, b):\n    return (a, b)\n"),
    ],
)
def test_not_is_dropped_wherever_it_sits(mc, src, expected):
    assert mutants_of(mc, src, "not") == [unparsed(expected)]


@pytest.mark.parametrize(("old", "new"), [("True", "False"), ("False", "True")])
def test_bool_constant_is_flipped(mc, old, new):
    src = f"def f():\n    x = {old}\n    return x\n"
    assert mutants_of(mc, src, "bool") == [unparsed(f"def f():\n    x = {new}\n    return x\n")]
    (site,) = [s for s in sites_of(mc, src) if s[2] == "bool"]
    assert site[3] == f"{old} -> {new}"


@pytest.mark.parametrize(("old", "new"), [(0, 1), (5, 6), (100, 101)])
def test_int_constant_is_incremented(mc, old, new):
    got = mutants_of(mc, f"def f():\n    x = {old}\n    return x\n", "int")
    assert got == [unparsed(f"def f():\n    x = {new}\n    return x\n")]


@pytest.mark.parametrize(("old", "new"), [("max", "min"), ("min", "max"), ("any", "all"), ("all", "any")])
def test_builtin_name_swap(mc, old, new):
    src = f"def f(xs):\n    return {old}(xs)\n"
    assert mutants_of(mc, src, "name") == [unparsed(f"def f(xs):\n    return {new}(xs)\n")]
    (site,) = [s for s in sites_of(mc, src) if s[2] == "name"]
    assert site[3] == f"{old}() -> {new}()"


def test_return_value_becomes_none(mc):
    got = mutants_of(mc, "def f(a):\n    return a\n", "return")
    assert got == [unparsed("def f(a):\n    return None\n")]


@pytest.mark.parametrize(
    ("src", "kinds"),
    [
        ("def f(a):\n    return None\n", set()),  # already None: an equivalent mutant
        ("def f(a):\n    return\n", set()),
        ("def f(a):\n    return ~a\n", {"return"}),  # invert and unary plus are not mutated
        ("def f(a):\n    return +a\n", {"return"}),
        ("def f(a, b):\n    return a - b\n", {"return", "binop"}),  # binary minus is not unary minus
        ("def f(a):\n    return a | None\n", {"return"}),  # `X | None` can only be a type union
        ("def f(a):\n    return None | a\n", {"return"}),
        ("def f(a: int | str) -> int | None:\n    return a\n", {"return"}),  # annotations are never mutated
        ("def f(a: Literal[1.5, -2] = None) -> float:\n    x: Literal[2.5, -1] = 0\n    return x\n", {"return", "int"}),
        ("def f():\n    return 1e300\n", {"return"}),  # c + 1.0 == c: an equivalent mutant
        ("def f():\n    return float('inf')\n", {"return"}),
        ("def f():\n    1.5\n    return None\n", set()),  # bare constant statement
        ("def f(x):\n    return x.max()\n", {"return"}),  # attribute call, not the builtin
        ("def f(xs):\n    return sum(xs)\n", {"return"}),
        ("def f():\n    return 'text'\n", {"return"}),  # strings are never mutated
    ],
)
def test_near_misses_make_no_site_of_the_named_kind(mc, src, kinds):
    assert {s[2] for s in sites_of(mc, src)} == kinds


# ---------------------------------------------------------------------------
# What is never mutated: docstrings, annotations, bare constant statements.
# ---------------------------------------------------------------------------


def test_docstrings_are_never_mutated(mc):
    src = (
        '"""Module doc: 5 >= 3 and 7 + 1, True."""\n\n\n'
        'class C:\n    """Class doc 1 + 2."""\n\n    def m(self):\n        """Method doc: x > 3, not y, 9."""\n\n\n'
        'async def g():\n    """Async doc 4 == 4."""\n'
    )
    assert sites_of(mc, src) == []


def test_annotations_are_never_mutated(mc):
    src = (
        "def f(a: Literal[3] = 4, *, b: Annotated[int, 1 + 2] = 5) -> Literal[1, 2]:\n"
        "    y: Literal[7] = 8\n"
        "    return y\n"
    )
    got = sorted(s[3] for s in sites_of(mc, src))
    assert got == ["4 -> 5", "5 -> 6", "8 -> 9", "return value -> return None"]


def test_bare_constant_statements_are_not_mutated(mc):
    src = "def f():\n    42\n    True\n    ...\n    return 1\n"
    assert [s[3] for s in sites_of(mc, src)] == ["return value -> return None", "1 -> 2"]


def test_sites_are_four_field_tuples(mc):
    (site,) = [s for s in sites_of(mc, "def f(a):\n    return a > 1\n") if s[2] == "cmp:0"]
    assert len(site) == 4
    assert isinstance(site[0], int) and site[1] == 2 and site[2] == "cmp:0" and site[3] == "> -> >="
    assert "(node id, line, kind, description)" in mc.SiteFinder.__doc__


def test_module_level_code_is_not_mutated(mc):
    assert sites_of(mc, "LIMIT = 5\nflag = 3 > 2\n") == []


# ---------------------------------------------------------------------------
# --function scoping.
# ---------------------------------------------------------------------------

SCOPED = (
    "class C:\n"                       # 1
    "    def method(self, a):\n"       # 2
    "        def inner(b):\n"          # 3
    "            return b > 1\n"       # 4
    "        return a > 2\n"           # 5
    "\n"
    "    def other(self, a):\n"        # 7
    "        return a > 3\n"           # 8
    "\n"
    "def outer(a):\n"                  # 10
    "    def closure(b):\n"            # 11
    "        return b > 4\n"           # 12
    "    return a > 5\n"               # 13
)


def _cmp_lines(mc, function):
    return sorted(s[1] for s in sites_of(mc, SCOPED, function) if s[2] == "cmp:0")


@pytest.mark.parametrize(
    ("function", "lines"),
    [
        (None, [4, 5, 8, 12, 13]),
        ("method", [4, 5]),          # bare method name includes its closures
        ("C.method", [4, 5]),        # dotted form agrees with the bare form
        ("inner", [4]),
        ("C.method.inner", [4]),
        ("other", [8]),
        ("outer", [12, 13]),         # top-level bare name includes closures
        ("closure", [12]),
        ("C", [4, 5, 8]),
        ("nope", []),
    ],
)
def test_function_scope(mc, function, lines):
    assert _cmp_lines(mc, function) == lines


def test_function_scope_cli_limits_mutants_to_that_function(proj):
    project = proj(PICK, PICK_CHECK)
    proc, payload = cli_json(project, "--function", "twice")
    assert proc.returncode == 0, proc.stderr
    assert [m["line"] for m in payload["mutants"]] == [6, 6, 6]
    assert payload["score"] == pytest.approx(100.0)


def test_function_with_no_match_exits_2(proj):
    project = proj(PICK, PICK_CHECK)
    proc = cli(project, "--function", "nope")
    assert proc.returncode == 2
    assert "no mutation sites" in proc.stderr


# ---------------------------------------------------------------------------
# Kill / survivor arithmetic, JSON fields, --max-survivors.
# ---------------------------------------------------------------------------


def test_text_summary_counts_killed_of_total(proj):
    project = proj(PICK, PICK_CHECK)
    proc = cli(project)
    assert proc.returncode == 1
    assert "4/5 mutants killed (80%)" in proc.stdout
    assert proc.stdout.count("SURVIVED line 2: > -> >=") == 1
    assert proc.stdout.count("SURVIVED") == 2  # the per-mutant line and the survivor list


def test_json_reports_every_field_of_every_mutant(proj):
    project = proj(PICK, PICK_CHECK)
    proc, payload = cli_json(project)
    assert proc.returncode == 1
    assert set(payload) == {"source", "score", "killed", "survived", "errored", "timeouts",
                            "baseline_seconds", "timeout_seconds", "mutants"}
    assert all(m["killed_by"] == [] for m in payload["mutants"])  # a plain script prints no test ids
    assert (payload["killed"], payload["survived"], payload["errored"], payload["timeouts"]) == (4, 1, 0, 0)
    assert payload["score"] == pytest.approx(80.0)
    assert payload["source"] == "shop.py"
    expected = [
        (1, 2, "return value -> return None", "killed", 1),
        (2, 2, "> -> >=", "SURVIVED", 0),
        (3, 6, "return value -> return None", "killed", 1),
        (4, 6, "* -> //", "killed", 1),
        (5, 6, "2 -> 3", "killed", 1),
    ]
    assert [(m["index"], m["line"], m["description"], m["status"], m["exit_code"]) for m in payload["mutants"]] == expected
    mutant_fields = {"index", "line", "description", "status", "seconds", "exit_code", "killed_by"}
    assert [set(m) for m in payload["mutants"]] == [mutant_fields] * len(expected)
    assert [isinstance(m["seconds"], float) and m["seconds"] > 0.0 for m in payload["mutants"]] == [True] * len(expected)


def test_score_is_rounded_to_one_decimal(proj):
    project = proj("def f(x):\n    return x + 1\n", "assert shop.f(0) is not None\n")
    _proc, payload = cli_json(project)
    assert (payload["killed"], payload["survived"]) == (1, 2)
    assert payload["score"] == pytest.approx(33.3)
    assert "1/3 mutants killed (33%)" in cli(project).stdout


@pytest.mark.parametrize(("limit", "code"), [(None, 1), ("0", 1), ("1", 1), ("2", 0), ("3", 0)])
def test_max_survivors_threshold(proj, limit, code):
    project = proj("def f(x):\n    return x + 1\n", "assert shop.f(0) is not None\n")  # two survivors
    proc = cli(project, *(["--max-survivors", limit] if limit is not None else []))
    assert proc.returncode == code


def test_one_survivor_exits_1_by_default_and_0_when_allowed(proj):
    project = proj(PICK, PICK_CHECK)  # exactly one survivor
    assert cli(project).returncode == 1
    assert cli(project, "--max-survivors", "0").returncode == 1
    assert cli(project, "--max-survivors", "1").returncode == 0


# ---------------------------------------------------------------------------
# Errored mutants are not kills.
# ---------------------------------------------------------------------------

BANNER = 'def banner(sep="-" * 3):\n    return sep\n\n\ndef inc(x):\n    return x + 1\n'


def test_import_time_crash_is_reported_as_errored_not_killed(project):
    # `"-" * 3` -> `"-" // 3` raises TypeError when the module is imported, so pytest
    # reports a collection error (exit 2) although no test touches banner().
    (project / "shop.py").write_text(BANNER)
    (project / "tests" / "test_shop.py").write_text("from shop import inc\n\n\ndef test_inc():\n    assert inc(1) == 2\n")
    proc = run(project, "--format", "json")
    payload = json.loads(proc.stdout)
    got = [(m["index"], m["description"], m["status"], m["exit_code"]) for m in payload["mutants"]]
    assert got == [
        (1, "* -> //", "errored (exit 2)", 2),
        (2, "3 -> 4", "SURVIVED", 0),
        (3, "return value -> return None", "SURVIVED", 0),
        (4, "return value -> return None", "killed", 1),
        (5, "+ -> -", "killed", 1),
        (6, "1 -> 2", "killed", 1),
    ]
    assert (payload["killed"], payload["survived"], payload["errored"]) == (3, 2, 1)
    assert payload["score"] == pytest.approx(60.0)  # 3 / (3 + 2); the errored mutant is excluded
    assert proc.returncode == 1


def test_text_output_names_errored_mutants_and_excludes_them(proj):
    project = proj(BANNER, "assert shop.inc(1) == 2\nassert shop.banner() == '---'\n")
    proc = cli(project)
    assert proc.returncode == 0, proc.stdout
    assert "errored (exit 2)" in proc.stdout
    assert "1 mutant(s) errored" in proc.stdout
    assert "excluded from the score" in proc.stdout
    assert "5/5 mutants killed (100%)" in proc.stdout


def test_exit_codes_outside_kill_codes_are_errored(proj):
    check = "import pathlib, sys\nsys.exit(0 if pathlib.Path('shop.py').read_text() == pathlib.Path('orig.txt').read_text() else 3)\n"
    project = proj(TINY, check, prelude=False)
    proc = cli(project)
    assert proc.returncode == 2
    assert "no mutant could be evaluated" in proc.stderr
    assert "--kill-codes" in proc.stderr
    assert (project / "shop.py").read_text() == TINY
    proc, payload = cli_json(project, "--kill-codes", "3")
    assert proc.returncode == 0
    assert [m["status"] for m in payload["mutants"]] == ["killed"]
    assert payload["errored"] == 0


def test_a_mutant_killed_by_a_signal_is_errored_not_killed(proj):
    check = ("import os, pathlib, signal\n"
             "if pathlib.Path('shop.py').read_text() != pathlib.Path('orig.txt').read_text():\n"
             "    os.kill(os.getpid(), signal.SIGKILL)\n")
    project = proj(TINY, check, prelude=False)
    proc = cli(project)
    assert proc.returncode == 2
    assert "errored (exit -9)" in proc.stdout


def test_kill_codes_must_be_integers(proj):
    project = proj(TINY, "")
    proc = cli(project, "--kill-codes", "one")
    assert proc.returncode == 2


# ---------------------------------------------------------------------------
# Timeouts: they count as kills, are labelled, take the whole process group down.
# ---------------------------------------------------------------------------


def _dead(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_timeout_counts_as_a_labelled_kill_and_kills_grandchildren(proj):
    project = proj(TINY, HANG_ON_MUTANT, prelude=False)
    proc, payload = cli_json(project, "--timeout-floor", "1", "--timeout-factor", "1")
    assert proc.returncode == 0, proc.stderr
    (mutant,) = payload["mutants"]
    assert (mutant["status"], mutant["exit_code"]) == ("killed (timeout)", None)
    assert mutant["killed_by"] == []  # ids are only read for a plain "killed"
    assert (payload["killed"], payload["survived"], payload["timeouts"]) == (1, 0, 1)
    assert payload["score"] == pytest.approx(100.0)
    assert payload["timeout_seconds"] == pytest.approx(1.0)
    assert 1.0 <= mutant["seconds"] < 10.0  # the time until the timeout fired
    assert _dead(int((project / "grandchild").read_text()))
    text = cli(project, "--timeout-floor", "1", "--timeout-factor", "1")
    assert "1 of the kills were timeouts" in text.stdout


def test_timeout_factor_scales_the_mutant_timeout(proj):
    check = "import time\ntime.sleep(0.4)\n"  # baseline ~0.4 s, every mutant survives
    project = proj(TINY, check, prelude=False)
    proc, payload = cli_json(project, "--timeout-floor", "0", "--timeout-factor", "0.1")
    assert payload["mutants"][0]["status"] == "killed (timeout)"
    assert payload["timeout_seconds"] == pytest.approx(payload["baseline_seconds"] * 0.1)
    assert proc.returncode == 0
    proc, payload = cli_json(project, "--timeout-floor", "0", "--timeout-factor", "20")
    assert payload["mutants"][0]["status"] == "SURVIVED"
    assert proc.returncode == 1


def test_timeout_floor_is_a_minimum_and_is_reported(proj):
    project = proj(TINY, "")
    _, payload = cli_json(project, "--timeout-floor", "30", "--timeout-factor", "0.01")
    assert payload["timeout_seconds"] == pytest.approx(30.0)
    assert "mutant timeout 30.0s" in cli(project, "--timeout-floor", "30").stdout
    _, default = cli_json(project)
    assert default["timeout_seconds"] == pytest.approx(2.0)  # the documented default floor (10 s before 2026-11-24)


# ---------------------------------------------------------------------------
# Restore after a signal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_restores_source_and_stops_the_test_command(proj, sig):
    project = proj(TINY, HANG_ON_MUTANT, prelude=False)
    before = (project / "shop.py").read_bytes()
    child = subprocess.Popen([sys.executable, str(SCRIPT), "shop.py", "--", sys.executable, "check.py"],
                             cwd=project, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        while not (project / "grandchild").exists() or not (project / "started").exists():  # audit: ignore[BRANCHING] bounded poll: the mutant subprocess signals readiness only through files
            assert time.monotonic() < deadline and child.poll() is None, "mutant run never started"
            time.sleep(0.05)  # audit: ignore[SLEEP] polling interval while waiting for an external process; no event exists to await
        assert (project / "shop.py").read_bytes() != before  # a mutant is on disk right now
        child.send_signal(sig)
        child.communicate(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    assert child.returncode == 130
    assert (project / "shop.py").read_bytes() == before
    assert list(project.glob("*.mutate_check.bak")) == []
    assert _dead(int((project / "grandchild").read_text()))


# ---------------------------------------------------------------------------
# Setup errors exit 2 with a one-line message (never a traceback, never 1).
# ---------------------------------------------------------------------------


def _one_line_error(proc):
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    lines = proc.stderr.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("mutate_check:"), proc.stderr
    return lines[0]


def test_missing_test_command_executable_exits_2(proj):
    project = proj(PICK, PICK_CHECK)
    msg = _one_line_error(cli(project, command=["/nonexistent/definitely-not-here"]))
    assert "/nonexistent/definitely-not-here" in msg
    assert msg.endswith(": No such file or directory")
    assert (project / "shop.py").read_text() == PICK
    assert list(project.glob("*.bak")) == []


def test_unparseable_source_exits_2(proj):
    project = proj("def broken(:\n    pass\n", "")
    msg = _one_line_error(cli(project))
    assert "shop.py" in msg and "line 1" in msg


def test_invalid_utf8_source_exits_2(proj):
    project = proj(PICK, "")
    (project / "shop.py").write_bytes(b"def f():\n    return '\xff\xfe'\n")
    msg = _one_line_error(cli(project))
    assert "UTF-8" in msg


def test_source_with_null_byte_exits_2(proj):
    project = proj(PICK, "")
    (project / "shop.py").write_bytes(b"def f():\n    return 1\x00\n")
    assert "shop.py" in _one_line_error(cli(project))


def test_flaky_baseline_exits_2_and_never_touches_the_source(proj):
    check = ("import pathlib, sys\np = pathlib.Path('runs')\n"
             "n = int(p.read_text()) if p.exists() else 0\np.write_text(str(n + 1))\n"
             "sys.exit(0 if n == 0 else 1)\n")
    project = proj(PICK, check, prelude=False)
    msg = _one_line_error(cli(project))
    assert "flaky" in msg
    assert (project / "runs").read_text() == "2"  # the baseline ran exactly twice
    assert (project / "shop.py").read_text() == PICK


def test_failing_baseline_exits_2_with_its_status(proj):
    project = proj(PICK, "import sys\nsys.exit(1)\n", prelude=False)
    msg = _one_line_error(cli(project))
    assert "did not pass (failed)" in msg
    assert (project / "shop.py").read_text() == PICK


def test_mutant_timeout_is_based_on_the_slower_baseline_run(proj):
    check = ("import pathlib, time\np = pathlib.Path('runs')\nn = int(p.read_text()) if p.exists() else 0\n"
             "p.write_text(str(n + 1))\nif n == 0:\n    time.sleep(0.6)\n")
    project = proj(TINY, check, prelude=False)
    _, payload = cli_json(project, "--timeout-floor", "0")
    assert payload["baseline_seconds"] >= 0.5
    assert payload["timeout_seconds"] == pytest.approx(payload["baseline_seconds"] * 5.0)


def test_command_that_vanishes_after_the_baseline_exits_2_and_restores(proj):
    project = proj(TINY, "")
    runner = project / "runner"
    runner.write_text("#!/bin/sh\n[ -e one ] && rm -- \"$0\"\ntouch one\n")
    runner.chmod(0o755)
    msg = _one_line_error(cli(project, command=[str(runner)]))
    assert msg.endswith(": No such file or directory")
    assert (project / "shop.py").read_text() == TINY
    assert list(project.glob("*.mutate_check.bak")) == []


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root can read a mode-000 file")
def test_unreadable_source_exits_2(proj):
    project = proj(PICK, "")
    (project / "shop.py").chmod(0)
    msg = _one_line_error(cli(project))
    assert "cannot read" in msg


def test_equal_numbers_of_kills_and_survivors_is_not_nothing_evaluated(proj):
    project = proj(PICK, PICK_CHECK)
    proc, payload = cli_json(project, "--function", "pick")
    assert proc.returncode == 1
    assert (payload["killed"], payload["survived"]) == (1, 1)
    assert payload["score"] == pytest.approx(50.0)


def test_baseline_runs_twice_before_mutating(proj):
    check = "import pathlib\np = pathlib.Path('runs')\np.write_text(str(int(p.read_text()) + 1 if p.exists() else 1))\n"
    project = proj(TINY, check, prelude=False)
    proc = cli(project)
    assert proc.returncode == 1  # the one mutant survives
    assert (project / "runs").read_text() == "3"  # two baseline runs + one mutant


def test_usage_without_separator_exits_2(proj):
    project = proj(PICK, PICK_CHECK)
    proc = subprocess.run([sys.executable, str(SCRIPT), "shop.py"], cwd=project, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "usage: mutate_check.py" in proc.stderr


def test_missing_source_exits_2(proj):
    project = proj(PICK, PICK_CHECK)
    proc = cli(project, source="nope.py")
    assert _one_line_error(proc)


def test_empty_test_command_exits_2(proj):
    project = proj(PICK, PICK_CHECK)
    proc = subprocess.run([sys.executable, str(SCRIPT), "shop.py", "--"], cwd=project, capture_output=True, text=True, check=False)
    assert _one_line_error(proc)


def test_stale_backup_exits_2_and_is_left_alone(proj):
    project = proj(PICK, PICK_CHECK)
    stale = project / "shop.py.mutate_check.bak"
    stale.write_text("precious")
    msg = _one_line_error(cli(project))
    assert "interrupted run" in msg
    assert stale.read_text() == "precious"
    assert (project / "shop.py").read_text() == PICK


def test_source_without_any_site_exits_2(proj):
    project = proj("def f():\n    return None\n", "")
    msg = _one_line_error(cli(project))
    assert "no mutation sites" in msg


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_works_without_the_separator_and_exits_0(flag):
    proc = subprocess.run([sys.executable, str(SCRIPT), flag], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    options = ("--function", "--max-survivors", "--timeout-factor", "--timeout-floor", "--kill-codes")
    assert [text for text in options if text not in proc.stdout] == []
    assert "at least" in proc.stdout  # the mutant-timeout floor is documented


# ---------------------------------------------------------------------------
# The new operator kinds: a weak test leaves the mutant alive, a strong test kills it.
# Weak checks use values for which the mutated operator gives the same result.
# ---------------------------------------------------------------------------

VEC = ("class V:\n    def __init__(self, n):\n        self.n = n\n\n"
       "    def __matmul__(self, o):\n        return self.n + o.n\n\n"
       "    def __mul__(self, o):\n        return self.n * o.n\n\n\n")

KIND_CASES = [
    # id, kind, source, weak check, strong check, description
    ("matmul", "matmul", VEC + "def f(a, b):\n    return a @ b\n",
     "assert shop.f(shop.V(2), shop.V(2)) == 4\n", "assert shop.f(shop.V(2), shop.V(3)) == 5\n", "@ -> *"),
    ("matmul-aug", "matmul", VEC + "def f(a, b):\n    a @= b\n    return a\n",
     "assert shop.f(shop.V(2), shop.V(2)) == 4\n", "assert shop.f(shop.V(2), shop.V(3)) == 5\n", "@= -> *="),
    ("shift", "shift", "def f(a, b):\n    return a << b\n",
     "assert shop.f(0, 3) == 0\n", "assert shop.f(1, 3) == 8\n", "<< -> >>"),
    ("shift-aug", "shift", "def f(a, b):\n    a >>= b\n    return a\n",
     "assert shop.f(0, 3) == 0\n", "assert shop.f(8, 3) == 1\n", ">>= -> <<="),
    ("bitor", "bitop", "def f(a, b):\n    return a | b\n",
     "assert shop.f(5, 5) == 5\n", "assert shop.f(4, 1) == 5\n", "| -> &"),
    ("bitor-aug", "bitop", "def f(a, b):\n    a |= b\n    return a\n",
     "assert shop.f(5, 5) == 5\n", "assert shop.f(4, 1) == 5\n", "|= -> &="),
    ("bitand", "bitop", "def f(a, b):\n    return a & b\n",
     "assert shop.f(5, 5) == 5\n", "assert shop.f(6, 3) == 2\n", "& -> |"),
    ("bitxor", "bitop", "def f(a, b):\n    return a ^ b\n",
     "assert shop.f(0, 0) == 0\n", "assert shop.f(6, 3) == 5\n", "^ -> &"),
    ("neg", "neg", "def f(a):\n    return -a\n",
     "assert shop.f(0) == 0\n", "assert shop.f(3) == -3\n", "drop unary minus"),
    ("float", "float", "def f():\n    return 2.5\n",
     "assert shop.f() > 0\n", "assert shop.f() == 2.5\n", "2.5 -> 3.5"),
]


@pytest.mark.parametrize(("kind", "source", "weak", "strong", "description"), [c[1:] for c in KIND_CASES],
                         ids=[c[0] for c in KIND_CASES])
def test_new_kind_weak_test_survives_and_strong_test_kills(proj, kind, source, weak, strong, description):
    project = proj(source, weak)
    proc, payload = cli_json(project, "--only-kinds", kind)
    assert proc.returncode == 1, proc.stderr
    assert [(m["description"], m["status"]) for m in payload["mutants"]] == [(description, "SURVIVED")]
    (project / "check.py").write_text(CHECK_PRELUDE + strong)
    proc, payload = cli_json(project, "--only-kinds", kind)
    assert proc.returncode == 0, proc.stderr
    assert [(m["description"], m["status"]) for m in payload["mutants"]] == [(description, "killed")]


# ---------------------------------------------------------------------------
# --skip-kinds / --only-kinds.
# ---------------------------------------------------------------------------


def _descriptions(payload):
    return [m["description"] for m in payload["mutants"]]


def test_only_kinds_limits_the_mutants(proj):
    project = proj(PICK, PICK_CHECK)
    proc, payload = cli_json(project, "--only-kinds", "cmp")
    assert proc.returncode == 1 and _descriptions(payload) == ["> -> >="]
    _, payload = cli_json(project, "--only-kinds", "binop, int")
    assert _descriptions(payload) == ["* -> //", "2 -> 3"]


def test_skip_kinds_drops_the_named_kinds(proj):
    project = proj(PICK, PICK_CHECK)
    _, payload = cli_json(project, "--skip-kinds", "return")
    assert _descriptions(payload) == ["> -> >=", "* -> //", "2 -> 3"]
    _, payload = cli_json(project, "--skip-kinds", "cmp,return,int")
    assert _descriptions(payload) == ["* -> //"]


@pytest.mark.parametrize("flag", ["--skip-kinds", "--only-kinds"])
def test_unknown_kind_exits_2_and_lists_the_known_kinds(proj, mc, flag):
    project = proj(PICK, PICK_CHECK)
    proc = cli(project, flag, "cmp,bogus")
    assert proc.returncode == 2
    assert "unknown kind 'bogus'" in proc.stderr
    assert all(name in proc.stderr for name in mc.KIND_NAMES)
    assert (project / "shop.py").read_text() == PICK


def test_skip_and_only_kinds_are_mutually_exclusive(proj):
    project = proj(PICK, PICK_CHECK)
    assert cli(project, "--skip-kinds", "cmp", "--only-kinds", "int").returncode == 2


def test_filtering_out_every_site_exits_2(proj):
    project = proj(PICK, PICK_CHECK)
    proc = cli(project, "--only-kinds", "neg")
    assert _one_line_error(proc).endswith("--skip-kinds, --only-kinds)")


def test_every_kind_is_documented_in_the_docstring_and_help(mc):
    help_text = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False).stdout
    assert [name for name in mc.KIND_NAMES if not re.search(rf"^    {name} ", mc.__doc__, re.MULTILINE)] == []
    assert [name for name in mc.KIND_NAMES if name not in help_text] == []


# ---------------------------------------------------------------------------
# Which test killed each mutant.
# ---------------------------------------------------------------------------

SUMMARY_SCRIPT = (
    "import pathlib, sys\n"
    "if pathlib.Path('shop.py').read_text() == pathlib.Path('orig.txt').read_text():\n"
    "    sys.exit(0)\n"
    "print('FAILED tests/noise.py::test_before_summary - printed by a test')\n"
    "print('=' * 20 + ' short test summary info ' + '=' * 20)\n"
    "{body}"
    "sys.exit({code})\n"
)


def _summary_check(body_lines, code=1):
    body = "".join(f"print({line!r})\n" for line in body_lines)
    return SUMMARY_SCRIPT.format(body=body, code=code)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("=== short test summary info ===\nFAILED tests/t.py::test_a - assert 1 == 2\n1 failed in 0.01s\n",
         ["tests/t.py::test_a"]),
        ("=== short test summary info ===\nFAILED t.py::TestK::test_p[2] - assert 2 == 1\nERROR t.py::test_e - RuntimeError: x\n",
         ["t.py::TestK::test_p[2]", "t.py::test_e"]),
        ("=== short test summary info ===\nFAILED t.py::test_x[a b] - msg\n", ["t.py::test_x[a b]"]),
        ("=== short test summary info ===\nFAILED t.py::test_nomsg\n", ["t.py::test_nomsg"]),
        ("=== short test summary info ===\nERROR test_bad.py - TypeError: boom\n", ["test_bad.py"]),
        ("=== short test summary info ===\nFAILED a.py::t - m\nFAILED a.py::t - again\n", ["a.py::t"]),
        ("\x1b[31m=== short test summary info ===\x1b[0m\nFAILED a.py::t - m\n", ["a.py::t"]),
        ("FAIL: test_a (mod.C.test_a)\nERROR: test_b (mod.C.test_b)\n", ["test_a (mod.C.test_a)", "test_b (mod.C.test_b)"]),
        # a marker needs both the leading "=" and the words; neither alone opens the summary
        ("the short test summary info is below\nFAILED t.py::test_x - m\n", []),
        ("=== 1 failed ===\nFAILED t.py::test_x - m\n", []),
        # not test ids: before the summary, usage errors, and free text
        ("FAILED t.py::test_printed_by_a_test - x\n", []),
        ("ERROR: usage: python -m pytest [options] [file_or_dir]\n", []),
        ("1 failed in 0.01s\n", []),
        ("", []),
    ],
)
def test_failing_test_id_extraction(mc, output, expected):
    assert mc.failing_test_ids(output) == expected


def test_each_mutant_reports_exactly_the_test_that_killed_it(project):
    (project / "shop.py").write_text("def inc(x):\n    return x + 1\n\n\ndef dec(x):\n    return x - 1\n")
    (project / "tests" / "test_shop.py").write_text(
        "from shop import dec, inc\n\n\ndef test_inc():\n    assert inc(1) == 2\n\n\n"
        "def test_dec():\n    assert dec(1) == 0\n")
    proc = run(project, "--only-kinds", "binop", "--format", "json")
    payload = json.loads(proc.stdout)
    assert [(m["description"], m["status"], m["killed_by"]) for m in payload["mutants"]] == [
        ("+ -> -", "killed", ["tests/test_shop.py::test_inc"]),
        ("- -> +", "killed", ["tests/test_shop.py::test_dec"]),
    ]
    text = run(project, "--only-kinds", "binop").stdout
    assert "+ -> -" in text and "killed   by tests/test_shop.py::test_inc\n" in text
    assert "killed   by tests/test_shop.py::test_dec\n" in text


def test_text_output_caps_the_list_of_killing_tests(proj):
    ids = [f"tests/t.py::test_{i}" for i in range(5)]
    project = proj(TINY, _summary_check([f"FAILED {i} - assert" for i in ids]), prelude=False)
    proc = cli(project)
    assert "killed   by tests/t.py::test_0, tests/t.py::test_1, tests/t.py::test_2 (+2 more)\n" in proc.stdout
    assert "test_3" not in proc.stdout and "noise" not in proc.stdout
    _, payload = cli_json(project)
    assert payload["mutants"][0]["killed_by"] == ids  # JSON keeps the whole list, nothing from before the summary


def test_one_id_over_the_cap_says_plus_one_more(proj):
    ids = [f"tests/t.py::test_{i}" for i in range(4)]
    project = proj(TINY, _summary_check([f"FAILED {i} - assert" for i in ids]), prelude=False)
    assert "tests/t.py::test_2 (+1 more)\n" in cli(project).stdout


def test_text_lines_have_exactly_the_documented_layout(proj):
    project = proj(PICK, PICK_CHECK)
    lines = cli(project, "--only-kinds", "cmp,int").stdout.splitlines()
    assert lines[:2] == ["  #1   line 2    > -> >=                          SURVIVED",
                         "  #2   line 6    2 -> 3                           killed"]


def test_exactly_three_ids_have_no_more_suffix(proj):
    ids = [f"tests/t.py::test_{i}" for i in range(3)]
    project = proj(TINY, _summary_check([f"FAILED {i} - assert" for i in ids]), prelude=False)
    assert "more)" not in cli(project).stdout


def test_output_without_test_ids_leaves_killed_by_empty_and_status_unchanged(proj):
    project = proj(TINY, "import pathlib, sys\nprint('boom: something failed')\n"
                   "sys.exit(0 if pathlib.Path('shop.py').read_text() == pathlib.Path('orig.txt').read_text() else 1)\n",
                   prelude=False)
    proc, payload = cli_json(project)
    assert proc.returncode == 0
    assert (payload["mutants"][0]["status"], payload["mutants"][0]["killed_by"]) == ("killed", [])
    assert "   by " not in cli(project).stdout


def test_printed_test_ids_never_change_the_classification(proj):
    # a summary line with an exit status of 0 is still a survivor; with exit 3 still errored
    survivor = proj(TINY, "print('=== short test summary info ===')\nprint('FAILED t.py::test_a - x')\n", prelude=False)
    _, payload = cli_json(survivor)
    assert (payload["mutants"][0]["status"], payload["mutants"][0]["killed_by"]) == ("SURVIVED", [])
    project = proj(TINY, _summary_check(["FAILED t.py::test_a - x"], code=3), prelude=False)
    proc = cli(project)
    assert "errored (exit 3)" in proc.stdout and "   by " not in proc.stdout


def test_large_output_does_not_hang_and_keeps_only_a_tail(mc):
    code = ("import sys\nfor i in range(60000):\n    sys.stdout.write('x' * 60 + str(i) + '\\n')\n"
            "    sys.stderr.write('e' * 60 + '\\n')\nprint('=== short test summary info ===')\n"
            "print('FAILED t.py::test_last - boom')\nsys.exit(1)\n")
    result = mc.run([sys.executable, "-c", code], timeout=60)
    assert (result.status, result.returncode) == ("failed", 1)
    assert len(result.output.encode()) <= mc.OUTPUT_TAIL_BYTES  # ~7 MB was written
    assert mc.failing_test_ids(result.output) == ["t.py::test_last"]


def test_large_output_end_to_end(proj):
    check = ("import pathlib, sys\n"
             "if pathlib.Path('shop.py').read_text() == pathlib.Path('orig.txt').read_text():\n    sys.exit(0)\n"
             "sys.stdout.write(('y' * 99 + '\\n') * 50000)\nsys.stdout.flush()\n"
             "print('=== short test summary info ===')\nprint('FAILED t.py::test_big - boom')\nsys.exit(1)\n")
    project = proj(TINY, check, prelude=False)
    proc, payload = cli_json(project, timeout=60)
    assert proc.returncode == 0
    assert payload["mutants"][0]["killed_by"] == ["t.py::test_big"]


TWO_SITES = "def ok(a, b):\n    a > b\n    a < b\n"  # the first mutant errors, the second is killed


def _errored_first(body):
    return ("import pathlib, sys\n"
            "src = pathlib.Path('shop.py').read_text()\n"
            "if src == pathlib.Path('orig.txt').read_text():\n    sys.exit(0)\n"
            + body + "sys.exit(2 if 'a >= b' in src else 1)\n")


def test_show_output_prints_the_tail_of_errored_mutants_only(proj):
    project = proj(TWO_SITES, _errored_first("print('line one')\nprint('ImportError: the mutant broke the import')\n"),
                   prelude=False)
    plain = cli(project)
    assert "ImportError" not in plain.stdout
    shown = cli(project, "--show-output")
    assert shown.stdout.count("        | ") == 2  # the killed mutant gets no output lines
    assert "        | ImportError: the mutant broke the import" in shown.stdout
    assert "        | line one" in shown.stdout
    _, payload = cli_json(project, "--show-output")
    assert [m["status"] for m in payload["mutants"]] == ["errored (exit 2)", "killed"]
    assert payload["mutants"][0]["output_tail"] == "line one\nImportError: the mutant broke the import"
    assert "output_tail" in payload["mutants"][1]
    _, payload = cli_json(project)
    assert all("output_tail" not in m for m in payload["mutants"])


def test_show_output_keeps_only_the_last_ten_lines(proj):
    project = proj(TWO_SITES, _errored_first("[print(f'row {i}') for i in range(30)]\n"), prelude=False)
    _, payload = cli_json(project, "--show-output")
    assert payload["mutants"][0]["output_tail"].splitlines() == [f"row {i}" for i in range(20, 30)]


# ---------------------------------------------------------------------------
# Cases that run the real pytest as the test command (slowest, so last).
# ---------------------------------------------------------------------------

SOURCE = "def shipping_cents(subtotal):\n    if subtotal >= 5000:\n        return 0\n    return 499\n"
STRONG = (
    "from shop import shipping_cents\n\n\n"
    "def test_boundary_is_free():\n    assert shipping_cents(5000) == 0\n\n\n"
    "def test_just_under_pays():\n    assert shipping_cents(4999) == 499\n"
)
WEAK = "from shop import shipping_cents\n\n\ndef test_big_order_is_free():\n    assert shipping_cents(9000) == 0\n"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "shop.py").write_text(SOURCE)
    (tmp_path / "tests").mkdir()
    return tmp_path


def run(project, *extra):
    cmd = [sys.executable, str(SCRIPT), "shop.py", *extra, "--", sys.executable, "-m", "pytest", "-x", "-q",
           "-p", "no:cacheprovider", "tests/"]
    env = {"PYTHONPATH": str(project), "PATH": "/usr/bin:/bin", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    return subprocess.run(cmd, cwd=project, capture_output=True, text=True, timeout=300, env=env, check=False)


def test_strong_tests_kill_every_mutant(project):
    (project / "tests" / "test_shop.py").write_text(STRONG)
    proc = run(project)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "6/6 mutants killed (100%)" in proc.stdout


def test_weak_tests_leave_survivors_and_exit_1(project):
    (project / "tests" / "test_shop.py").write_text(WEAK)
    proc = run(project)
    assert proc.returncode == 1
    assert "SURVIVED line 2: >= -> >" in proc.stdout


def test_json_output_lists_each_mutant(project):
    (project / "tests" / "test_shop.py").write_text(WEAK)
    proc = run(project, "--format", "json")
    payload = json.loads(proc.stdout)
    statuses = {m["status"] for m in payload["mutants"]}
    assert statuses == {"SURVIVED", "killed"}


def test_source_is_restored_after_the_run(project):
    (project / "tests" / "test_shop.py").write_text(WEAK)
    run(project)
    assert (project / "shop.py").read_text() == SOURCE
    assert not list(project.glob("*.bak"))


def test_red_baseline_exits_2(project):
    (project / "tests" / "test_shop.py").write_text("def test_red():\n    assert 1 + 1 == 3\n")
    assert run(project).returncode == 2


def test_children_reuse_the_bytecode_cache_without_writing_to_it(project):
    # A fresh, empty PYTHONPYCACHEPREFIX per run made every child recompile pytest and the
    # stdlib (about 3x slower per run). Not writing bytecode is enough to keep mutants fresh.
    log = project / "probe.log"
    (project / "tests" / "test_a_probe.py").write_text(
        "import sys\n\n\ndef test_probe():\n"
        f"    with open({str(log)!r}, 'a') as f:\n"
        "        f.write(f'{sys.pycache_prefix} {sys.flags.dont_write_bytecode}\\n')\n"
    )
    (project / "tests" / "test_shop.py").write_text(STRONG)
    assert run(project).returncode == 0
    lines = log.read_text().splitlines()
    assert len(lines) > 1
    assert set(lines) == {"None 1"}


def test_infinite_loop_mutant_is_killed_without_waiting_ten_seconds(tmp_path):
    (tmp_path / "shop.py").write_text("def count_down(n):\n    while n > 0:\n        n -= 1\n    return n\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_shop.py").write_text(
        "from shop import count_down\n\n\ndef test_reaches_zero():\n    assert count_down(3) == 0\n"
    )
    payload = json.loads(run(tmp_path, "--format", "json").stdout)
    timed_out = [m for m in payload["mutants"] if m["status"] == "killed (timeout)"]
    assert [m["description"] for m in timed_out] == ["-= -> +="]
    assert timed_out[0]["seconds"] < 9.0
