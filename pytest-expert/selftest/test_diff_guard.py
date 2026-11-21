"""diff_guard: each test-gaming edit is reported, and legitimate edits are not.

Every scenario builds a throwaway git repository, commits a base, applies the edit and
compares the exact (code, severity) findings, so a spurious extra finding, a missing one
or a severity downgrade all fail. Legitimate refactors assert that there are NO findings.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

GIT_OPT_OUT = "PYTEST_EXPERT_SKIP_GIT_TESTS"

SEVERITY = {
    "TEST_FILE_DELETED": "error",
    "TEST_DELETED": "error",
    "TEST_EMPTIED": "error",
    "ASSERTS_REDUCED": "error",
    "ASSERT_WEAKENED": "error",
    "CASES_REDUCED": "error",
    "ASSERT_TRIVIALIZED": "error",
    "SKIP_ADDED": "error",
    "PRAGMA_ADDED": "error",
    "COLLECTION_NARROWED": "error",
    "CONFIG_NARROWED": "error",
    "COVERAGE_LOWERED": "error",
    "SYNTAX": "error",
    "EXPECTATION_CHANGED": "warning",
    "CONSTANT_EQ": "warning",
    "SPECIAL_CASE": "warning",
    "SNAPSHOT_CHANGED": "warning",
    "PARAM_HOOK_CHANGED": "warning",
    "SKIP_VIA_HELPER": "warning",
}


@pytest.mark.skipif(os.environ.get(GIT_OPT_OUT) == "1", reason=f"{GIT_OPT_OUT}=1: the diff_guard tests are deliberately not run")
def test_git_is_available_or_the_opt_out_is_explicit():
    """Without git every other test here is skipped; that must never pass unnoticed."""
    git = shutil.which("git")
    assert git is not None, f"git is not installed, so no diff_guard test can run. Install git, or set {GIT_OPT_OUT}=1 to opt out"
    version = subprocess.run([git, "--version"], capture_output=True, text=True, check=True)
    assert version.stdout.startswith("git version ")


BASE_TEST = '''import pytest


def test_withdraw_rejects_overdraft():
    with pytest.raises(ValueError, match="insufficient"):
        withdraw(10, 20)


def test_balance_after_deposit():
    assert deposit(10, 5) == 15
    assert history() == ["deposit"]


class TestFees:
    def test_fee_is_one_percent(self):
        assert fee(1000) == 10
'''

EXTRA_TEST = '''import unittest

import pytest


@pytest.mark.skip
def test_bare_skip():
    assert one() == 1


@pytest.mark.xfail
def test_bare_xfail():
    assert two() == 2


@pytest.mark.parametrize(("a", "b"), [(1, 2), (5000, 0), (3, 4)])
def test_param_rows(a, b):
    assert f(a) == b


class TestUnit(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(add(1, 2), 3)
        with self.assertRaises(ValueError):
            add("a", 1)
'''

COND_TEST = '''import sys

import pytest

if sys.platform != "none":
    def test_cond_a():
        assert one() == 1

try:
    import numpy
except ImportError:
    numpy = None
else:
    def test_cond_b():
        assert two() == 2
'''

PYPROJECT = '''[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-ra"]

[tool.coverage.report]
fail_under = 90
'''

TEST_FILE = "tests/test_wallet.py"
EXTRA_FILE = "tests/test_extra.py"
COND_FILE = "tests/test_cond.py"

DEFAULT_FILES: dict[str, str | bytes] = {
    TEST_FILE: BASE_TEST,
    EXTRA_FILE: EXTRA_TEST,
    COND_FILE: COND_TEST,
    "tests/conftest.py": "import pytest\n",
    "pyproject.toml": PYPROJECT,
}


def sub(text: str, old: str, new: str) -> str:
    """str.replace that refuses to be a silent no-op, so a case cannot pass vacuously."""
    assert old in text, f"edit target not found: {old!r}"
    return text.replace(old, new, 1)


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def write(root: Path, rel: str, content: str | bytes | None) -> None:
    path = root / rel
    if content is None:
        path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def make_repo(root: Path, files: dict[str, str | bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "t")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "commit.gpgsign", "false")
    for rel, content in files.items():
        write(root, rel, content)
    commit(root, "base")
    return root


def commit(root: Path, message: str) -> None:
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)


@pytest.fixture(autouse=True)
def isolated_git_config(monkeypatch):
    """A user's global or system git configuration must not change a probe's behaviour."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture(scope="session")
def guard_mod():
    from conftest import load_script

    return load_script("scripts/diff_guard.py")


@pytest.fixture
def require_git():
    if shutil.which("git") is None:
        if os.environ.get(GIT_OPT_OUT) == "1":
            pytest.skip(f"git is not installed and {GIT_OPT_OUT}=1")
        pytest.fail(f"git is not installed (set {GIT_OPT_OUT}=1 to opt out of these tests)")


@pytest.fixture
def repo(tmp_path, require_git):
    return make_repo(tmp_path / "repo", DEFAULT_FILES)


@pytest.fixture
def run(tmp_path, guard_mod, require_git):
    """run(edits, base=None, moves=()) -> sorted [(code, severity)] after applying the edits."""

    def _run(edits, base=None, moves=(), git_mv=True):
        files = {**DEFAULT_FILES, **(base or {})}
        root = make_repo(tmp_path / "scenario", {k: v for k, v in files.items() if v is not None})
        for old, new in moves:
            (root / new).parent.mkdir(parents=True, exist_ok=True)
            if git_mv:
                git(root, "mv", old, new)
            else:
                (root / old).rename(root / new)
        for rel, content in edits.items():
            write(root, rel, content)
        return sorted((f.code, f.severity) for f in guard_mod.guard(root))

    return _run


def expect(*codes: str) -> list[tuple[str, str]]:
    return sorted((c, SEVERITY[c]) for c in codes)


def case(case_id: str, edits: dict, *codes: str, base: dict | None = None, moves: tuple = (), git_mv: bool = True):
    return pytest.param(edits, base, moves, git_mv, expect(*codes), id=case_id)


def wallet(old: str, new: str) -> dict:
    return {TEST_FILE: sub(BASE_TEST, old, new)}


def extra(old: str, new: str) -> dict:
    return {EXTRA_FILE: sub(EXTRA_TEST, old, new)}


def test_code_table_pins_every_severity(guard_mod):
    assert {code: sev for code, (sev, _) in guard_mod.CODES.items()} == SEVERITY
    assert len(guard_mod.CODES) == 19  # 17 before PARAM_HOOK_CHANGED and SKIP_VIA_HELPER; a new code is a deliberate edit here


# ---------------------------------------------------------------------------------------
# Pure helpers, checked directly (no git): fast, and they pin the exact boundaries
# ---------------------------------------------------------------------------------------
NARROWING = [
    (["-k", "not x"], {"-k not x": 1}),
    (["-knot x"], {"-k not x": 1}),
    (["-m", "slow"], {"-m slow": 1}),
    (["-mslow"], {"-m slow": 1}),
    (["--deselect", "a", "--deselect=b"], {"--deselect a": 1, "--deselect b": 1}),
    (["--ignore", "a", "--ignore=b"], {"--ignore a": 1, "--ignore b": 1}),
    (["--ignore-glob", "g", "--ignore-glob=h"], {"--ignore-glob g": 1, "--ignore-glob h": 1}),
    (["--co", "--collect-only", "--no-cov", "--noconftest"], {"--co": 1, "--collect-only": 1, "--no-cov": 1, "--noconftest": 1}),
    (["-k"], {"-k": 1}),
    (["-k", "a", "-k", "a"], {"-k a": 2}),
    (["--collect-only", "--collect-only"], {"--collect-only": 2}),
    (["--maxfail=1", "-x", "-q", "-ra", "--cov-fail-under=5", "-p", "no:cov", "--kk", "--strict-markers"], {}),
    ([], {}),
]


@pytest.mark.parametrize(("options", "expected"), NARROWING)
def test_unit_narrowing_tokens(guard_mod, options, expected):
    assert dict(guard_mod.narrowing_tokens(options)) == expected


COV_FLAGS = [
    (["--cov-fail-under=80"], 80.0),
    (["--cov-fail-under", "80"], 80.0),
    (["--cov-fail-under=80", "--cov-fail-under", "50"], 50.0),
    (["--cov-fail-under", "50", "--cov-fail-under=80"], 80.0),
    (["--cov-fail-under"], None),
    (["-ra"], None),
    ([], None),
]


@pytest.mark.parametrize(("options", "expected"), COV_FLAGS)
def test_unit_cov_fail_under_flag(guard_mod, options, expected):
    assert guard_mod.cov_fail_under_flag("x", options) == expected


def test_unit_cov_fail_under_flag_rejects_a_non_number(guard_mod):
    with pytest.raises(guard_mod.UnreadableError, match="fail_under"):
        guard_mod.cov_fail_under_flag("x", ["--cov-fail-under=abc"])


def test_unit_effective_floor_prefers_the_flag_over_the_coverage_file(guard_mod):
    cov_cfg = guard_mod.CoverageCfg(".coveragerc", 90.0, set())
    assert guard_mod.effective_floor(guard_mod.PytestCfg("pytest.ini", ["--cov-fail-under=10"], {}), cov_cfg) == (10.0, "pytest.ini")  # audit: ignore[FLOAT_EQ] the floor is parsed from the same literal ("10", "90", "0"), so the float is exact
    assert guard_mod.effective_floor(guard_mod.PytestCfg("pytest.ini", ["-ra"], {}), cov_cfg) == (90.0, ".coveragerc")  # audit: ignore[FLOAT_EQ] the floor is parsed from the same literal ("10", "90", "0"), so the float is exact
    assert guard_mod.effective_floor(None, cov_cfg) == (90.0, ".coveragerc")  # audit: ignore[FLOAT_EQ] the floor is parsed from the same literal ("10", "90", "0"), so the float is exact
    assert guard_mod.effective_floor(None, guard_mod.CoverageCfg(".coveragerc", None, set())) is None
    assert guard_mod.effective_floor(None, None) is None
    assert guard_mod.effective_floor(guard_mod.PytestCfg("pytest.ini", ["--cov-fail-under=0"], {}), None) == (0.0, "pytest.ini")  # audit: ignore[FLOAT_EQ] the floor is parsed from the same literal ("10", "90", "0"), so the float is exact


TRIVIAL = [
    "True", "1", "(1,)", "[1]", "{1}", "{1: 2}", "not False", "not None", "x == x", "x is x", "x <= x", "x >= x",
    "x == x == x", "1 < 2", "2 >= 1", "'a' == 'a'", "1 != 2", "x or True", "True and True", "isinstance(x, object)",
    "x == pytest.approx(3, abs=1e9)", "x == approx(3, rel=1)", "x == approx(0, abs=2)", "x == approx(3, abs=3)", "x == approx(y, abs=1e9)",
    "x != 3 or x == 3 or True", "x == approx(3, abs=1e9) == y",
]
NOT_TRIVIAL = [
    "x", "x == 3", "x != x", "x < x", "x > x", "2 < 1", "1 == 2", "False", "0", "()", "[]", "not True", "not x",
    "x and True", "x or y", "True and x", "isinstance(x, int)", "x == pytest.approx(3)", "x == approx(3, abs=1e-6)",
    "x == approx(3, rel=0.5)", "x == approx(1e12, abs=1e6)", "x == approx(3, abs=2)", "x == approx(y, abs=1e8)", "x in y",
    "x == y == z", "x == x == y", "1 < x", "f(x)", "x < 3", "x is not None", "x == approx(3, abs=y)", "x != approx(3, abs=1e9)",
    "x == other(3, abs=1e9)", "1 < 2 > 3", "1 < 'a'", "None < 1", "1 < 2 < 3 < 2",
]


@pytest.mark.parametrize("source", TRIVIAL)
def test_unit_trivial_assertions_are_detected(guard_mod, source):
    import ast

    assert guard_mod.is_trivial_assert(ast.parse(f"assert {source}").body[0]) is True


@pytest.mark.parametrize("source", NOT_TRIVIAL)
def test_unit_real_assertions_are_not_trivial(guard_mod, source):
    import ast

    assert guard_mod.is_trivial_assert(ast.parse(f"assert {source}").body[0]) is False


SKIP_COUNTS = [
    ("import pytest\n@pytest.mark.skip\ndef f(): pass", 1),
    ("import pytest\n@pytest.mark.skip(reason='x')\ndef f(): pass", 1),
    ("import pytest\n@pytest.mark.skipif(True, reason='x')\ndef f(): pass", 1),
    ("import pytest\n@pytest.mark.xfail(strict=True)\ndef f(): pass", 1),
    ("import pytest\npytest.skip('x')", 1),
    ("import pytest\npytest.xfail('x')", 1),
    ("import pytest\nnp = pytest.importorskip('m')", 1),
    ("import pytest\npytest.skip('a')\npytest.skip('b')", 2),
    ("import pytest as pt\npt.skip('x')", 1),
    ("from pytest import skip\nskip()", 1),
    ("from pytest import skip as bail\nbail()", 1),
    ("from pytest import mark\n@mark.skip\ndef f(): pass", 1),
    ("import pytest\nm = pytest.mark\n@m.skip\ndef f(): pass", 1),
    ("import pytest\nslow = pytest.mark.skip(reason='x')\n@slow\ndef f(): pass", 2),  # the definition and the use; facts_for_tests counts only the use
    ("import unittest\n@unittest.skip('x')\ndef f(): pass", 1),
    ("import unittest\n@unittest.skipIf(True, 'x')\ndef f(): pass", 1),
    ("import unittest\n@unittest.skipUnless(False, 'x')\ndef f(): pass", 1),
    ("import unittest\n@unittest.expectedFailure\ndef f(): pass", 1),
    ("from unittest import skipIf\n@skipIf(True, 'x')\ndef f(): pass", 1),
    ("from unittest.case import skip\n@skip('x')\ndef f(): pass", 1),
    ("self.skipTest('x')", 1),
    ("import unittest\nraise unittest.SkipTest('x')", 1),
    ("from unittest import SkipTest\nraise SkipTest", 1),
    ("import pytest\nx = pytest.param(1, marks=pytest.mark.skip)", 1),
    ("import pytest\nx = pytest.param(1, marks=[pytest.mark.skip, pytest.mark.xfail])", 2),
    ("import pytest\n@pytest.mark.parametrize('a', [1])\ndef f(a): pass", 0),
    ("import pytest\nwith pytest.raises(ValueError): pass", 0),
    ("import pytest\n@pytest.mark.usefixtures('db')\ndef f(): pass", 0),
    ("skip()", 0),
    ("mark.skip", 0),
    ("def skip(): pass\nskip()", 0),
    ("import unittest.mock\nunittest.mock.patch('x')", 0),
    ("raise ValueError('x')", 0),
    ("x.skip", 0),
]


@pytest.mark.parametrize(("source", "expected"), SKIP_COUNTS)
def test_unit_count_skips(guard_mod, source, expected):
    import ast

    tree = ast.parse(source)
    assert guard_mod.count_skips(tree, guard_mod.build_aliases(tree)) == expected


CONSTANT_DUNDERS = [
    ("class C:\n    def __eq__(self, o):\n        return True\n", {"C.__eq__"}),
    ("class C:\n    def __ne__(self, o):\n        return False\n", {"C.__ne__"}),
    ("class C:\n    def __hash__(self):\n        return 1\n", {"C.__hash__"}),
    ("class C:\n    def __eq__(self, o):\n        'doc'\n        return True\n", {"C.__eq__"}),
    ("class C:\n    def __eq__(self, o):\n        return not False\n", {"C.__eq__"}),
    ("class C:\n    def __eq__(self, o):\n        return True if o else False\n", {"C.__eq__"}),
    ("class C:\n    def __eq__(self, o):\n        return bool(1)\n", {"C.__eq__"}),
    ("class C:\n    def __eq__(self, o):\n        return -1\n", {"C.__eq__"}),
    ("class C:\n    def __eq__(self, o):\n        x = o\n        return True\n", {"C.__eq__"}),
    ("class C:\n    __eq__ = lambda self, o: True\n", {"C.__eq__"}),
    ("class C:\n    def __eq__(self, o):\n        return self.a == o.a\n", set()),
    ("class C:\n    def __eq__(self, o):\n        return bool(o)\n", set()),
    ("class C:\n    def __eq__(self, o):\n        if o is None:\n            return False\n        return True\n", set()),
    ("class C:\n    def __eq__(self, o):\n        return True if o else self.a\n", set()),
    ("class C:\n    def other(self):\n        return True\n", set()),
    ("class C:\n    other = lambda self: True\n", set()),
    ("def __eq__(a, b):\n    return True\n", set()),
]


@pytest.mark.parametrize(("source", "expected"), CONSTANT_DUNDERS)
def test_unit_constant_dunders(guard_mod, source, expected):
    import ast

    assert guard_mod.constant_dunders(ast.parse(source)) == expected


def test_unit_literal_values_fold_negatives_and_skip_trivia(guard_mod):
    source = '"""docstring text"""\ndef test_a():\n    """another docstring"""\n    assert f(-41) == -42.5\n    assert g("abc") == "x"\n    assert g("ab") == "y"\n    assert h(0) == -1\n    assert h(1) == 2\n    assert h(+7) == 99\n    assert k(True) is None\n'
    assert guard_mod.literal_values(source) == {-41, -42.5, "abc", "ab", 99, 7}  # audit: ignore[FLOAT_EQ] -42.5 is an exact binary fraction and set equality has no approx form


def test_unit_literal_values_of_unparsable_source_is_empty(guard_mod):
    assert guard_mod.literal_values("def (:\n") == set()
    assert guard_mod.literal_values(b"\xff\xfe") == set()


A_FILE = b"".join(b"line %d\n" % i for i in range(10))


def test_unit_pair_renames_matches_identical_and_similar_files(guard_mod):
    similar = A_FILE.replace(b"line 3\n", b"changed\n")
    other = b"".join(b"unrelated %d\n" % i for i in range(10))
    assert guard_mod.pair_renames({"a.py": A_FILE}, {"b.py": A_FILE}) == {"a.py": "b.py"}
    assert guard_mod.pair_renames({"a.py": A_FILE}, {"b.py": similar}) == {"a.py": "b.py"}
    assert guard_mod.pair_renames({"a.py": A_FILE}, {"b.py": other}) == {}
    assert guard_mod.pair_renames({}, {"b.py": A_FILE}) == {}
    assert guard_mod.pair_renames({"a.py": A_FILE}, {}) == {}


def test_unit_pair_renames_pairs_each_file_once_and_prefers_the_closest(guard_mod):
    second = b"".join(b"second %d\n" % i for i in range(10))
    removed = {"a.py": A_FILE, "c.py": second}
    added = {"d.py": second.replace(b"second 1\n", b"edited\n"), "b.py": A_FILE.replace(b"line 3\n", b"changed\n")}
    assert guard_mod.pair_renames(removed, added) == {"a.py": "b.py", "c.py": "d.py"}
    assert guard_mod.pair_renames({"a.py": A_FILE, "z.py": A_FILE}, {"b.py": A_FILE}) == {"a.py": "b.py"}


def test_unit_pair_renames_falls_back_to_exact_matches_when_there_are_too_many_candidates(guard_mod):
    removed = {f"old{i}.py": A_FILE.replace(b"line 3\n", b"v%d\n" % i) for i in range(25)}
    added = {f"new{i}.py": A_FILE.replace(b"line 4\n", b"w%d\n" % i) for i in range(20)}
    assert guard_mod.pair_renames(removed, added) == {}
    added["exact.py"] = removed["old5.py"]
    assert guard_mod.pair_renames(removed, added) == {"old5.py": "exact.py"}


def test_unit_a_marker_alias_is_counted_where_it_is_used_not_where_it_is_defined(guard_mod):
    source = "import pytest\nslow = pytest.mark.skip(reason='x')\n\n\n@slow\ndef test_a():\n    assert 1 + 1 == 3\n\n\ndef test_b():\n    assert 1 + 1 == 2\n"
    facts, module_skips, _ = guard_mod.facts_for_tests(source)
    assert (module_skips, facts["test_a"].skips, facts["test_b"].skips) == (0, 1, 0)


@pytest.mark.parametrize(("source", "expected"), [
    ("approx(3, abs=1e9)", True), ("pytest.approx(3, rel=1)", True), ("approx(3)", False), ("approx(3, abs=0.1)", False),
    ("approx(3, rel=0.9)", False), ("other(3, abs=1e9)", False), ("approx", False), ("approx(3, nan_ok=True)", False),
])
def test_unit_is_loose_approx(guard_mod, source, expected):
    import ast

    assert guard_mod.is_loose_approx(ast.parse(source, mode="eval").body) is expected


def test_unit_pair_renames_similarity_threshold_is_exactly_sixty_percent(guard_mod):
    base = b"a\nb\nc\nd\ne\n"
    assert guard_mod.pair_renames({"o.py": base}, {"n.py": b"a\nb\nc\nx\ny\n"}) == {"o.py": "n.py"}  # 2*3/10 = 0.6
    assert guard_mod.pair_renames({"o.py": b"a\nb\nc\nd\n"}, {"n.py": b"a\nb\nx\ny\n"}) == {}  # 2*2/8 = 0.5


def test_unit_pair_renames_picks_the_closest_candidate_and_never_reuses_a_file(guard_mod):
    near = A_FILE.replace(b"line 3\n", b"changed\n")
    far = A_FILE.replace(b"line 3\n", b"changed\n").replace(b"line 4\n", b"other\n").replace(b"line 5\n", b"more\n")
    assert guard_mod.pair_renames({"a.py": A_FILE}, {"far.py": far, "near.py": near}) == {"a.py": "near.py"}
    two_old = {"a.py": A_FILE.replace(b"line 1\n", b"p\n"), "z.py": A_FILE.replace(b"line 2\n", b"q\n")}
    assert len(guard_mod.pair_renames(two_old, {"b.py": near})) == 1
    two_new = {"b.py": near, "c.py": A_FILE.replace(b"line 7\n", b"r\n")}
    assert len(guard_mod.pair_renames({"a.py": A_FILE.replace(b"line 8\n", b"s\n")}, two_new)) == 1


def test_unit_pair_renames_examines_exactly_the_limit_of_candidate_pairs(guard_mod):
    removed = {f"old{i}.py": A_FILE.replace(b"line 3\n", b"v%d\n" % i) for i in range(20)}
    added = {f"new{i}.py": A_FILE.replace(b"line 4\n", b"w%d\n" % i) for i in range(20)}
    assert len(guard_mod.pair_renames(removed, added)) == 20  # 20 x 20 = 400 = the limit: still examined


@pytest.mark.parametrize(("source", "expected"), [
    ("-41", -41), ("-1.5", -1.5), ("+3", 3), ("-True", None), ("-'a'", None), ("-x", None), ("~1", None), ("41", None), ("-(1)", -1),
])
def test_unit_signed_number(guard_mod, source, expected):
    import ast

    assert guard_mod._signed_number(ast.parse(source, mode="eval").body) == expected


def _detail(repo, guard_mod, code):
    return [f.detail for f in guard_mod.guard(repo) if f.code == code]


def test_details_list_at_most_three_changed_items(repo, guard_mod):
    base = "def test_many():\n" + "".join(f"    assert f({i}) == {i}\n" for i in range(5))
    write(repo, "tests/test_many.py", base)
    commit(repo, "many")
    write(repo, "tests/test_many.py", base.replace("== 0", "== 10").replace("== 1\n", "== 11\n").replace("== 2", "== 12").replace("== 3", "== 13"))
    assert _detail(repo, guard_mod, "EXPECTATION_CHANGED") == ["test_many: 0 -> 10; 1 -> 11; 2 -> 12"]
    write(repo, "tests/test_many.py", "def test_many():\n    assert f(0) == g()\n    assert f(1) == g()\n    assert f(2) == g()\n    assert f(3) == g()\n    assert f(4) == g()\n")
    assert _detail(repo, guard_mod, "EXPECTATION_CHANGED") == ["test_many: expected literal(s) no longer asserted: 0; 1; 2"]


def test_row_replacement_detail_lists_at_most_three_rows(repo, guard_mod):
    rows = "import pytest\n\n\n@pytest.mark.parametrize(\"n\", [1, 2, 3, 4, 5])\ndef test_n(n):\n    assert f(n)\n"
    write(repo, "tests/test_n.py", rows)
    commit(repo, "rows")
    write(repo, "tests/test_n.py", rows.replace("[1, 2, 3, 4, 5]", "[10, 20, 30, 40, 5]"))
    assert _detail(repo, guard_mod, "EXPECTATION_CHANGED") == ["test_n: 4 parametrize row(s) replaced: 1; 2; 3"]


def test_trivialized_detail_counts_only_the_new_trivial_assertions(repo, guard_mod):
    write(repo, "tests/test_t.py", "def test_t():\n    assert True\n    assert f() == 1\n    assert g() == 2\n    assert h() == 3\n")
    commit(repo, "t")
    write(repo, "tests/test_t.py", "def test_t():\n    assert True\n    assert True\n    assert 1 < 2\n    assert h() == 3\n")
    assert _detail(repo, guard_mod, "ASSERT_TRIVIALIZED") == ["test_t: 2 assertion(s) that cannot fail"]


def test_equal_check_count_with_a_dropped_literal_is_reported(run):
    base = {"tests/test_lit.py": "def test_lit():\n    assert f() == 1\n    assert g() == 2\n"}
    edits = {"tests/test_lit.py": "def test_lit():\n    assert f() == 1\n    assert g() == h()\n"}
    assert run(edits, base=base) == expect("EXPECTATION_CHANGED")


SKIPPED_MODULE = 'import pytest\n\npytestmark = pytest.mark.skip(reason="wip")\n\n\ndef test_x():\n    assert f() == 1\n'


def test_moving_a_test_between_two_modules_that_are_both_skipped_is_not_a_new_skip(run):
    other = "\n\ndef test_other():\n    assert g() == 2\n"
    base = {"tests/test_a.py": SKIPPED_MODULE + other}
    edits = {"tests/test_a.py": SKIPPED_MODULE.replace("\n\ndef test_x():\n    assert f() == 1\n", "") + other, "tests/test_b.py": SKIPPED_MODULE}
    assert run(edits, base=base) == []


def test_unit_conftest_facts_tolerates_odd_shapes(guard_mod):
    odd = "collect_ignore = []\ncollect_ignore.insert(0)\ncollect_ignore.pop()\nother = []\nother.append('x')\nitems.append(1)\ncollect_ignore: list\n"
    facts = guard_mod.conftest_facts(odd)
    assert (dict(facts.entries), facts.hooks, facts.skips) == ({}, {}, 0)
    chained = guard_mod.conftest_facts("a = collect_ignore = ['x.py']\nb, collect_ignore_glob = 1, ['*.py']\n")
    assert dict(chained.entries) == {"'x.py'": 1}  # a tuple-unpacking target is not a collect_ignore assignment
    assert guard_mod.conftest_facts("def (:\n") == guard_mod.ConftestFacts(guard_mod.Counter(), {}, 0)
    assert guard_mod.conftest_facts("import pytest\npytest.skip('x', allow_module_level=True)\n").skips == 1


def test_an_unparsable_conftest_is_only_a_syntax_finding(run):
    assert run({"tests/conftest.py": "import pytest\npytest.skip('x'\n"}) == expect("SYNTAX")


def test_file_level_findings_point_at_line_1_of_the_right_file(repo, guard_mod):
    write(repo, "pyproject.toml", sub(PYPROJECT, 'testpaths = ["tests"]', 'testpaths = ["x"]\nnorecursedirs = ["tests/a"]\npython_files = ["c_*.py"]').replace('"-ra"', '"-ra", "-k", "x"').replace("fail_under = 90", "fail_under = 1\nomit = [\"s/*\"]"))
    write(repo, "tests/conftest.py", "import pytest\ncollect_ignore = ['a']\ndef pytest_collect_file(parent, file_path):\n    return None\ndef pytest_runtest_setup(item):\n    pytest.skip('x')\n")
    found = sorted((f.file, f.line, f.code, f.detail) for f in guard_mod.guard(repo))
    assert found == sorted([
        ("pyproject.toml", 1, "CONFIG_NARROWED", "addopts gained '-k x'"),
        ("pyproject.toml", 1, "CONFIG_NARROWED", "testpaths lost ['tests']"),
        ("pyproject.toml", 1, "CONFIG_NARROWED", "norecursedirs gained ['tests/a']"),
        ("pyproject.toml", 1, "CONFIG_NARROWED", "python_files lost ['test_*.py', '*_test.py']"),
        ("pyproject.toml", 1, "COVERAGE_LOWERED", "fail_under 90 -> 1"),
        ("pyproject.toml", 1, "COVERAGE_LOWERED", "coverage omit gained ['s/*']"),
        ("tests/conftest.py", 1, "COLLECTION_NARROWED", "collect_ignore gained [\"'a'\"]"),
        ("tests/conftest.py", 1, "COLLECTION_NARROWED", "pytest_collect_file added; check it does not deselect failing tests"),
        ("tests/conftest.py", 1, "SKIP_ADDED", "conftest gained a skip/xfail (a hook or autouse fixture can skip every test)"),
    ])


def test_changed_hook_body_finding_points_at_line_1(repo, guard_mod):
    write(repo, "tests/conftest.py", CONFTEST_RICH)
    commit(repo, "hook")
    write(repo, "tests/conftest.py", sub(CONFTEST_RICH, "item.add_marker(pytest.mark.slow)", "items.remove(item)"))
    assert [(f.file, f.line, f.detail) for f in guard_mod.guard(repo)] == [
        ("tests/conftest.py", 1, "pytest_collection_modifyitems body changed; check it does not deselect failing tests")
    ]


def test_unit_github_annotation_plain(guard_mod):
    finding = guard_mod.Finding("tests/a.py", 7, "TEST_DELETED", "x gone", "error")
    assert guard_mod.github_annotation(finding) == "::error file=tests/a.py,line=7::TEST_DELETED x gone"
    assert guard_mod.github_annotation(guard_mod.Finding("a:b.py", 1, "SPECIAL_CASE", "r\rs", "warning")) == "::warning file=a%3Ab.py,line=1::SPECIAL_CASE r%0Ds"


# ---------------------------------------------------------------------------------------
# The original gaming edits, now with exact finding sets and severities
# ---------------------------------------------------------------------------------------
PYPROJECT_NO_COV = '[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = ["-ra"]\n'
CONFTEST_BASE = "import pytest\n"
FEE_LINE = "assert fee(1000) == 10"

CORE = [
    case("file-deleted", {TEST_FILE: None}, "TEST_FILE_DELETED"),
    case("test-deleted", wallet('def test_withdraw_rejects_overdraft():\n    with pytest.raises(ValueError, match="insufficient"):\n        withdraw(10, 20)\n', ""), "TEST_DELETED"),
    case("assert-dropped", wallet('    assert history() == ["deposit"]\n', ""), "ASSERTS_REDUCED"),
    case("assert-true", wallet(FEE_LINE, "assert True"), "ASSERT_TRIVIALIZED"),
    case("self-compare", wallet(FEE_LINE, "assert fee(1000) == fee(1000)"), "ASSERT_TRIVIALIZED"),
    case("skip-marker", wallet("def test_balance_after_deposit():\n", '@pytest.mark.skip(reason="flaky")\ndef test_balance_after_deposit():\n'), "SKIP_ADDED"),
    case("class-xfail", wallet("class TestFees:", '@pytest.mark.xfail(reason="later", strict=False)\nclass TestFees:'), "SKIP_ADDED"),
    case("module-pytestmark", wallet("import pytest\n", 'import pytest\n\npytestmark = pytest.mark.skip(reason="wip")\n'), "SKIP_ADDED"),
    case("skip-call", wallet("    assert deposit(10, 5) == 15\n", '    pytest.skip("todo")\n    assert deposit(10, 5) == 15\n'), "SKIP_ADDED"),
    case("early-return", wallet("    assert fee(1000) == 10\n", "    return\n        assert fee(1000) == 10\n"), "TEST_EMPTIED"),
    case("body-pass", wallet("    assert fee(1000) == 10\n", "        pass\n"), "TEST_EMPTIED"),
    case("audit-pragma", wallet("assert deposit(10, 5) == 15", "assert deposit(10, 5) == 15  # audit: ignore[FLOAT_EQ] fine"), "PRAGMA_ADDED"),
    case("collect-ignore", {"tests/conftest.py": "import pytest\ncollect_ignore = ['test_wallet.py']\n"}, "COLLECTION_NARROWED"),
    case("modifyitems-hook", {"tests/conftest.py": "import pytest\n\ndef pytest_collection_modifyitems(config, items):\n    items[:] = [i for i in items if 'fee' not in i.name]\n"}, "COLLECTION_NARROWED"),
    case("deselect", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "--deselect", "tests/test_wallet.py::test_balance_after_deposit"]')}, "CONFIG_NARROWED"),
    case("dash-k", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "-k", "not fee"]')}, "CONFIG_NARROWED"),
    case("ignore-equals", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "--ignore=tests/test_wallet.py"]')}, "CONFIG_NARROWED"),
    case("testpaths-shrunk", {"pyproject.toml": sub(PYPROJECT, 'testpaths = ["tests"]', 'testpaths = ["tests/unit"]')}, "CONFIG_NARROWED"),
    case("fail-under-lowered", {"pyproject.toml": sub(PYPROJECT, "fail_under = 90", "fail_under = 60")}, "COVERAGE_LOWERED"),
    case("cov-fail-under-flag", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "--cov-fail-under=50"]')}, "COVERAGE_LOWERED"),
    case("expected-literal-changed", wallet(FEE_LINE, "assert fee(1000) == 11"), "EXPECTATION_CHANGED"),
    case("unparsable-file", {TEST_FILE: BASE_TEST + "\ndef test_broken(:\n"}, "SYNTAX"),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), CORE)
def test_core_gaming_edit_is_reported_exactly(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


LEGITIMATE = [
    case("no-change", {}),
    case("test-added", {TEST_FILE: BASE_TEST + "\n\ndef test_new_behavior():\n    assert deposit(0, 1) == 1\n"}),
    case("assert-added", wallet("    assert fee(1000) == 10\n", "    assert fee(1000) == 10\n        assert fee(0) == 0\n")),
    case("test-moved-to-another-file", {
        "tests/test_other.py": "def test_balance_after_deposit():\n    assert deposit(10, 5) == 15\n    assert history() == ['deposit']\n",
        TEST_FILE: sub(BASE_TEST, "def test_balance_after_deposit():\n    assert deposit(10, 5) == 15\n    assert history() == [\"deposit\"]\n", ""),
    }),
    case("fail-under-raised", {"pyproject.toml": sub(PYPROJECT, "fail_under = 90", "fail_under = 95")}),
    case("testpaths-extended", {"pyproject.toml": sub(PYPROJECT, 'testpaths = ["tests"]', 'testpaths = ["tests", "more"]')}),
    case("harmless-addopts", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "-q", "--strict-markers", "--maxfail=1", "-x", "-p", "no:randomly"]')}),
    case("source-only-change", {"src/wallet.py": "def fee(x):\n    return x // 100\n"}),
    case("unrelated-docs-and-config", {"README.md": "hi\n", "setup.cfg": "[metadata]\nname = x\n"}),
    case("added-assert-of-a-call", wallet("    assert fee(1000) == 10\n", "    assert fee(1000) == 10\n        assert fee(0)\n        assert fee(1) < 3\n        assert 1 in fee_list()\n")),
    case("assert-wrapped-in-with", wallet("        assert fee(1000) == 10\n", "        with fee_lock:\n            assert fee(1000) == 10\n")),
    case("asserts-merged-with-and", wallet('    assert deposit(10, 5) == 15\n    assert history() == ["deposit"]\n', '    assert deposit(10, 5) == 15 and history() == ["deposit"]\n')),
    case("equality-replaced-by-isclose", wallet(FEE_LINE, "assert math.isclose(fee(1000), 10)")),
    case("equality-replaced-by-pytest-approx", wallet(FEE_LINE, "assert fee(1000) == pytest.approx(10)")),
    case("assert-in-live-if-true", wallet(FEE_LINE, "if True:\n            assert fee(1000) == 10")),
    case("nested-function-that-is-called", wallet(FEE_LINE, "def check():\n            assert fee(1000) == 10\n\n        check()")),
    case("raises-match-made-more-specific", wallet('match="insufficient"', 'match="insufficient funds"')),
    case("assert-converted-to-unittest-style", extra("        self.assertEqual(add(1, 2), 3)\n", "        assert add(1, 2) == 3\n")),
    case("reason-added-to-bare-skip", extra("@pytest.mark.skip\ndef test_bare_skip", '@pytest.mark.skip(reason="documented")\ndef test_bare_skip')),
    case("strict-added-to-bare-xfail", extra("@pytest.mark.xfail\ndef test_bare_xfail", "@pytest.mark.xfail(strict=True)\ndef test_bare_xfail")),
    case("parametrize-rewritten-to-keyword-argvalues", extra('("a", "b"), [(1, 2), (5000, 0), (3, 4)])', '("a", "b"), argvalues=[(1, 2), (5000, 0), (3, 4)])')),
    case("parametrize-row-added", extra("(3, 4)]", "(3, 4), (7, 8)]")),
    case("module-marker-that-skips-nothing", wallet("import pytest\n", 'import pytest\n\npytestmark = pytest.mark.usefixtures("db")\n')),
    case("new-test-file-with-module-skip-and-no-moved-tests", {"tests/test_new_skipped.py": 'import pytest\n\npytestmark = pytest.mark.skip(reason="wip")\n\ndef test_fresh():\n    assert 1 + 1 == 2\n'}),
    case("conftest-fixture-added", {
        "tests/conftest.py": "import pytest\n\n@pytest.fixture\ndef db():\n    return 1\n",
    }),
    case("uncommitted-new-test-file", {"tests/test_new.py": "def test_x():\n    assert True\n"}),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), LEGITIMATE)
def test_legitimate_edit_has_no_findings_at_all(run, edits, base, moves, git_mv, expected):
    assert expected == []
    assert run(edits, base, moves, git_mv) == []


# ---------------------------------------------------------------------------------------
# Weakened assertions that keep the count
# ---------------------------------------------------------------------------------------
WEAKENED = [
    case("equality-loosened-to-truthiness", wallet(FEE_LINE, "assert fee(1000)"), "ASSERT_WEAKENED"),
    case("equality-loosened-to-is-not-none", wallet(FEE_LINE, "assert fee(1000) is not None"), "ASSERT_WEAKENED"),
    case("equality-loosened-to-membership", wallet(FEE_LINE, "assert fee(1000) in (10, 11, 12)"), "ASSERT_WEAKENED"),
    case("equality-loosened-to-inequality", wallet(FEE_LINE, "assert fee(1000) >= 0"), "ASSERT_WEAKENED"),
    case("approx-with-huge-tolerance", wallet(FEE_LINE, "assert fee(1000) == pytest.approx(10, abs=1e9)"), "ASSERT_TRIVIALIZED"),
    case("approx-with-relative-tolerance-of-one", wallet(FEE_LINE, "assert fee(1000) == pytest.approx(10, rel=1)"), "ASSERT_TRIVIALIZED"),
    case("or-true", wallet(FEE_LINE, "assert fee(1000) == 10 or True"), "ASSERT_TRIVIALIZED"),
    case("constant-comparison", wallet(FEE_LINE, "assert 1 < 2"), "ASSERT_TRIVIALIZED"),
    case("chained-self-comparison", wallet(FEE_LINE, "assert fee(1000) == fee(1000) == fee(1000)"), "ASSERT_TRIVIALIZED"),
    case("isinstance-of-object", wallet(FEE_LINE, "assert isinstance(fee(1000), object)"), "ASSERT_TRIVIALIZED"),
    case("not-false", wallet(FEE_LINE, "assert not False"), "ASSERT_TRIVIALIZED"),
    case("raises-widened-to-exception", wallet("pytest.raises(ValueError,", "pytest.raises(Exception,"), "ASSERT_WEAKENED"),
    case("raises-match-removed", wallet('pytest.raises(ValueError, match="insufficient")', "pytest.raises(ValueError)"), "ASSERT_WEAKENED"),
    case("raises-match-loosened-to-anything", wallet('match="insufficient"', 'match=".*"'), "ASSERT_WEAKENED"),
    case("assert-under-if-false", wallet(FEE_LINE, "if False:\n            assert fee(1000) == 10"), "ASSERTS_REDUCED"),
    case("assert-after-return", wallet(FEE_LINE, "x = fee(1000)\n        return\n        assert x == 10"), "ASSERTS_REDUCED"),
    case("assert-after-if-true-return", wallet(FEE_LINE, "x = fee(1000)\n        if True:\n            return\n        assert x == 10"), "ASSERTS_REDUCED"),
    case("assert-after-raise", wallet(FEE_LINE, "raise RuntimeError\n        assert fee(1000) == 10"), "ASSERTS_REDUCED"),
    case("assert-in-nested-function-never-called", wallet(FEE_LINE, "def check():\n            assert fee(1000) == 10"), "ASSERTS_REDUCED"),
    case("unittest-assert-call-removed", extra("        self.assertEqual(add(1, 2), 3)\n", ""), "ASSERTS_REDUCED"),
    case("unittest-assert-raises-removed", extra('        with self.assertRaises(ValueError):\n            add("a", 1)\n', ""), "ASSERTS_REDUCED"),
    case("with-raises-block-removed", wallet('    with pytest.raises(ValueError, match="insufficient"):\n        withdraw(10, 20)\n', "    withdraw(10, 20)\n"), "ASSERTS_REDUCED"),
    case("parametrize-row-swapped-for-another", extra("(5000, 0)", "(5001, 0)"), "EXPECTATION_CHANGED"),
    case("parametrize-row-dropped", extra("(5000, 0), ", ""), "CASES_REDUCED"),
    case("expected-literal-changed-and-assert-added", wallet("    assert fee(1000) == 10\n", "    assert fee(1000) == 11\n        assert fee(5) == 0\n"), "EXPECTATION_CHANGED"),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), WEAKENED)
def test_weakening_that_keeps_the_count_is_reported(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


KEYWORD_ROWS = sub(EXTRA_TEST, '("a", "b"), [(1, 2), (5000, 0), (3, 4)])', '("a", "b"), argvalues=[(1, 2), (5000, 0), (3, 4)])')


def test_dropping_a_row_from_a_keyword_argvalues_list_is_reported(run):
    edits = {EXTRA_FILE: sub(KEYWORD_ROWS, "(5000, 0), ", "")}
    assert run(edits, base={EXTRA_FILE: KEYWORD_ROWS}) == expect("CASES_REDUCED")


def test_parametrize_rows_from_a_module_constant_are_counted(run):
    rows = 'import pytest\n\nROWS = [(1, 2), (5000, 0), (3, 4)]\n\n@pytest.mark.parametrize(("a", "b"), ROWS)\ndef test_const(a, b):\n    assert f(a) == b\n'
    edits = {"tests/test_const.py": sub(rows, "(5000, 0), ", "")}
    assert run(edits, base={"tests/test_const.py": rows}) == expect("CASES_REDUCED")


CLASS_ROWS = 'import pytest\n\n\n@pytest.mark.parametrize("n", [1, 2, 3])\nclass TestRows:\n    def test_n(self, n):\n        assert f(n) == n\n'
MODULE_ROWS = 'import pytest\n\npytestmark = pytest.mark.parametrize("n", [1, 2, 3])\n\n\ndef test_n(n):\n    assert f(n) == n\n'


@pytest.mark.parametrize("text", [CLASS_ROWS, MODULE_ROWS], ids=["class-decorator", "module-pytestmark"])
def test_rows_from_class_and_module_level_parametrize_are_counted(run, text):
    edits = {"tests/test_rows_shared.py": sub(text, "[1, 2, 3]", "[1, 2]")}
    assert run(edits, base={"tests/test_rows_shared.py": text}) == expect("CASES_REDUCED")


def test_a_reduced_case_message_names_the_counts(repo, guard_mod):
    write(repo, EXTRA_FILE, sub(EXTRA_TEST, "(5000, 0), ", ""))
    assert [(f.code, f.detail) for f in guard_mod.guard(repo)] == [("CASES_REDUCED", "test_param_rows: 3 parametrize rows -> 2")]


# ---------------------------------------------------------------------------------------
# Skips and collection the guard has to count
# ---------------------------------------------------------------------------------------
SKIPS = [
    case("importorskip-assignment", wallet("import pytest\n", 'import pytest\n\nnp = pytest.importorskip("numpy_does_not_exist")\n'), "SKIP_ADDED"),
    case("annotated-pytestmark", wallet("import pytest\n", 'import pytest\n\npytestmark: list = [pytest.mark.skip(reason="wip")]\n'), "SKIP_ADDED"),
    case("pytestmark-augmented", wallet("import pytest\n", "import pytest\n\npytestmark = []\npytestmark += [pytest.mark.skip]\n"), "SKIP_ADDED"),
    case("pytestmark-appended", wallet("import pytest\n", 'import pytest\n\npytestmark = [pytest.mark.usefixtures("db")]\npytestmark.append(pytest.mark.skip)\n'), "SKIP_ADDED"),
    case("class-body-pytestmark", wallet("class TestFees:\n", "class TestFees:\n    pytestmark = pytest.mark.skip\n\n"), "SKIP_ADDED"),
    case("module-skip-under-if", wallet("import pytest\n", 'import pytest\nimport sys\n\nif sys.platform:\n    pytest.skip("x", allow_module_level=True)\n'), "SKIP_ADDED"),
    case("autouse-fixture-that-skips", wallet("import pytest\n", 'import pytest\n\n\n@pytest.fixture(autouse=True)\ndef _gate():\n    pytest.skip("x")\n'), "SKIP_ADDED"),
]


def _aliased_import_edit(import_line: str, call: str) -> dict:
    text = sub(BASE_TEST, "import pytest\n", f"import pytest\n{import_line}\n")
    return {TEST_FILE: sub(text, "    assert deposit(10, 5) == 15\n", f"    {call}\n    assert deposit(10, 5) == 15\n")}


SKIPS += [
    case("aliased-pytest-module", _aliased_import_edit("import pytest as pt", 'pt.skip("x")'), "SKIP_ADDED"),
    case("skip-imported-by-name", _aliased_import_edit("from pytest import skip", 'skip("x")'), "SKIP_ADDED"),
    case("skip-imported-under-another-name", _aliased_import_edit("from pytest import skip as bail", 'bail("x")'), "SKIP_ADDED"),
    case("mark-imported-by-name", {TEST_FILE: sub(sub(BASE_TEST, "import pytest\n", "import pytest\nfrom pytest import mark\n"), "def test_balance_after_deposit", '@mark.skip(reason="x")\ndef test_balance_after_deposit')}, "SKIP_ADDED"),
    case("mark-alias-assignment", {TEST_FILE: sub(sub(BASE_TEST, "import pytest\n", "import pytest\n\nm = pytest.mark\n"), "def test_balance_after_deposit", "@m.skip\ndef test_balance_after_deposit")}, "SKIP_ADDED"),
    case("unittest-skip-decorator", extra("    def test_equal", '    @unittest.skip("x")\n    def test_equal'), "SKIP_ADDED"),
    case("unittest-expected-failure", extra("    def test_equal", "    @unittest.expectedFailure\n    def test_equal"), "SKIP_ADDED"),
    case("unittest-skip-test-call", extra("        self.assertEqual(add(1, 2), 3)\n", '        self.skipTest("x")\n        self.assertEqual(add(1, 2), 3)\n'), "SKIP_ADDED"),
    case("unittest-skiptest-raised", extra('            add("a", 1)\n', '            add("a", 1)\n        raise unittest.SkipTest("x")\n'), "SKIP_ADDED"),
    case("skip-mark-inside-pytest-param", extra("(5000, 0)", "pytest.param(5000, 0, marks=pytest.mark.skip)"), "SKIP_ADDED"),
    case("skip-mark-list-inside-pytest-param", extra("(5000, 0)", 'pytest.param(5000, 0, marks=[pytest.mark.skipif(True, reason="x")])'), "SKIP_ADDED"),
    case("test-moved-into-a-new-file-that-skips-everything", {
        TEST_FILE: sub(BASE_TEST, 'def test_balance_after_deposit():\n    assert deposit(10, 5) == 15\n    assert history() == ["deposit"]\n', ""),
        "tests/test_moved.py": 'import pytest\n\npytestmark = pytest.mark.skip(reason="later")\n\n\ndef test_balance_after_deposit():\n    assert deposit(10, 5) == 15\n    assert history() == ["deposit"]\n',
    }, "SKIP_ADDED"),
    case("test-moved-into-a-new-file-with-annotated-pytestmark", {
        TEST_FILE: sub(BASE_TEST, 'def test_balance_after_deposit():\n    assert deposit(10, 5) == 15\n    assert history() == ["deposit"]\n', ""),
        "tests/test_moved.py": 'import pytest\n\npytestmark: list = [pytest.mark.skip(reason="later")]\n\n\ndef test_balance_after_deposit():\n    assert deposit(10, 5) == 15\n    assert history() == ["deposit"]\n',
    }, "SKIP_ADDED"),
    case("conftest-hook-that-skips", {"tests/conftest.py": 'import pytest\n\ndef pytest_runtest_setup(item):\n    pytest.skip("x")\n'}, "SKIP_ADDED"),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), SKIPS)
def test_added_skip_in_any_spelling_is_reported(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


def test_skip_reported_once_for_a_module_marker_not_once_per_test(repo, guard_mod):
    write(repo, TEST_FILE, sub(BASE_TEST, "import pytest\n", 'import pytest\n\npytestmark = pytest.mark.skip(reason="wip")\n'))
    findings = guard_mod.guard(repo)
    assert [(f.code, f.line) for f in findings] == [("SKIP_ADDED", 1)]


CONDITIONAL = [
    case("test-under-if-gutted", {COND_FILE: sub(COND_TEST, "assert one() == 1", "pass")}, "TEST_EMPTIED"),
    case("test-under-else-deleted", {COND_FILE: sub(COND_TEST, "    def test_cond_b():\n        assert two() == 2\n", "    pass\n")}, "TEST_DELETED"),
    case("test-under-if-assert-dropped", {COND_FILE: sub(COND_TEST, "        assert two() == 2\n", "        x = two()\n")}, "ASSERTS_REDUCED"),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), CONDITIONAL)
def test_tests_defined_under_module_level_compound_statements_are_compared(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


CONFTEST_RICH = '''import pytest

collect_ignore = ["legacy.py"]


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.slow)
'''

COLLECTION = [
    case("collect-ignore-annotated", {"tests/conftest.py": "import pytest\ncollect_ignore: list = ['test_wallet.py']\n"}, "COLLECTION_NARROWED"),
    case("collect-ignore-insert", {"tests/conftest.py": "collect_ignore = []\ncollect_ignore.insert(0, 'test_wallet.py')\n"}, "COLLECTION_NARROWED"),
    case("collect-ignore-append", {"tests/conftest.py": "collect_ignore = []\ncollect_ignore.append('test_wallet.py')\n"}, "COLLECTION_NARROWED"),
    case("collect-ignore-extend", {"tests/conftest.py": "collect_ignore = []\ncollect_ignore.extend(['test_wallet.py'])\n"}, "COLLECTION_NARROWED"),
    case("collect-ignore-augmented", {"tests/conftest.py": "collect_ignore = []\ncollect_ignore += ['test_wallet.py']\n"}, "COLLECTION_NARROWED"),
    case("collect-ignore-glob", {"tests/conftest.py": "collect_ignore_glob = ['*wallet*']\n"}, "COLLECTION_NARROWED"),
    case("collect-ignore-entry-swapped", {"tests/conftest.py": sub(CONFTEST_RICH, '"legacy.py"', '"test_wallet.py"')}, "COLLECTION_NARROWED", base={"tests/conftest.py": CONFTEST_RICH}),
    case("existing-hook-body-changed", {"tests/conftest.py": sub(CONFTEST_RICH, "        item.add_marker(pytest.mark.slow)\n", "        pass\n    items[:] = [i for i in items if 'fee' not in i.name]\n")}, "COLLECTION_NARROWED", base={"tests/conftest.py": CONFTEST_RICH}),
    case("collect-file-hook-added", {"tests/conftest.py": "def pytest_collect_file(parent, file_path):\n    return None\n"}, "COLLECTION_NARROWED"),
    case("pycollect-makeitem-hook-added", {"tests/conftest.py": "def pytest_pycollect_makeitem(collector, name, obj):\n    return []\n"}, "COLLECTION_NARROWED"),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), COLLECTION)
def test_collection_narrowing_in_any_spelling_is_reported(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


COLLECTION_CLEAN = [
    case("collect-ignore-entry-removed", {"tests/conftest.py": sub(CONFTEST_RICH, 'collect_ignore = ["legacy.py"]', "collect_ignore = []")}, base={"tests/conftest.py": CONFTEST_RICH}),
    case("hook-reformatted-with-comment", {"tests/conftest.py": sub(CONFTEST_RICH, "    for item in items:\n", "    # mark everything\n    for item in items:\n")}, base={"tests/conftest.py": CONFTEST_RICH}),
    case("collect-ignore-unchanged-but-fixture-added", {"tests/conftest.py": CONFTEST_RICH + "\n\n@pytest.fixture\ndef db():\n    return 1\n"}, base={"tests/conftest.py": CONFTEST_RICH}),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), COLLECTION_CLEAN)
def test_harmless_conftest_edits_have_no_findings(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == []


# ---------------------------------------------------------------------------------------
# Config files: every file pytest and coverage read, in their precedence order
# ---------------------------------------------------------------------------------------
PYTEST_INI = "[pytest]\ntestpaths = tests\naddopts = -ra\n"
COV_ONLY_PYPROJECT = "[tool.coverage.report]\nfail_under = 90\n"
NO_COV_PYPROJECT = PYPROJECT_NO_COV


def ini(**extra_opts: str) -> str:
    return "[pytest]\ntestpaths = tests\naddopts = -ra " + extra_opts.get("addopts", "") + "\n"


CONFIG = [
    case("hidden-pytest-ini-gains-deselect", {".pytest.ini": "[pytest]\ntestpaths = tests\naddopts = -ra --deselect tests/test_wallet.py::test_x\n"}, "CONFIG_NARROWED"),
    case("hidden-pytest-toml-gains-deselect", {".pytest.toml": '[pytest]\ntestpaths = ["tests"]\naddopts = ["-ra", "--deselect", "x"]\n'}, "CONFIG_NARROWED"),
    case("pytest-toml-gains-k", {"pytest.toml": '[pytest]\ntestpaths = ["tests"]\naddopts = ["-ra", "-k", "not fee"]\n'}, "CONFIG_NARROWED"),
    case("new-pytest-ini-shadows-pyproject-and-drops-testpaths", {"pytest.ini": "[pytest]\naddopts = -ra\n"}, "CONFIG_NARROWED"),
    case("pytest-ini-gains-deselect", {"pytest.ini": ini(addopts="--deselect x")}, "CONFIG_NARROWED", base={"pytest.ini": ini()}),
    case("tox-ini-gains-k", {"tox.ini": ini(addopts="-k 'not fee'")}, "CONFIG_NARROWED", base={"tox.ini": ini(), "pyproject.toml": COV_ONLY_PYPROJECT}),
    case("setup-cfg-gains-ignore-glob", {"setup.cfg": ini(addopts="--ignore-glob=*wallet*").replace("[pytest]", "[tool:pytest]")}, "CONFIG_NARROWED", base={"setup.cfg": ini().replace("[pytest]", "[tool:pytest]"), "pyproject.toml": COV_ONLY_PYPROJECT}),
    case("setup-cfg-loses-testpaths", {"setup.cfg": "[tool:pytest]\naddopts = -ra\n"}, "CONFIG_NARROWED", base={"setup.cfg": ini().replace("[pytest]", "[tool:pytest]"), "pyproject.toml": COV_ONLY_PYPROJECT}),
    case("ignore-glob-space-form", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "--ignore-glob", "*wallet*"]')}, "CONFIG_NARROWED"),
    case("attached-short-k-in-string-addopts", {"pyproject.toml": sub(PYPROJECT, 'addopts = ["-ra"]', "addopts = \"-ra -k'not fee'\"")}, "CONFIG_NARROWED"),
    case("attached-short-k-without-space", {"pyproject.toml": sub(PYPROJECT, 'addopts = ["-ra"]', 'addopts = "-ra -knotfee"')}, "CONFIG_NARROWED"),
    case("marker-expression", {"pyproject.toml": sub(PYPROJECT, 'addopts = ["-ra"]', "addopts = \"-ra -m 'not wallet'\"")}, "CONFIG_NARROWED"),
    case("no-cov-flag", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "--no-cov"]')}, "CONFIG_NARROWED"),
    case("collect-only-flag", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "--collect-only"]')}, "CONFIG_NARROWED"),
    case("co-flag", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "--co"]')}, "CONFIG_NARROWED"),
    case("noconftest-flag", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "--noconftest"]')}, "CONFIG_NARROWED"),
    case("norecursedirs-gained", {"pyproject.toml": sub(PYPROJECT, 'addopts = ["-ra"]', 'addopts = ["-ra"]\nnorecursedirs = ["tests/wallet_dir"]')}, "CONFIG_NARROWED"),
    case("python-files-replaced", {"pyproject.toml": sub(PYPROJECT, 'addopts = ["-ra"]', 'addopts = ["-ra"]\npython_files = ["check_*.py"]')}, "CONFIG_NARROWED"),
    case("python-functions-replaced", {"pyproject.toml": sub(PYPROJECT, 'addopts = ["-ra"]', 'addopts = ["-ra"]\npython_functions = ["check"]')}, "CONFIG_NARROWED"),
    case("python-classes-replaced", {"pyproject.toml": sub(PYPROJECT, 'addopts = ["-ra"]', 'addopts = ["-ra"]\npython_classes = ["Check"]')}, "CONFIG_NARROWED"),
    case("first-pytest-config-file-added-with-a-deselect", {"pytest.ini": "[pytest]\naddopts = --deselect x\n"}, "CONFIG_NARROWED", base={"pyproject.toml": COV_ONLY_PYPROJECT}),
    case("pyproject-deleted-with-testpaths-and-floor", {"pyproject.toml": None}, "CONFIG_NARROWED", "COVERAGE_LOWERED"),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), CONFIG)
def test_config_narrowing_is_reported(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


PYPROJECT_K = sub(PYPROJECT, '"-ra"]', '"-ra", "-k", "not slow"]')
CONFIG_CLEAN = [
    case("narrowing-flag-that-was-already-in-the-base", {"pyproject.toml": PYPROJECT_K.replace("fail_under = 90", "fail_under = 95")}, base={"pyproject.toml": PYPROJECT_K}),
    case("python-files-superset", {"pyproject.toml": sub(PYPROJECT, 'addopts = ["-ra"]', 'addopts = ["-ra"]\npython_files = ["test_*.py", "*_test.py", "check_*.py"]')}),
    case("python-files-broader-glob", {"pyproject.toml": sub(PYPROJECT, 'addopts = ["-ra"]', 'addopts = ["-ra"]\npython_files = ["*.py"]')}),
    case("new-pytest-ini-that-keeps-everything", {"pytest.ini": PYTEST_INI}),
    case("config-moved-from-pyproject-to-pytest-toml", {
        "pytest.toml": '[pytest]\ntestpaths = ["tests"]\naddopts = ["-ra"]\n',
        "pyproject.toml": COV_ONLY_PYPROJECT,
    }),
    case("setup-cfg-pytest-section-shadowed-by-pyproject", {"setup.cfg": "[tool:pytest]\naddopts = -k nothing\n"}),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), CONFIG_CLEAN)
def test_harmless_config_edits_have_no_findings(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == []


COV_RC = "[report]\nfail_under = 90\n"
COV_CFG = "[coverage:report]\nfail_under = 90\n"
FLAG_PYPROJECT = '[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = "-ra --cov-fail-under 90"\n'

COVERAGE = [
    case("coveragerc-lowered", {".coveragerc": COV_RC.replace("90", "60")}, "COVERAGE_LOWERED", base={".coveragerc": COV_RC, "pyproject.toml": NO_COV_PYPROJECT}),
    case("coveragerc-floor-removed", {".coveragerc": "[report]\nshow_missing = True\n"}, "COVERAGE_LOWERED", base={".coveragerc": COV_RC, "pyproject.toml": NO_COV_PYPROJECT}),
    case("coveragerc-toml-lowered", {".coveragerc.toml": "[report]\nfail_under = 10\n"}, "COVERAGE_LOWERED", base={".coveragerc.toml": "[report]\nfail_under = 90\n", "pyproject.toml": NO_COV_PYPROJECT}),
    case("setup-cfg-lowered", {"setup.cfg": COV_CFG.replace("90", "60")}, "COVERAGE_LOWERED", base={"setup.cfg": COV_CFG, "pyproject.toml": NO_COV_PYPROJECT}),
    case("tox-ini-lowered", {"tox.ini": COV_CFG.replace("90", "60")}, "COVERAGE_LOWERED", base={"tox.ini": COV_CFG, "pyproject.toml": NO_COV_PYPROJECT}),
    case("new-coveragerc-with-zero-floor-shadows-pyproject", {".coveragerc": "[report]\nfail_under = 0\n"}, "COVERAGE_LOWERED"),
    case("new-coveragerc-toml-with-zero-floor-shadows-pyproject", {".coveragerc.toml": "[report]\nfail_under = 0\n"}, "COVERAGE_LOWERED"),
    case("floor-key-deleted-from-pyproject", {"pyproject.toml": sub(PYPROJECT, "\n[tool.coverage.report]\nfail_under = 90\n", "")}, "COVERAGE_LOWERED"),
    case("space-separated-cov-fail-under-lowered", {"pyproject.toml": FLAG_PYPROJECT.replace("90", "50")}, "COVERAGE_LOWERED", base={"pyproject.toml": FLAG_PYPROJECT}),
    case("cov-fail-under-flag-zero-overrides-the-file-floor", {"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "--cov-fail-under", "0"]')}, "COVERAGE_LOWERED"),
    case("coverage-omit-gained", {"pyproject.toml": PYPROJECT + '\n[tool.coverage.run]\nomit = ["src/wallet.py"]\n'}, "COVERAGE_LOWERED"),
    case("coveragerc-omit-gained", {".coveragerc": COV_RC + "\n[run]\nomit =\n    src/wallet.py\n"}, "COVERAGE_LOWERED", base={".coveragerc": COV_RC, "pyproject.toml": NO_COV_PYPROJECT}),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), COVERAGE)
def test_lowered_coverage_floor_is_reported(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


COVERAGE_CLEAN = [
    case("floor-moved-from-pyproject-to-coveragerc", {"pyproject.toml": NO_COV_PYPROJECT, ".coveragerc": COV_RC}),
    case("floor-moved-from-pyproject-to-setup-cfg", {"pyproject.toml": NO_COV_PYPROJECT, "setup.cfg": COV_CFG}),
    case("new-coveragerc-with-higher-floor", {".coveragerc": "[report]\nfail_under = 95\n"}),
    case("flag-raised-with-space-form", {"pyproject.toml": FLAG_PYPROJECT.replace("90", "95")}, base={"pyproject.toml": FLAG_PYPROJECT}),
    case("flag-added-where-no-floor-existed", {"pyproject.toml": sub(PYPROJECT_NO_COV, '"-ra"]', '"-ra", "--cov-fail-under=50"]')}, base={"pyproject.toml": PYPROJECT_NO_COV}),
    case("omit-removed", {"pyproject.toml": PYPROJECT}, base={"pyproject.toml": PYPROJECT + '\n[tool.coverage.run]\nomit = ["src/wallet.py"]\n'}),
    case("setup-cfg-without-coverage-section-does-not-shadow", {"setup.cfg": "[metadata]\nname = x\n"}),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), COVERAGE_CLEAN)
def test_harmless_coverage_edits_have_no_findings(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == []


def test_toml_config_without_tomllib_exits_2_loudly(repo, guard_mod, monkeypatch, capsys):
    monkeypatch.setattr(guard_mod, "tomllib", None)
    assert guard_mod.main([str(repo)]) == 2
    assert "Python 3.11" in capsys.readouterr().err


def test_unparsable_toml_config_exits_2_not_clean(repo, guard_mod, capsys):
    write(repo, "pyproject.toml", "[tool.pytest\n")
    assert guard_mod.main([str(repo)]) == 2
    assert "pyproject.toml" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------
# Paths, encodings and binary files
# ---------------------------------------------------------------------------------------
UNICODE_TEST = "tests/test_ä.py"
UNICODE_BODY = 'def test_umlaut():\n    assert name() == "ä"\n    assert size() == 3\n'
LATIN1 = b'# -*- coding: latin-1 -*-\n\ndef test_cafe():\n    assert name() == "caf\xe9"\n    assert size() == 4\n'
PNG = b"\x89PNG\r\n\x1a\n\xff\xfe\x00\x80binary"

UNICODE_BASE = {UNICODE_TEST: UNICODE_BODY}

PATHS = [
    case("non-ascii-test-file-deleted", {UNICODE_TEST: None}, "TEST_FILE_DELETED", base=UNICODE_BASE),
    case("non-ascii-test-file-gutted", {UNICODE_TEST: 'def test_umlaut():\n    assert name() == "ä"\n'}, "ASSERTS_REDUCED", base=UNICODE_BASE),
    case("non-ascii-source-gains-constant-eq", {"src/café.py": "class Money:\n    def __eq__(self, other):\n        return True\n"}, "CONSTANT_EQ"),
    case("latin-1-test-file-weakened", {"tests/test_latin.py": LATIN1.replace(b'    assert size() == 4\n', b"")}, "ASSERTS_REDUCED", base={"tests/test_latin.py": LATIN1}),
    case("latin-1-test-file-unchanged", {}, base={"tests/test_latin.py": LATIN1}),
    case("binary-snapshot-unchanged", {}, base={"tests/__snapshots__/test_img.png": PNG}),
    case("binary-snapshot-rewritten", {"tests/__snapshots__/test_img.png": PNG + b"x"}, "SNAPSHOT_CHANGED", base={"tests/__snapshots__/test_img.png": PNG}),
    case("binary-snapshot-deleted", {"tests/__snapshots__/test_img.png": None}, "SNAPSHOT_CHANGED", base={"tests/__snapshots__/test_img.png": PNG}),
    case("ambr-snapshot-deleted", {"tests/__snapshots__/test_wallet.ambr": None}, "SNAPSHOT_CHANGED", base={"tests/__snapshots__/test_wallet.ambr": "# name: t\n  1\n"}),
    case("unhashable-dict-key-literal-unchanged", {}, base={"tests/test_odd.py": "def test_odd():\n    assert f() == {[1]: 2}\n"}),
    case("unhashable-dict-key-literal-changed", {"tests/test_odd.py": "def test_odd():\n    assert f() == {[1]: 3}\n"}, "EXPECTATION_CHANGED", base={"tests/test_odd.py": "def test_odd():\n    assert f() == {[1]: 2}\n"}),
    case("test-file-undecodable-without-a-cookie", {TEST_FILE: BASE_TEST.encode() + b"\n# \xff\xfe\n"}, "SYNTAX"),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), PATHS)
def test_unusual_paths_encodings_and_binaries(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


@pytest.mark.parametrize(
    ("edits", "base", "moves", "git_mv", "expected"),
    [
        case("renamed-test-file", {}, moves=((TEST_FILE, "tests/test_wallet_renamed.py"),)),
        case("renamed-test-file-without-git-mv", {}, moves=((TEST_FILE, "tests/test_wallet_renamed.py"),), git_mv=False),
        case("test-file-moved-to-a-subdirectory", {}, moves=((TEST_FILE, "tests/unit/test_wallet.py"),)),
        case("renamed-test-file-with-an-assert-dropped", {"tests/test_wallet_renamed.py": sub(BASE_TEST, '    assert history() == ["deposit"]\n', "")}, "ASSERTS_REDUCED", moves=((TEST_FILE, "tests/test_wallet_renamed.py"),)),
        case("renamed-test-file-that-gains-a-module-skip", {"tests/test_wallet_renamed.py": sub(BASE_TEST, "import pytest\n", 'import pytest\n\npytestmark = pytest.mark.skip(reason="x")\n')}, "SKIP_ADDED", moves=((TEST_FILE, "tests/test_wallet_renamed.py"),)),
        case("deleted-and-replaced-by-an-unrelated-file", {TEST_FILE: None, "tests/test_other.py": "def test_unrelated():\n    assert totally_different() == 1\n"}, "TEST_FILE_DELETED"),
    ],
)
def test_renaming_a_test_file_is_not_a_deletion(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


CONST_EQ_SRC = "class Money:\n    def __eq__(self, other):\n        return True\n"


@pytest.mark.parametrize(
    ("edits", "base", "moves", "git_mv", "expected"),
    [
        case("moved-source-keeps-its-old-constant-eq-quiet", {}, base={"src/money.py": CONST_EQ_SRC}, moves=(("src/money.py", "src/finance/money.py"),)),
        case("moved-source-without-git-mv", {}, base={"src/money.py": CONST_EQ_SRC}, moves=(("src/money.py", "src/finance/money.py"),), git_mv=False),
        case("moved-source-that-gains-a-constant-eq", {"src/finance/money.py": "class Money:\n    def __ne__(self, other):\n        return self is not other\n\n    def __eq__(self, other):\n        return True\n"}, "CONSTANT_EQ", base={"src/money.py": "class Money:\n    def __ne__(self, other):\n        return self is not other\n\n    def __eq__(self, other):\n        return self is other\n"}, moves=(("src/money.py", "src/finance/money.py"),)),
    ],
)
def test_renamed_source_file_is_not_new_code(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


def test_non_ascii_names_are_not_quoted_in_findings(repo, guard_mod):
    write(repo, UNICODE_TEST, UNICODE_BODY)
    commit(repo, "umlaut")
    (repo / UNICODE_TEST).unlink()
    assert [f.file for f in guard_mod.guard(repo)] == [UNICODE_TEST]


def test_undecodable_test_file_is_a_finding_and_exit_1_not_a_crash(repo, guard_mod, capsys):
    write(repo, TEST_FILE, BASE_TEST.encode() + b"\n# \xff\xfe\n")
    assert guard_mod.main([str(repo)]) == 1
    assert "SYNTAX" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------
# Existing behaviours kept
# ---------------------------------------------------------------------------------------
def test_inline_snapshot_rewrite_is_an_expectation_change(repo, guard_mod):
    path = repo / "tests" / "test_snap.py"
    path.write_text("from inline_snapshot import snapshot\n\ndef test_total():\n    assert total() == snapshot(250)\n")
    commit(repo, "snap")
    path.write_text(path.read_text().replace("snapshot(250)", "snapshot(251)"))
    assert [(f.code, f.detail) for f in guard_mod.guard(repo)] == [("EXPECTATION_CHANGED", "test_total: 250 -> 251")]


def test_syrupy_snapshot_rewrite_is_flagged(repo, guard_mod):
    snap = repo / "tests" / "__snapshots__" / "test_wallet.ambr"
    snap.parent.mkdir()
    snap.write_text("# name: test_total\n  250\n# ---\n")
    commit(repo, "snap")
    snap.write_text("# name: test_total\n  251\n# ---\n")
    assert [(f.code, f.severity) for f in guard_mod.guard(repo)] == [("SNAPSHOT_CHANGED", "warning")]


def test_wrapping_an_assert_in_with_does_not_reorder_expected_literals(repo, guard_mod):
    base = "def test_two():\n    assert f() == 1\n    assert g() == 2\n"
    write(repo, "tests/test_two.py", base)
    commit(repo, "two")
    write(repo, "tests/test_two.py", "def test_two():\n    with lock:\n        assert f() == 1\n    assert g() == 2\n")
    assert guard_mod.guard(repo) == []


def test_renaming_a_test_function_reads_as_a_deletion(run):
    """The guard cannot tell a rename from delete-and-add; a human approves with --allow."""
    renamed = BASE_TEST.replace("def test_balance_after_deposit", "def test_balance_after_a_deposit_of_five")
    assert run({TEST_FILE: renamed}) == expect("TEST_DELETED")


def test_syntax_error_is_not_reported_as_mass_deletion(run):
    assert run({TEST_FILE: BASE_TEST + "\ndef test_broken(:\n"}) == expect("SYNTAX")


def test_syntax_description_does_not_claim_pytest_always_fails_to_collect(guard_mod):
    text = guard_mod.CODES["SYNTAX"][1]
    assert "helper" in text and "imported" in text


def test_uncommitted_new_file_with_a_pragma_is_read_as_current(run):
    """If untracked files were ignored this finding could not appear."""
    edits = {"tests/test_new.py": "def test_x():\n    assert f() == 1  # audit: ignore[FLOAT_EQ] ok\n"}
    assert run(edits) == expect("PRAGMA_ADDED")


def test_staged_new_file_with_a_pragma_is_read_as_current(repo, guard_mod):
    write(repo, "tests/test_new.py", "def test_x():\n    assert f() == 1  # audit: ignore[FLOAT_EQ] ok\n")
    git(repo, "add", "tests/test_new.py")
    assert [f.code for f in guard_mod.guard(repo)] == ["PRAGMA_ADDED"]


def test_pragma_text_inside_a_string_is_not_a_pragma(run):
    docs = 'def test_doc():\n    """Mention # audit: ignore[X] here."""\n    assert f() == 1\n'
    assert run({"tests/test_doc.py": docs}) == []
    base = {"tests/test_doc.py": "def test_doc():\n    assert f() == 1\n"}
    assert run({"tests/test_doc.py": docs}, base=base) == []


def test_allow_suppresses_a_code(repo, guard_mod):
    (repo / TEST_FILE).unlink()
    assert guard_mod.guard(repo, allow=frozenset({"TEST_FILE_DELETED"})) == []


def test_base_ref_option_compares_against_older_commit(repo, guard_mod):
    (repo / TEST_FILE).write_text(BASE_TEST.replace("    assert history() == [\"deposit\"]\n", ""))
    git(repo, "commit", "-qam", "weaken")
    assert [f.code for f in guard_mod.guard(repo)] == []
    assert [f.code for f in guard_mod.guard(repo, base="HEAD~1")] == ["ASSERTS_REDUCED"]


ERROR_EDITS = [
    pytest.param({TEST_FILE: None}, id="TEST_FILE_DELETED"),
    pytest.param(wallet("def test_balance_after_deposit", "def test_balance_after_deposit_renamed"), id="TEST_DELETED"),
    pytest.param(wallet("    assert history() == [\"deposit\"]\n", ""), id="ASSERTS_REDUCED"),
    pytest.param(wallet(FEE_LINE, "assert True"), id="ASSERT_TRIVIALIZED"),
    pytest.param(wallet(FEE_LINE, "assert fee(1000)"), id="ASSERT_WEAKENED"),
    pytest.param(wallet("def test_balance_after_deposit():\n", "@pytest.mark.skip\ndef test_balance_after_deposit():\n"), id="SKIP_ADDED"),
    pytest.param({"pyproject.toml": sub(PYPROJECT, '"-ra"]', '"-ra", "-k", "x"]')}, id="CONFIG_NARROWED"),
    pytest.param({"pyproject.toml": sub(PYPROJECT, "fail_under = 90", "fail_under = 1")}, id="COVERAGE_LOWERED"),
    pytest.param(extra("(5000, 0), ", ""), id="CASES_REDUCED"),
    pytest.param({TEST_FILE: BASE_TEST + "\ndef test_broken(:\n"}, id="SYNTAX"),
    pytest.param({"tests/conftest.py": "collect_ignore = ['x']\n"}, id="COLLECTION_NARROWED"),
]


@pytest.mark.parametrize("edits", ERROR_EDITS)
def test_every_error_code_makes_the_cli_exit_1(repo, guard_mod, edits):
    for rel, content in edits.items():
        write(repo, rel, content)
    assert guard_mod.main([str(repo)]) == 1


def test_cli_exit_codes_and_json(repo, guard_mod, capsys):
    assert guard_mod.main([str(repo)]) == 0
    assert capsys.readouterr().out.startswith("clean:")
    write(repo, TEST_FILE, sub(BASE_TEST, FEE_LINE, "assert True"))
    assert guard_mod.main([str(repo), "--format", "json"]) == 1
    line = BASE_TEST.splitlines().index("    def test_fee_is_one_percent(self):") + 1
    assert json.loads(capsys.readouterr().out) == [{
        "file": TEST_FILE, "line": line, "code": "ASSERT_TRIVIALIZED", "severity": "error",
        "detail": "TestFees.test_fee_is_one_percent: 1 assertion(s) that cannot fail",
    }]


def test_github_format_prints_annotations(repo, guard_mod, capsys):
    write(repo, TEST_FILE, sub(BASE_TEST, FEE_LINE, "assert True"))
    write(repo, EXTRA_FILE, sub(EXTRA_TEST, "(5000, 0)", "(5001, 0)"))
    assert guard_mod.main([str(repo), "--format", "github"]) == 1
    line = BASE_TEST.splitlines().index("    def test_fee_is_one_percent(self):") + 1
    extra_line = EXTRA_TEST.splitlines().index("def test_param_rows(a, b):") + 1
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"::warning file={EXTRA_FILE},line={extra_line}::EXPECTATION_CHANGED test_param_rows: 1 parametrize row(s) replaced: (5000, 0)",
        f"::error file={TEST_FILE},line={line}::ASSERT_TRIVIALIZED TestFees.test_fee_is_one_percent: 1 assertion(s) that cannot fail",
    ]


def test_github_format_escapes_newlines_and_commas(guard_mod, capsys):
    finding = guard_mod.Finding("a,b.py", 3, "SYNTAX", "bad\nthing 100%", "error")
    assert guard_mod.github_annotation(finding) == "::error file=a%2Cb.py,line=3::SYNTAX bad%0Athing 100%25"


def test_list_codes_prints_every_code_with_its_severity(guard_mod, capsys):
    assert guard_mod.main(["--list-codes"]) == 0
    printed = {line.split()[0]: line.split()[1] for line in capsys.readouterr().out.splitlines()}
    assert printed == SEVERITY


def test_warning_only_passes_unless_warnings_as_errors(repo, guard_mod):
    write(repo, TEST_FILE, sub(BASE_TEST, FEE_LINE, "assert fee(1000) == 11"))
    assert guard_mod.main([str(repo)]) == 0
    assert guard_mod.main([str(repo), "--warnings-as-errors"]) == 1


def test_not_a_repository_exits_2(tmp_path, guard_mod, capsys):
    assert guard_mod.main([str(tmp_path)]) == 2
    assert capsys.readouterr().err.startswith("diff_guard:")


def test_unknown_allow_code_exits_2(repo, guard_mod, capsys):
    assert guard_mod.main([str(repo), "--allow", "NOPE"]) == 2
    assert "NOPE" in capsys.readouterr().err


def test_repeated_allow_options_all_apply(repo, guard_mod):
    (repo / TEST_FILE).unlink()
    write(repo, EXTRA_FILE, EXTRA_TEST + "\ndef test_broken(:\n")
    assert guard_mod.main([str(repo)]) == 1
    assert guard_mod.main([str(repo), "--allow", "TEST_FILE_DELETED", "--allow", "SYNTAX"]) == 0


def test_unknown_base_exits_2_and_names_the_ref(repo, guard_mod, capsys):
    assert guard_mod.main([str(repo), "--base", "no-such-ref"]) == 2
    assert "no-such-ref" in capsys.readouterr().err


def test_a_repository_with_nothing_to_compare_exits_2_not_clean(tmp_path, guard_mod, require_git, capsys):
    root = make_repo(tmp_path / "bare", {"README.md": "hi\n"})
    assert guard_mod.main([str(root)]) == 2
    captured = capsys.readouterr()
    assert "nothing to compare" in captured.err
    assert "clean" not in captured.out


def test_missing_git_exits_2_with_a_message(repo, guard_mod, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(repo / "no-such-bin"))
    assert guard_mod.main([str(repo)]) == 2
    assert "git" in capsys.readouterr().err


SRC = "src/wallet.py"
BASE_SRC = "class Money:\n    def __init__(self, cents):\n        self.cents = cents\n\n\ndef fee(cents):\n    return cents // 100\n"
NEG_TEST = "def test_negative():\n    assert fee(-41) == -42\n    assert fee_f(-3.5) == 1.5\n"


@pytest.fixture
def repo_with_src(repo):
    write(repo, SRC, BASE_SRC)
    write(repo, "tests/test_neg.py", NEG_TEST)
    commit(repo, "src")
    return repo


def source_findings(repo, guard_mod, new_src: str) -> list[tuple[str, str, str]]:
    write(repo, SRC, new_src)
    return [(f.code, f.severity, f.detail) for f in guard_mod.guard(repo)]


def test_constant_eq_in_source_is_flagged(repo_with_src, guard_mod):
    new = BASE_SRC.replace("        self.cents = cents\n", "        self.cents = cents\n\n    def __eq__(self, other):\n        return True\n")
    assert source_findings(repo_with_src, guard_mod, new) == [("CONSTANT_EQ", "warning", "Money.__eq__ returns a constant")]


@pytest.mark.parametrize("body", [
    "        return not False\n",
    "        return True if other else True\n",
    "        return bool(1)\n",
    "        x = other\n        return True\n",
    "        return -1\n",
])
def test_constant_eq_in_other_spellings_is_flagged(repo_with_src, guard_mod, body):
    new = BASE_SRC.replace("        self.cents = cents\n", "        self.cents = cents\n\n    def __eq__(self, other):\n" + body)
    assert [c for c, _, _ in source_findings(repo_with_src, guard_mod, new)] == ["CONSTANT_EQ"]


def test_constant_eq_as_a_lambda_is_flagged(repo_with_src, guard_mod):
    new = BASE_SRC.replace("        self.cents = cents\n", "        self.cents = cents\n\n    __eq__ = lambda self, other: True\n")
    assert [c for c, _, _ in source_findings(repo_with_src, guard_mod, new)] == ["CONSTANT_EQ"]


@pytest.mark.parametrize("body", [
    "        return self.cents == other.cents\n",
    "        if not isinstance(other, Money):\n            return NotImplemented\n        return self.cents == other.cents\n",
])
def test_a_real_eq_is_not_flagged(repo_with_src, guard_mod, body):
    new = BASE_SRC.replace("        self.cents = cents\n", "        self.cents = cents\n\n    def __eq__(self, other):\n" + body)
    assert source_findings(repo_with_src, guard_mod, new) == []


def test_special_casing_a_test_input_is_flagged(repo_with_src, guard_mod):
    write(repo_with_src, "tests/test_big.py", "def test_big():\n    assert fee(1000) == 10\n")
    commit(repo_with_src, "big")
    new = BASE_SRC.replace("    return cents // 100\n", "    if cents == 1000:\n        return 10\n    return cents // 100\n")
    assert source_findings(repo_with_src, guard_mod, new) == [("SPECIAL_CASE", "warning", "compares with 1000, which a test also uses")]


@pytest.mark.parametrize(("condition", "value"), [
    ("cents == -41", "-41"),
    ("cents != -3.5", "-3.5"),
    ("-41 == cents", "-41"),
    ("cents in (-41, 7)", "-41"),
    ("cents not in (-41, 7)", "-41"),
])
def test_special_casing_with_negative_or_reversed_comparisons_is_flagged(repo_with_src, guard_mod, condition, value):
    new = BASE_SRC.replace("    return cents // 100\n", f"    if {condition}:\n        return 0\n    return cents // 100\n")
    assert source_findings(repo_with_src, guard_mod, new) == [("SPECIAL_CASE", "warning", f"compares with {value}, which a test also uses")]


def test_comparison_with_a_literal_no_test_uses_is_not_flagged(repo_with_src, guard_mod):
    new = BASE_SRC.replace("    return cents // 100\n", "    if cents == 77777:\n        return 0\n    return cents // 100\n")
    assert source_findings(repo_with_src, guard_mod, new) == []


def test_comparison_with_a_trivial_literal_is_not_flagged(repo_with_src, guard_mod):
    new = BASE_SRC.replace("    return cents // 100\n", "    if cents == 0:\n        return 0\n    return cents // 100\n")
    assert source_findings(repo_with_src, guard_mod, new) == []


def test_source_warnings_do_not_fail_the_cli(repo_with_src, guard_mod):
    new = BASE_SRC.replace("    return cents // 100\n", "    if cents == -41:\n        return 0\n    return cents // 100\n")
    write(repo_with_src, SRC, new)
    assert guard_mod.main([str(repo_with_src)]) == 0
    assert guard_mod.main([str(repo_with_src), "--warnings-as-errors"]) == 1


def test_unparsable_changed_source_file_is_reported(repo_with_src, guard_mod):
    (code, severity, detail), = source_findings(repo_with_src, guard_mod, "def fee(:\n")
    assert (code, severity) == ("SYNTAX", "error")
    assert detail.startswith(f"{SRC} does not parse")


CASES = '@pytest.mark.parametrize(("a", "b"), [(1, 2), (5000, 0), (3, 4)])\ndef test_rows(a, b):\n    assert f(a) == b\n'


def test_dropping_a_parametrize_row_is_flagged(repo, guard_mod):
    path = repo / "tests" / "test_rows.py"
    path.write_text("import pytest\n\n" + CASES)
    commit(repo, "rows")
    path.write_text(path.read_text().replace("(5000, 0), ", ""))
    assert [(f.code, f.detail) for f in guard_mod.guard(repo)] == [("CASES_REDUCED", "test_rows: 3 parametrize rows -> 2")]


def test_adding_a_parametrize_row_is_fine(repo, guard_mod):
    path = repo / "tests" / "test_rows.py"
    path.write_text("import pytest\n\n" + CASES)
    commit(repo, "rows")
    path.write_text(path.read_text().replace("(3, 4)]", "(3, 4), (7, 8)]"))
    assert guard_mod.guard(repo) == []


def test_deleting_a_raises_exc_check_is_reported(repo, guard_mod):
    check = "    with pytest.RaisesExc(ValueError, match=\"negative\"):\n        deposit(10, -1)\n"
    path = repo / TEST_FILE
    path.write_text(BASE_TEST + "\n\ndef test_rejects_negative():\n" + check)
    git(repo, "commit", "-qam", "add RaisesExc test")
    path.write_text(path.read_text().replace(check, "    deposit(10, -1)\n"))
    assert [f.code for f in guard_mod.guard(repo)] == ["ASSERTS_REDUCED"]



# ---------------------------------------------------------------------------------------
# Two warnings: a pytest_generate_tests hook that is new or changed, and a skip reached
# through a helper. Each has legitimate uses, so neither fails the run by itself.
# ---------------------------------------------------------------------------------------
HOOK = '''import pytest


def pytest_generate_tests(metafunc):
    """Run every wallet test with each currency."""
    if "currency" in metafunc.fixturenames:
        metafunc.parametrize("currency", ["USD", "EUR", "GBP"])
'''
HOOK_PLUS_FUNCTION = HOOK + "\n\ndef make_wallet():\n    return {}\n"
CONFTEST = "tests/conftest.py"

HOOK_DETECTED = [
    case("hook-added-to-conftest", {CONFTEST: HOOK}, "PARAM_HOOK_CHANGED"),
    case("hook-added-to-a-test-module", {TEST_FILE: BASE_TEST + "\n\ndef pytest_generate_tests(metafunc):\n    metafunc.parametrize('x', [1])\n"}, "PARAM_HOOK_CHANGED"),
    case("hook-body-changed", {CONFTEST: sub(HOOK, '["USD", "EUR", "GBP"]', '["USD"]')}, "PARAM_HOOK_CHANGED", base={CONFTEST: HOOK}),
    case("hook-condition-changed", {CONFTEST: sub(HOOK, '"currency" in', '"nothing" in')}, "PARAM_HOOK_CHANGED", base={CONFTEST: HOOK}),
    case("hook-decorator-changed", {CONFTEST: sub(HOOK, "def pytest_generate_tests", "@pytest.hookimpl(tryfirst=True)\ndef pytest_generate_tests")}, "PARAM_HOOK_CHANGED", base={CONFTEST: HOOK}),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), HOOK_DETECTED)
def test_a_new_or_changed_generate_hook_is_reported_as_a_warning(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


HOOK_CLEAN = [
    case("hook-unchanged", {}, base={CONFTEST: HOOK}),
    case("hook-unchanged-next-to-a-new-function", {CONFTEST: HOOK_PLUS_FUNCTION}, base={CONFTEST: HOOK}),
    case("hook-reformatted", {CONFTEST: sub(HOOK, 'metafunc.parametrize("currency", ["USD", "EUR", "GBP"])', "metafunc.parametrize(\n            'currency',\n            ['USD', 'EUR', 'GBP'],\n        )")}, base={CONFTEST: HOOK}),
    case("hook-comment-added", {CONFTEST: sub(HOOK, '    if "currency"', '    # only wallet tests use it\n    if "currency"')}, base={CONFTEST: HOOK}),
    case("hook-docstring-changed", {CONFTEST: sub(HOOK, "Run every wallet test with each currency.", "Parametrize currency.")}, base={CONFTEST: HOOK}),
    case("hook-docstring-removed", {CONFTEST: sub(HOOK, '    """Run every wallet test with each currency."""\n', "")}, base={CONFTEST: HOOK}),
    case("hook-removed", {CONFTEST: "import pytest\n"}, base={CONFTEST: HOOK}),
    case("unrelated-function-added-to-conftest", {CONFTEST: "import pytest\n\n\ndef make_wallet():\n    return {}\n"}),
    case("hook-moves-with-its-conftest", {}, base={CONFTEST: HOOK}, moves=((CONFTEST, "tests/sub/conftest.py"),)),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), HOOK_CLEAN)
def test_an_unchanged_or_reformatted_generate_hook_has_no_findings_at_all(run, edits, base, moves, git_mv, expected):
    assert expected == []
    assert run(edits, base, moves, git_mv) == []


GPU_FILE = "tests/test_gpu.py"
SKIP_BODY = '    if not has_gpu():\n        pytest.skip("no gpu")\n'
PLAIN_BODY = "    return has_gpu()\n"
CONFTEST_HELPER = "import pytest\n\n\ndef require_gpu():\n" + SKIP_BODY


def helper_module(helper_body: str | None, calls: tuple[str, ...] = (), *, extra: str = "", imports: str = "") -> str:
    """tests/test_gpu.py: an optional helper `require_gpu`, tests a and b; `calls` names the tests that call it."""

    def test(name: str, value: int) -> str:
        gate = "    require_gpu()\n" if name in calls else ""
        return f"def test_{name}():\n{gate}    assert compute({value}) == {value}\n"

    helper = f"def require_gpu():\n{helper_body}\n\n" if helper_body is not None else ""
    return f"import pytest\n{imports}\n\n{helper}{test('a', 1)}\n\n{test('b', 2)}{extra}"


def skip_case(case_id: str, base_module: str, edited: str | None, *codes: str, extra_base: dict | None = None, extra_edits: dict | None = None):
    edits = {GPU_FILE: edited if edited is not None else base_module, **(extra_edits or {})}
    return case(case_id, edits, *codes, base={GPU_FILE: base_module, **(extra_base or {})})


FIXTURE_BASE = '''import pytest


@pytest.fixture
def gpu():
    pytest.skip("no gpu")


def test_a():
    assert compute(1) == 1


def test_b():
    assert compute(2) == 2
'''
GATED = sub(helper_module(SKIP_BODY), "def test_a():\n", "def test_a():\n    gate()\n") + "\n\ndef gate():\n    require_gpu()\n"

HELPER_DETECTED = [
    skip_case("existing-helper-newly-called", helper_module(SKIP_BODY), helper_module(SKIP_BODY, ("a",)), "SKIP_VIA_HELPER"),
    skip_case("new-helper-newly-called", helper_module(None), helper_module(SKIP_BODY, ("a",)), "SKIP_VIA_HELPER"),
    skip_case("existing-helper-gains-a-skip", helper_module(PLAIN_BODY, ("a", "b")), helper_module(SKIP_BODY, ("a", "b")), "SKIP_VIA_HELPER"),
    skip_case("skip-reached-through-another-helper", helper_module(SKIP_BODY), GATED, "SKIP_VIA_HELPER"),
    skip_case("helper-gains-a-call-to-a-skipping-helper", sub(GATED, "def gate():\n    require_gpu()\n", "def gate():\n    return 1\n"), GATED, "SKIP_VIA_HELPER"),
    skip_case("helper-skips-with-xfail", helper_module(PLAIN_BODY), helper_module('    pytest.xfail("x")\n', ("a",)), "SKIP_VIA_HELPER"),
    skip_case("helper-skips-with-importorskip", helper_module(PLAIN_BODY), helper_module('    return pytest.importorskip("torch")\n', ("a",)), "SKIP_VIA_HELPER"),
    skip_case("helper-raises-skiptest", helper_module(PLAIN_BODY), helper_module('    raise unittest.SkipTest("x")\n', ("a",)), "SKIP_VIA_HELPER"),
    skip_case("helper-skips-through-an-imported-name", helper_module(PLAIN_BODY), helper_module('    bail("x")\n', ("a",), imports="from pytest import skip as bail"), "SKIP_VIA_HELPER"),
    skip_case("helper-from-conftest-newly-called", helper_module(None), helper_module(None, imports="from conftest import require_gpu").replace("def test_a():\n", "def test_a():\n    require_gpu()\n", 1), "SKIP_VIA_HELPER", extra_base={CONFTEST: CONFTEST_HELPER}),
    skip_case("fixture-newly-requested", FIXTURE_BASE, sub(FIXTURE_BASE, "def test_a():", "def test_a(gpu):"), "SKIP_VIA_HELPER"),
    skip_case("fixture-newly-requested-with-usefixtures", FIXTURE_BASE, sub(FIXTURE_BASE, "def test_a():", '@pytest.mark.usefixtures("gpu")\ndef test_a():'), "SKIP_VIA_HELPER"),
    skip_case("fixture-gains-a-skip", sub(sub(FIXTURE_BASE, '    pytest.skip("no gpu")', "    return 1"), "def test_a():", "def test_a(gpu):"), sub(FIXTURE_BASE, "def test_a():", "def test_a(gpu):"), "SKIP_VIA_HELPER"),
    skip_case("fixture-conftest-newly-requested", helper_module(None), sub(FIXTURE_BASE, 'import pytest\n\n\n@pytest.fixture\ndef gpu():\n    pytest.skip("no gpu")\n\n\n', "import pytest\n\n\n").replace("def test_a():", "def test_a(gpu):"), "SKIP_VIA_HELPER", extra_base={CONFTEST: "import pytest\n\n\n@pytest.fixture\ndef gpu():\n    pytest.skip('no gpu')\n"}),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), HELPER_DETECTED)
def test_a_skip_reached_through_a_helper_is_reported_as_a_warning(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


HELPER_NOT_REPORTED = [
    skip_case("helper-that-cannot-skip-newly-called", helper_module(PLAIN_BODY), helper_module(PLAIN_BODY, ("a",))),
    skip_case("skipping-helper-that-no-test-calls", helper_module(None), helper_module(SKIP_BODY)),
    skip_case("test-already-called-the-skipping-helper", helper_module(SKIP_BODY, ("a",)), helper_module(SKIP_BODY, ("a",), extra="\n\ndef test_c():\n    assert compute(3) == 3\n")),
    skip_case("a-new-test-may-call-a-skipping-helper", helper_module(SKIP_BODY), helper_module(SKIP_BODY, extra="\n\ndef test_c():\n    require_gpu()\n    assert compute(3) == 3\n")),
    skip_case("call-to-a-name-from-another-module-is-not-followed", helper_module(None), helper_module(None, imports="import other").replace("def test_a():\n", "def test_a():\n    other.require_gpu()\n", 1)),
    skip_case("skip-in-a-nested-function-that-is-never-called", helper_module(None), helper_module('    def inner():\n        pytest.skip("x")\n    return 1\n', ("a",))),
    skip_case("test-already-requested-the-skipping-fixture", sub(FIXTURE_BASE, "def test_a():", "def test_a(gpu):"), sub(sub(FIXTURE_BASE, "def test_a():", "def test_a(gpu):"), "    assert compute(1) == 1\n", "    assert compute(1) == 1\n    assert compute(0) == 0\n")),
    skip_case("autouse-fixture-is-skip-added-only", FIXTURE_BASE, sub(FIXTURE_BASE, "@pytest.fixture\ndef gpu", "@pytest.fixture(autouse=True)\ndef gpu"), "SKIP_ADDED"),
    case("conftest-autouse-fixture-is-skip-added-only", {CONFTEST: "import pytest\n\n\n@pytest.fixture(autouse=True)\ndef gate():\n    pytest.skip('x')\n"}, "SKIP_ADDED"),
    case("new-conftest-helper-that-skips-is-skip-added-only", {CONFTEST: CONFTEST_HELPER, GPU_FILE: helper_module(None, imports="from conftest import require_gpu").replace("def test_a():\n", "def test_a():\n    require_gpu()\n", 1)}, "SKIP_ADDED", base={GPU_FILE: helper_module(None, imports="from conftest import require_gpu")}),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), HELPER_NOT_REPORTED)
def test_a_skip_that_is_not_new_or_not_reachable_is_not_reported_as_a_helper_skip(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


def _helper_repo(repo, base_module: str, edited: str):
    write(repo, GPU_FILE, base_module)
    commit(repo, "gpu")
    write(repo, GPU_FILE, edited)
    return repo


def test_a_helper_that_gains_a_skip_is_one_finding_naming_how_many_tests_call_it(repo, guard_mod):
    _helper_repo(repo, helper_module(PLAIN_BODY, ("a", "b")), helper_module(SKIP_BODY, ("a", "b")))
    [finding] = guard_mod.guard(repo)
    assert (finding.code, finding.severity, finding.file, finding.line) == ("SKIP_VIA_HELPER", "warning", GPU_FILE, 4)
    assert "require_gpu" in finding.detail
    assert "2 tests" in finding.detail
    assert "confirm the skip condition is real" in finding.detail


def test_a_test_that_gains_a_helper_call_is_reported_at_the_test_and_names_the_helper(repo, guard_mod):
    _helper_repo(repo, helper_module(SKIP_BODY), helper_module(SKIP_BODY, ("a",)))
    [finding] = guard_mod.guard(repo)
    assert (finding.code, finding.file, finding.line) == ("SKIP_VIA_HELPER", GPU_FILE, 9)
    assert "test_a" in finding.detail
    assert "require_gpu" in finding.detail
    assert "confirm the skip condition is real" in finding.detail


def test_the_generate_hook_warning_says_what_to_check(repo, guard_mod):
    write(repo, CONFTEST, HOOK)
    [finding] = guard_mod.guard(repo)
    assert (finding.code, finding.file, finding.line) == ("PARAM_HOOK_CHANGED", CONFTEST, 4)
    assert "pytest_generate_tests" in finding.detail
    assert "confirm no cases were dropped" in finding.detail


ARRANGE = {
    "PARAM_HOOK_CHANGED": lambda repo: write(repo, CONFTEST, HOOK),
    "SKIP_VIA_HELPER": lambda repo: _helper_repo(repo, helper_module(SKIP_BODY), helper_module(SKIP_BODY, ("a",))),
}


@pytest.mark.parametrize("code", list(ARRANGE))
def test_the_new_warnings_pass_unless_warnings_as_errors_and_can_be_allowed(repo, guard_mod, code):
    ARRANGE[code](repo)
    assert [f.code for f in guard_mod.guard(repo)] == [code]
    assert guard_mod.main([str(repo)]) == 0
    assert guard_mod.main([str(repo), "--warnings-as-errors"]) == 1
    assert guard_mod.guard(repo, allow=frozenset({code})) == []
    assert guard_mod.main([str(repo), "--allow", code, "--warnings-as-errors"]) == 0


def test_the_new_codes_text_tells_the_reader_what_to_check(guard_mod):
    assert "confirm no cases were dropped" in guard_mod.CODES["PARAM_HOOK_CHANGED"][1]
    assert "confirm the skip condition is real" in guard_mod.CODES["SKIP_VIA_HELPER"][1]


# ---------------------------------------------------------------------------
# Follow-up to the sequential mutation run of 2026-11-21: behaviours whose mutants survived.
# ---------------------------------------------------------------------------

REACHABLE_CHECKS = [
    ("if c:\n        pass\n    else:\n        assert a == 1", 1),  # an else branch that can run counts
    ("if True:\n        pass\n    else:\n        assert a == 1", 0),  # the else of `if True` cannot run
    ("if c:\n        return\n    assert a == 1", 1),  # only one branch exits, so the rest can run
    ("if c:\n        pass\n    else:\n        return\n    assert a == 1", 1),
    ("if c:\n        return\n    else:\n        return\n    assert a == 1", 0),  # both branches exit
    ("if False:\n        pass\n    else:\n        return\n    assert a == 1", 0),  # the live branch exits
    ("with x:\n        if True:\n            return\n    assert a == 1", 0),  # an exit two blocks deep ends the test
    ("if True:\n        with x:\n            return\n    assert a == 1", 0),
    ("while False:\n        assert a == 1", 0),
    ("while c:\n        assert a == 1", 1),
    ("for i in x:\n        if True:\n            break\n        assert a == 1", 0),  # nothing after an unconditional break runs
    ("for i in x:\n        if True:\n            continue\n        assert a == 1", 0),
    ("try:\n        assert a == 1\n    except* ValueError:\n        pass", 1),  # try/except* blocks are scanned like try/except
]


@pytest.mark.parametrize(("body", "expected"), REACHABLE_CHECKS)
def test_unit_only_checks_that_can_run_are_counted(guard_mod, body, expected):
    facts, _, _ = guard_mod.facts_for_tests(f"def test_a():\n    {body}\n")
    assert facts["test_a"].checks == expected


@pytest.mark.parametrize(("error", "words"), [
    (FileNotFoundError("no such file: 'x'"), "cannot run git or read a file"),
    (PermissionError("denied"), "I/O error"),
    (RuntimeError("planted bug"), "internal error, nothing was checked: RuntimeError: planted bug"),
])
def test_an_error_inside_the_guard_exits_2_with_a_message(repo, guard_mod, monkeypatch, capsys, error, words):
    def boom(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(guard_mod, "guard", boom)
    assert guard_mod.main([str(repo)]) == 2
    captured = capsys.readouterr()
    assert words in captured.err
    assert captured.out == ""


def test_deleting_the_only_file_the_guard_reads_is_a_finding_not_nothing_to_compare(tmp_path, guard_mod, require_git):
    root = make_repo(tmp_path / "one", {TEST_FILE: BASE_TEST})
    (root / TEST_FILE).unlink()
    assert sorted((f.code, f.severity) for f in guard_mod.guard(root)) == expect("TEST_FILE_DELETED")


def test_a_first_test_file_in_a_repository_that_had_none_is_clean(tmp_path, guard_mod, require_git):
    root = make_repo(tmp_path / "none", {"README.md": "hi\n"})
    write(root, TEST_FILE, BASE_TEST)
    assert guard_mod.guard(root) == []


NATIVE_PYPROJECT = '[tool.pytest]\ntestpaths = ["tests"]\naddopts = ["-ra"]\n'

CONFIG_FOLLOW_UP = [
    case("native-tool-pytest-table-gains-deselect", {"pyproject.toml": sub(NATIVE_PYPROJECT, '"-ra"]', '"-ra", "--deselect", "x"]')}, "CONFIG_NARROWED", base={"pyproject.toml": NATIVE_PYPROJECT}),
    case("pytest-ini-without-a-section-shadows-pyproject", {"pytest.ini": "# empty on purpose\n"}, "CONFIG_NARROWED"),
]


@pytest.mark.parametrize(("edits", "base", "moves", "git_mv", "expected"), CONFIG_FOLLOW_UP)
def test_config_read_from_the_file_pytest_would_use(run, edits, base, moves, git_mv, expected):
    assert run(edits, base, moves, git_mv) == expected


SPECIAL = "    if cents == 1000:\n        return 10\n"


@pytest.fixture
def repo_with_special_case(repo_with_src):
    write(repo_with_src, "tests/test_big.py", "def test_big():\n    assert fee(1000) == 10\n")
    write(repo_with_src, SRC, BASE_SRC.replace("    return cents // 100\n", SPECIAL + "    return cents // 100\n"))
    commit(repo_with_src, "special case already in the base")
    return repo_with_src


def test_a_special_case_already_in_the_base_is_not_reported_again(repo_with_special_case, guard_mod):
    new = BASE_SRC.replace("    return cents // 100\n", SPECIAL + "    return cents // 100  # unrelated edit\n")
    assert source_findings(repo_with_special_case, guard_mod, new) == []


def test_a_second_copy_of_a_special_case_in_the_base_is_reported_once(repo_with_special_case, guard_mod):
    new = BASE_SRC.replace("    return cents // 100\n", SPECIAL + SPECIAL + "    return cents // 100\n")
    assert source_findings(repo_with_special_case, guard_mod, new) == [("SPECIAL_CASE", "warning", "compares with 1000, which a test also uses")]
