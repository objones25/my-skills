#!/usr/bin/env python3
"""Static audit of a pytest suite against the pytest-expert hard gates.

Pure stdlib (ast, tokenize, tomllib). It never imports the code under test, so it
is safe on any repo and fast enough for a pre-commit hook or a Claude Code hook.
It reports 26 rule codes (--list-codes prints them with their severity).

Usage:
    python audit_tests.py tests/
    python audit_tests.py tests/test_wallet.py
    python audit_tests.py tests/ --ignore PRINT,BRANCHING --ignore SLEEP
    python audit_tests.py tests/ --format json|github|sarif
    python audit_tests.py --list-codes

Inline suppression (a reason after the bracket is required):
    x = compute()  # audit: ignore[FLOAT_EQ] exact binary fraction
    def test_x():  # audit: ignore[NO_ASSERT] smoke test: must not raise
    # audit: ignore-file[PRINT] demo module

A pragma that suppresses nothing is itself reported (SUPPRESSION_INERT, a warning): an unknown or
misspelt code (the nearest valid one is suggested), empty brackets, a lower-case code, ALL, a code that
pragmas cannot suppress, or a well-formed pragma with no matching finding in its line, scope or file
(stale: the finding went away and the pragma now hides the next one). SUPPRESSION_INERT and
SUPPRESSION_NO_REASON cannot be suppressed by a pragma; --ignore works for them like for any code.

Assertion helpers (NO_ASSERT only):
    python audit_tests.py tests/ --assert-helper 'check_*' --assert-helper helpers.expect_ok
    # pyproject.toml:  [tool.pytest-expert]
    #                  assert_helpers = ["check_*", "verify_invoice"]
A call to a function matching a pattern counts as an assertion for NO_ASSERT and for nothing else: a
test that only calls the helper is not NO_ASSERT, but WEAK_ASSERT and every other rule see what they
saw before. Patterns are fnmatch globs, case-sensitive. A pattern without a dot is matched against
the final name of the callee (`check_*` matches `check_x(r)`, `self.check_x(r)`, `helpers.check_x(r)`
and an import alias of one); a pattern with a dot against the whole dotted callee, as written or with
the import alias resolved (`helpers.expect_*` matches `helpers.expect_ok(r)` and, after
`import helpers as h`, `h.expect_ok(r)`). The call must be able to run, by the same rules as `assert`:
not after an unconditional return/raise, not in a nested function nothing calls, not in a lambda.
A pattern that would match every call (`*`, `?`, `*.*`, `[a-z]`) or an empty one is rejected with exit
2. The CLI patterns add to the configured ones; the patterns in effect are shown in the text summary,
in the JSON (`{"findings": [...], "assert_helpers": [...]}`, a plain list when none are in effect), in
the SARIF run properties (`assertHelpers`) and as a GitHub `::notice`. On Python < 3.11 the TOML table
is not read and a warning says so.

The path is resolved once, so `.`, a relative path and an absolute path give the same answer.
Everything the audit reads from outside the test files (pytest config, seeding conftest.py,
integration/e2e directory names) is looked up only from the audited directory up to the project
root: the nearest directory with a pytest config file or a .git, else the audited directory.

Exit status: 0 = checked and clean, 1 = an error-severity finding remains (or any warning, with
--warnings-as-errors), 2 = could not check (bad arguments, no test files found, an unreadable
file or config, an internal error). Exit 2 is never reported as clean.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import difflib
import fnmatch
import io
import json
import math
import operator
import os
import re
import sys
import tokenize
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import quote

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

ERROR, WARNING = "error", "warning"

CODES: dict[str, tuple[str, str]] = {
    "NO_ASSERT": (ERROR, "test contains no reachable assert, pytest.raises/warns/fail that runs, or real mock assertion"),
    "ALWAYS_TRUE": (ERROR, "assertion can never fail (truthy literal, `x or True`, 1 == 1, non-empty tuple/list/dict/set)"),
    "BARE_COMPARISON": (ERROR, "comparison used as a statement; it is computed and thrown away, add assert"),
    "NAN_COMPARE": (ERROR, "== / != against NaN is always False / True; use math.isnan or nan_ok"),
    "FLOAT_EQ": (ERROR, "float value compared with == or != (or `in` a float list); use pytest.approx"),
    "SWALLOWED_FAILURE": (ERROR, "except AssertionError/Exception/bare, or contextlib.suppress of them, without re-raise; failures vanish"),
    "FAKE_MOCK_ASSERT": (ERROR, (
        "mock assertion that cannot fail: an assert_* method named without parentheses, or a misspelt "
        "assert_* / called_once / not_called name (those only pass silently on Mock(unsafe=True), non-Mock "
        "objects, and called_once-style names on Python < 3.12; a default Mock raises AttributeError)"
    )),
    "BROAD_RAISES": (ERROR, "pytest.raises on Exception/BaseException with no match= (or a match that matches anything); passes on the wrong error"),
    "WEAK_ASSERT": (WARNING, "every assertion is existence-only (is not None, isinstance, len > 0)"),
    "SLEEP": (ERROR, "time.sleep in a test or fixture; slow and flaky, inject a clock"),
    "NETWORK": (ERROR, "network call (requests/httpx/urllib/socket/aiohttp/http.client) in a test without an integration or e2e marker or directory"),
    "UNSEEDED_RANDOM": (ERROR, "unseeded randomness and no seeding conftest; failures become unreproducible"),
    "PRINT": (WARNING, "print() in a test; assert instead, or use capsys/caplog"),
    "BRANCHING": (ERROR, "if/while or for-over-cases in a test body; use parametrize"),
    "ENV_MUTATION": (ERROR, "os.environ mutated directly; use monkeypatch.setenv/delenv"),
    "SYS_PATH": (ERROR, "sys.path mutated in a test; fix packaging or use monkeypatch.syspath_prepend"),
    "SKIP_NO_REASON": (ERROR, "skip/skipif/xfail without a reason (or an empty one); the report will not say why"),
    "XFAIL_NOT_STRICT": (ERROR, "xfail with no strict= (or strict=False) while the pytest config does not enable strict_xfail"),
    "PATCH_NO_SPEC": (ERROR, "mock.patch / patch.object without autospec/spec/new; the double drifts from the API"),
    "UNITTEST_CLASS": (ERROR, "unittest.TestCase subclass; plain functions get fixtures and assert rewriting"),
    "MUTABLE_MODULE_STATE": (ERROR, "module-level mutable is mutated by tests; state leaks between tests"),
    "SHADOWED_TEST_NAME": (ERROR, "two tests with the same name in one scope; the first never runs"),
    "YIELD_FIXTURE": (ERROR, "pytest.yield_fixture is deprecated; pytest 9.2 schedules removal for pytest 10 or 11; use @pytest.fixture"),
    "SUPPRESSION_NO_REASON": (ERROR, "audit: ignore[...] pragma without a reason after the bracket"),
    "SUPPRESSION_INERT": (WARNING, (
        "audit: ignore[...] pragma that suppresses nothing: unknown, misspelt, lower-case or empty code, ALL, "
        "or no matching finding in its line, scope or file (stale)"
    )),
    "SYNTAX": (ERROR, "file does not parse"),
}

REAL_MOCK_ASSERTIONS = frozenset({
    "assert_called", "assert_called_once", "assert_called_with", "assert_called_once_with",
    "assert_any_call", "assert_has_calls", "assert_not_called", "assert_awaited",
    "assert_awaited_once", "assert_awaited_with", "assert_awaited_once_with",
    "assert_any_await", "assert_has_awaits", "assert_not_awaited",
})
FAKE_MOCK_ATTRS = frozenset({"called_once", "called_once_with", "called_with", "not_called", "has_calls"})
MOCK_TYPO_CUTOFF = 0.88  # assert_callable (0.86) is a real helper name; assert_called_onec (0.94) is a typo
# Qualified names (see import_aliases); with-statement forms that assert, and how many positional
# arguments make the call form `pytest.raises(Exc, func, *args)` actually run something.
CONTEXT_ASSERTS = {"pytest.raises": 2, "pytest.warns": 2, "pytest.deprecated_call": 1, "pytest.RaisesGroup": None, "pytest.RaisesExc": None}
NETWORK_CALLS = frozenset({
    *(f"requests.{n}" for n in ("get", "post", "put", "patch", "delete", "head", "options", "request", "Session", "session")),
    *(f"httpx.{n}" for n in ("get", "post", "put", "patch", "delete", "head", "options", "request", "stream", "Client", "AsyncClient")),
    "urllib.request.urlopen", "urllib.request.urlretrieve",
    "urllib3.PoolManager", "urllib3.request", "urllib3.HTTPConnectionPool", "urllib3.HTTPSConnectionPool", "urllib3.ProxyManager",
    "socket.socket", "socket.create_connection", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
    "socket.gethostbyaddr", "socket.getfqdn",
    "aiohttp.ClientSession", "aiohttp.request",
    "http.client.HTTPConnection", "http.client.HTTPSConnection",
})
SWALLOWING = frozenset({"AssertionError", "Exception", "BaseException"})
BROAD_EXCEPTIONS = frozenset({"Exception", "BaseException"})
VACUOUS_MATCH = frozenset({"", ".*", "^.*$", ".*?", "(?s).*"})
NAN_NAMES = frozenset({"math.nan", "cmath.nan", "numpy.nan", "numpy.NaN", "torch.nan", "jax.numpy.nan"})
NAN_CONSTRUCTORS = frozenset({"float", "numpy.float64", "numpy.float32", "decimal.Decimal"})
INTEGRATION_MARKERS = frozenset({"integration", "e2e"})
SEED_FUNCTIONS = frozenset({
    "random.seed", "numpy.random.seed", "tensorflow.random.set_seed", "torch.cuda.manual_seed", "torch.cuda.manual_seed_all",
})
SEED_TAILS = frozenset({"manual_seed", "manual_seed_all", "seed_everything", "set_seed"})
SEEDED_CONSTRUCTORS = frozenset({"random.Random", "numpy.random.default_rng", "numpy.random.RandomState"})
PATCH_NAMES = frozenset({
    "unittest.mock.patch", "unittest.mock.patch.object", "mock.patch", "mock.patch.object",
    "mocker.patch", "mocker.patch.object", "patch", "patch.object",
})
PATCH_DICT_NAMES = frozenset({"unittest.mock.patch.dict", "mock.patch.dict", "mocker.patch.dict", "patch.dict"})
ENV_CALLS = frozenset({"os.environ.update", "os.environ.setdefault", "os.environ.pop", "os.environ.popitem",
                       "os.environ.clear", "os.putenv", "os.unsetenv"})
SYS_PATH_CALLS = frozenset({"sys.path.insert", "sys.path.append", "sys.path.extend", "sys.path.remove",
                            "sys.path.pop", "sys.path.clear"})
SLEEP_NAMES = frozenset({"time.sleep", "asyncio.sleep", "anyio.sleep", "trio.sleep"})
UNITTEST_BASES = frozenset({"unittest.TestCase", "unittest.IsolatedAsyncioTestCase", "unittest.case.TestCase"})
MUTATORS = frozenset({"append", "extend", "add", "update", "insert", "pop", "clear", "setdefault", "remove",
                      "discard", "sort", "reverse", "popitem"})
SKIP_DIRS = frozenset({".venv", "venv", ".tox", ".nox", "node_modules", ".git", "build", "dist", "mutants", "site-packages"})
# Mirrors CONFIG_LOADERS_BY_NAME in pytest 9.2.0 _pytest/config/findpaths.py (precedence order).
CONFIG_NAMES = ("pytest.toml", ".pytest.toml", "pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg")
DEFAULT_ALIASES = {"np": "numpy"}  # so unimported `np.random.rand()` in a snippet still resolves
# Any comment that tries to be a pragma, bracket or not, so a typo is seen rather than silently inert.
PRAGMA = re.compile(r"#\s*audit:\s*(ignore-file|ignore)\b(?:\[([^\]]*)\])?\s*(.*)$")
UNSUPPRESSIBLE = frozenset({"SUPPRESSION_NO_REASON", "SUPPRESSION_INERT", "SYNTAX"})
_COMPARE_OPS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
                ast.Gt: operator.gt, ast.GtE: operator.ge}


class AuditError(Exception):
    """The audit could not check what it was asked to check (exit status 2)."""


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    code: str
    detail: str
    severity: str

    def render(self) -> str:
        return f"{self.file}:{self.line}: {self.code}: {self.detail}"


@dataclass(frozen=True)
class Report:
    findings: list[Finding]
    files_checked: int
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assert_helpers: tuple[str, ...] = ()


# --------------------------------------------------------------------------- names
def dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def import_aliases(tree: ast.AST) -> dict[str, str]:
    """Map each name bound by an import to the dotted name it stands for.

    `import pytest as pt` -> pt: pytest; `from math import nan` -> nan: math.nan;
    `import os.path` -> os: os. One helper for every rule that keys on a module name.
    """
    aliases = dict(DEFAULT_ALIASES)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    aliases[item.asname] = item.name
                else:
                    root = item.name.split(".", 1)[0]
                    aliases[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for item in node.names:
                if item.name != "*":
                    aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def qualified(node: ast.AST, aliases: dict[str, str]) -> str:
    """Dotted name of a node (of the callee, for a call) with its import alias resolved."""
    name = dotted(node.func) if isinstance(node, ast.Call) else dotted(node)
    root, dot, rest = name.partition(".")
    target = aliases.get(root)
    return f"{target}{dot}{rest}" if target else name


def has_kw(call: ast.Call, *names: str) -> bool:
    return any(kw.arg in names for kw in call.keywords)


def kw_value(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def numeric_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)


def has_float(node: ast.AST, aliases: dict[str, str]) -> bool:
    """True if the expression is (or builds) a finite float, so == on it is an exact float compare."""
    if isinstance(node, ast.UnaryOp):
        return has_float(node.operand, aliases)
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float) and math.isfinite(node.value)
    if isinstance(node, ast.BinOp):
        true_division = isinstance(node.op, ast.Div) and numeric_constant(node.left) and numeric_constant(node.right)
        return true_division or has_float(node.left, aliases) or has_float(node.right, aliases)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(has_float(e, aliases) for e in node.elts)
    if isinstance(node, ast.Dict):
        return any(has_float(v, aliases) for v in node.values)
    if isinstance(node, ast.Call) and qualified(node, aliases) == "float" and len(node.args) == 1:
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            try:
                return math.isfinite(float(arg.value))
            except ValueError:
                return False
    return False


def is_nan_expr(node: ast.AST, aliases: dict[str, str]) -> bool:
    if qualified(node, aliases) in NAN_NAMES:
        return True
    if isinstance(node, ast.Call) and qualified(node, aliases) in NAN_CONSTRUCTORS and node.args:
        arg = node.args[0]
        return isinstance(arg, ast.Constant) and str(arg.value).strip().lower().lstrip("+-") == "nan"
    return False


def const_truth(node: ast.AST) -> bool | None:
    """Truth value of an expression that does not depend on anything at run time, else None."""
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.Lambda):
        return True
    if isinstance(node, ast.JoinedStr):
        parts = [v for v in node.values]
        if all(isinstance(v, ast.Constant) for v in parts):
            return any(bool(v.value) for v in parts if isinstance(v, ast.Constant))
        return None
    if isinstance(node, ast.UnaryOp):
        inner = const_truth(node.operand)
        if isinstance(node.op, ast.Not):
            return None if inner is None else not inner
        if isinstance(node.op, (ast.USub, ast.UAdd)) and numeric_constant(node.operand):
            return inner
        return None
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        if any(isinstance(e, ast.Starred) for e in node.elts):
            return None
        return bool(node.elts)
    if isinstance(node, ast.Dict):
        return None if any(k is None for k in node.keys) else bool(node.keys)
    if isinstance(node, ast.BoolOp):
        values = [const_truth(v) for v in node.values]
        if isinstance(node.op, ast.Or):
            return True if any(v is True for v in values) else (False if all(v is False for v in values) else None)
        return False if any(v is False for v in values) else (True if all(v is True for v in values) else None)
    if isinstance(node, ast.Compare):
        return _const_compare(node)
    return None


def _const_compare(node: ast.Compare) -> bool | None:
    operands = [node.left, *node.comparators]
    if (
        len(node.ops) == 1
        and isinstance(node.ops[0], ast.GtE)
        and isinstance(operands[0], ast.Call)
        and dotted(operands[0].func) == "len"
        and isinstance(operands[1], ast.Constant)
        and operands[1].value == 0
    ):
        return True  # len(x) >= 0
    if not all(isinstance(o, ast.Constant) or (isinstance(o, ast.UnaryOp) and numeric_constant(o.operand)) for o in operands):
        return None
    try:
        values = [ast.literal_eval(o) for o in operands]
        result = True
        for i, op in enumerate(node.ops):
            func = _COMPARE_OPS.get(type(op))
            if func is None:
                return None  # `is` on constants is implementation-defined
            result = result and bool(func(values[i], values[i + 1]))
        return result
    except (TypeError, ValueError):
        return None


def is_seed_call(call: ast.Call, aliases: dict[str, str]) -> bool:
    """A call that seeds a global generator (it needs a seed argument: random.seed() is entropy)."""
    if not (call.args or call.keywords):
        return False
    name = qualified(call, aliases)
    return name in SEED_FUNCTIONS or name.rsplit(".", 1)[-1] in SEED_TAILS


def walk_without_defs(node: ast.AST) -> Iterable[ast.AST]:
    """ast.walk that does not descend into function or class bodies."""
    todo = [node]
    while todo:
        current = todo.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                todo.append(child)


def source_seeds_globally(tree: ast.Module, aliases: dict[str, str]) -> bool:
    """Does importing this module seed the generators for every test that runs after it?

    True for a seed call at module level, or inside an autouse fixture or a pytest_* hook.
    A seed function that is never called, or a commented-out call, does not count.
    """
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            hook = stmt.name.startswith("pytest_")
            autouse = any(_autouse_fixture(d, aliases) for d in stmt.decorator_list)
            scope: Iterable[ast.AST] = ast.walk(stmt) if (hook or autouse) else ()
        elif isinstance(stmt, ast.ClassDef):
            continue
        else:
            scope = walk_without_defs(stmt)
        if any(isinstance(n, ast.Call) and is_seed_call(n, aliases) for n in scope):
            return True
    return False


def _autouse_fixture(dec: ast.expr, aliases: dict[str, str]) -> bool:
    if not isinstance(dec, ast.Call) or qualified(dec, aliases) not in {"pytest.fixture", "pytest.yield_fixture"}:
        return False
    value = kw_value(dec, "autouse")
    return isinstance(value, ast.Constant) and value.value is True


def pytestmark_values(stmt: ast.stmt) -> list[ast.expr]:
    """The marker expressions assigned to `pytestmark` by this statement (plain or annotated)."""
    if isinstance(stmt, ast.Assign):
        targets, value = stmt.targets, stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        targets, value = [stmt.target], stmt.value
    else:
        return []
    if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets):
        return []
    return list(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else [value]


def helper_matches(call: ast.Call, aliases: dict[str, str], patterns: tuple[str, ...]) -> bool:
    """Does this call go to a project assertion helper?

    A pattern without a dot is matched (fnmatch, case-sensitive) against the final name of the
    callee: `check_x` for `check_x(r)`, `self.check_x(r)` and `helpers.check_x(r)`, also after an import
    alias is resolved. A pattern with a dot is matched against the whole dotted callee, as written and
    with the import alias resolved (`helpers.expect_*` for `helpers.expect_ok(r)` and `h.expect_ok(r)`).
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        tail = func.attr
    elif isinstance(func, ast.Name):
        tail = func.id
    else:
        return False
    resolved = qualified(call, aliases)
    tails = {tail, resolved.rsplit(".", 1)[-1]}
    wholes = {dotted(func), resolved} - {""}
    return any(
        any(fnmatch.fnmatchcase(w, p) for w in wholes) if "." in p else any(fnmatch.fnmatchcase(t, p) for t in tails)
        for p in patterns
    )


def is_test_function(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")


# --------------------------------------------------------------------------- config
def _toml_load(path: Path, warnings: list[str]) -> dict | None:
    """Parse a TOML file; None (with a loud warning) when this Python cannot."""
    if tomllib is None:
        warnings.append(
            f"{path}: not read because this Python ({sys.version_info.major}.{sys.version_info.minor}) has no tomllib "
            "(3.11+); strict xfail settings in it are ignored, so XFAIL_NOT_STRICT may be a false positive"
        )
        return None
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
        raise AuditError(f"cannot read pytest config {path}: {exc}") from exc


def _ini_section(path: Path, section: str, *, definitive: bool, warnings: list[str]) -> dict[str, str] | None:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read(path, encoding="utf-8")
    except (configparser.Error, OSError, UnicodeDecodeError) as exc:
        if definitive:
            raise AuditError(f"cannot read pytest config {path}: {exc}") from exc
        warnings.append(f"{path}: ignored, it does not parse as an INI file ({exc.__class__.__name__})")
        return None
    return dict(parser[section]) if parser.has_section(section) else None


def _load_config_file(path: Path, warnings: list[str]) -> dict | None:
    """Options of one config file, or None if pytest 9.2 would not treat it as its config."""
    name = path.name
    if name in ("pytest.toml", ".pytest.toml"):  # always the config, even when empty
        data = _toml_load(path, warnings)
        table = (data or {}).get("pytest")
        return dict(table) if isinstance(table, dict) else {}
    if name in ("pytest.ini", ".pytest.ini"):  # always the config, even without a [pytest] section
        return _ini_section(path, "pytest", definitive=True, warnings=warnings) or {}
    if name == "pyproject.toml":
        data = _toml_load(path, warnings)
        tool = ((data or {}).get("tool") or {}).get("pytest")
        if not isinstance(tool, dict):
            return None
        options = {k: v for k, v in tool.items() if k != "ini_options"}
        ini_options = tool.get("ini_options")
        if ini_options is None and not options:
            return None  # [tool.pytest] absent or empty: pytest does not treat pyproject.toml as its config
        if isinstance(ini_options, dict):
            options.update(ini_options)
        return options
    if name == "tox.ini":
        return _ini_section(path, "pytest", definitive=False, warnings=warnings)
    return _ini_section(path, "tool:pytest", definitive=False, warnings=warnings)  # setup.cfg


def _option_bool(options: dict, key: str) -> bool | None:
    if key not in options:
        return None
    value = options[key]
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None  # pytest itself rejects other values; treat as unset


def strict_xfail_enabled(options: dict) -> bool:
    """pytest 9.2 skipping.py: strict_xfail (alias xfail_strict) if set, else the `strict` option."""
    for key in ("strict_xfail", "xfail_strict", "strict"):
        value = _option_bool(options, key)
        if value is not None:
            return value
    return False


ASSERT_HELPER_PATTERN = re.compile(r"[A-Za-z0-9_.*?\[\]!-]+")
EXPERT_TABLE = "pytest-expert"
EXPERT_KEYS = ("assert_helpers",)


def validate_assert_helper(pattern: str, origin: str) -> str:
    """The pattern, if it can only match some calls; AuditError saying why otherwise."""
    if not pattern.strip():
        raise AuditError(f"{origin}: empty pattern; give a function name or a glob such as check_*")
    if not ASSERT_HELPER_PATTERN.fullmatch(pattern):
        raise AuditError(
            f"{origin}: {pattern!r} is not a function name or glob (letters, digits, _ . * ? [ ] only); "
            "give one pattern per option, not a list, a call or text with spaces"
        )
    if not any(c.isalnum() or c == "_" for c in re.sub(r"\[[^\]]*\]", "", pattern)):
        raise AuditError(
            f"{origin}: {pattern!r} has no literal name characters, so it would match every call and switch "
            "NO_ASSERT off for every test; name the helper, e.g. check_* or helpers.expect_*"
        )
    return pattern


def _expert_helpers(directory: Path, warnings: list[str]) -> tuple[str, ...] | None:
    """assert_helpers from [tool.pytest-expert] in directory/pyproject.toml; None if there is no such table."""
    path = directory / "pyproject.toml"
    if not path.is_file():
        return None
    if tomllib is None:
        try:
            has_table = f"[tool.{EXPERT_TABLE}]" in path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise AuditError(f"cannot read {path}: {exc}") from exc
        if has_table:
            warnings.append(
                f"{path}: [tool.{EXPERT_TABLE}] not read because this Python ({sys.version_info.major}."
                f"{sys.version_info.minor}) has no tomllib (3.11+); its assert_helpers are ignored, so NO_ASSERT "
                "may be a false positive for tests that check through a project helper"
            )
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc
    table = (data.get("tool") or {}).get(EXPERT_TABLE)
    if table is None:
        return None
    where = f"{path}: [tool.{EXPERT_TABLE}]"
    if not isinstance(table, dict):
        raise AuditError(f"{where} must be a table")
    unknown = sorted(set(table) - set(EXPERT_KEYS))
    if unknown:
        raise AuditError(f"{where}: unknown key(s) {', '.join(unknown)}; supported: {', '.join(EXPERT_KEYS)}")
    value = table.get("assert_helpers", [])
    if not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
        raise AuditError(f"{where}: assert_helpers must be a list of strings, e.g. [\"check_*\"]")
    return tuple(validate_assert_helper(v, f"{where} assert_helpers") for v in value)


@dataclass(frozen=True)
class Project:
    root: Path
    strict_xfail: bool
    assert_helpers: tuple[str, ...]
    warnings: tuple[str, ...]


def find_project(start: Path) -> Project:
    """Bound every upward lookup: the nearest directory with a pytest config, a .git or a
    [tool.pytest-expert] table, else `start`."""
    warnings: list[str] = []
    for directory in [start, *start.parents]:
        strict: bool | None = None
        for name in CONFIG_NAMES:
            path = directory / name
            if path.is_file():
                options = _load_config_file(path, warnings)
                if options is not None:
                    strict = strict_xfail_enabled(options)
                    break
        helpers = _expert_helpers(directory, warnings)
        if strict is not None or helpers is not None or (directory / ".git").exists():
            return Project(directory, bool(strict), helpers or (), tuple(warnings))
    return Project(start, False, (), tuple(warnings))


class SeedingIndex:
    """Which conftest.py files seed the generators; read each one once."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._cache: dict[Path, bool] = {}

    def _conftest_seeds(self, path: Path) -> bool:
        if path not in self._cache:
            try:
                tree = ast.parse(path.read_bytes())
            except (OSError, SyntaxError, ValueError):
                # An unreadable or unparsable conftest inside the audited tree is reported as a
                # problem / SYNTAX finding when it is collected; here it just cannot seed anything.
                self._cache[path] = False
            else:
                self._cache[path] = source_seeds_globally(tree, import_aliases(tree))
        return self._cache[path]

    def seeded(self, test_file: Path) -> bool:
        """A conftest.py in the test's directory or any directory above it, up to the project root."""
        for directory in [test_file.parent, *test_file.parent.parents]:
            conftest = directory / "conftest.py"
            if conftest.is_file() and self._conftest_seeds(conftest):
                return True
            if directory == self.project_root:
                break
        return False


# --------------------------------------------------------------------------- pragmas
@dataclass(frozen=True)
class Pragma:
    line: int
    file_level: bool
    bracketed: bool
    tokens: tuple[str, ...]  # the codes as written, stripped, empty ones dropped
    has_reason: bool


def read_pragmas(source: str) -> list[Pragma]:
    """Every `# audit: ignore...` comment (real comments only: not text inside a string or docstring)."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    pragmas: list[Pragma] = []
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        match = PRAGMA.search(tok.string)
        if not match:
            continue
        kind, codes_text, reason = match.groups()
        codes = tuple(c.strip() for c in (codes_text or "").split(",") if c.strip())
        pragmas.append(Pragma(tok.start[0], kind == "ignore-file", codes_text is not None, codes, bool(reason.strip())))
    return pragmas


def pragma_code_problems(pragma: Pragma) -> tuple[list[str], list[str]]:
    """(what is wrong with the pragma's codes, the codes that are valid and can suppress)."""
    if not pragma.bracketed:
        return ["no [CODE] list; write `audit: ignore[CODE] reason`"], []
    if not pragma.tokens:
        return ["empty brackets; name the code to suppress, e.g. ignore[NO_ASSERT]"], []
    problems: list[str] = []
    valid: list[str] = []
    for token in pragma.tokens:
        upper = token.upper()
        if upper == "ALL":
            problems.append("ALL is not supported; name each code to suppress")
        elif token in UNSUPPRESSIBLE:
            problems.append(f"{token} cannot be suppressed by a pragma (use --ignore on the command line)")
        elif token in CODES:
            valid.append(token)
        elif upper in CODES:
            problems.append(f"{token} must be upper case; write {upper}")
        else:
            near = difflib.get_close_matches(upper, CODES, n=1, cutoff=0.6)
            problems.append(f"unknown code {token}" + (f" (did you mean {near[0]}?)" if near else ""))
    return problems, valid


# --------------------------------------------------------------------------- reachability
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _collect_live(func: ast.AST, included: set[int]) -> tuple[list[ast.AST], set[str], list[ast.AST]]:
    """Nodes that can run in `func`, the names it loads, and the nested defs/lambdas/classes it contains.

    Statements after an unconditional return/raise are dead. A nested def is entered only if its
    id is in `included`; lambdas and classes are never entered.
    """
    nodes: list[ast.AST] = []
    loads: set[str] = set()
    nested: list[ast.AST] = []

    def visit(node: ast.AST, root: bool = False) -> None:
        if not root and isinstance(node, (*FUNCTION_NODES, ast.Lambda, ast.ClassDef)):
            nested.append(node)
            if id(node) not in included or not isinstance(node, FUNCTION_NODES):
                return
        nodes.append(node)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads.add(node.id)
        for _name, value in ast.iter_fields(node):
            for child in value if isinstance(value, list) else [value]:
                if isinstance(child, ast.AST):
                    visit(child)
                    if isinstance(child, (ast.Return, ast.Raise)):
                        break  # the rest of this statement list is unreachable

    visit(func, root=True)
    return nodes, loads, nested


def live_nodes(func: ast.AST) -> list[ast.AST]:
    """Reachable nodes of a test; nested functions count only when something refers to them."""
    included: set[int] = set()
    nodes: list[ast.AST] = []
    for _ in range(8):  # bounded: each round can only admit more nested functions
        nodes, loads, nested = _collect_live(func, included)
        new = {id(n) for n in nested if isinstance(n, FUNCTION_NODES) and n.name in loads} - included
        if not new:
            break
        included |= new
    return nodes


def direct_nested_defs(fn: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    todo = list(getattr(fn, "body", []))
    while todo:
        node = todo.pop()
        if isinstance(node, FUNCTION_NODES):
            out.append(node)
        elif not isinstance(node, (ast.ClassDef, ast.Lambda)):
            todo.extend(ast.iter_child_nodes(node))
    return out


def flatten_targets(node: ast.AST) -> list[ast.expr]:
    """Every assignment/deletion target of a statement, tuples and stars unpacked."""
    raw: list[ast.expr] = []
    if isinstance(node, (ast.Assign, ast.Delete)):
        raw = list(node.targets)
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        raw = [node.target]
    out: list[ast.expr] = []
    while raw:
        item = raw.pop()
        if isinstance(item, (ast.Tuple, ast.List)):
            raw.extend(item.elts)
        elif isinstance(item, ast.Starred):
            raw.append(item.value)
        else:
            out.append(item)
    return out


def subscript_root(node: ast.AST) -> str:
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def is_none_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


# --------------------------------------------------------------------------- auditor
class Auditor(ast.NodeVisitor):
    def __init__(self, display: str, tree: ast.Module, *, seeded: bool, strict_xfail: bool, integration_dir: bool,
                 assert_helpers: tuple[str, ...] = ()) -> None:
        self.path = display
        self.assert_helpers = assert_helpers
        self.seeded = seeded
        self.strict_xfail = strict_xfail
        self.findings: list[Finding] = []
        self.fn_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.class_stack: list[ast.ClassDef] = []
        self.scopes: list[dict[str, int]] = [{}]
        self.aliases = import_aliases(tree)
        self.unittest_classes: set[str] = set()
        self.env_restored = 0  # depth of `with patch.dict(os.environ, ...)` / decorator scopes
        self.seeds_itself: dict[int, bool] = {}
        self.module_seeds = source_seeds_globally(tree, self.aliases)
        module_marks = [value for stmt in tree.body for value in pytestmark_values(stmt)]
        self.integration_module = integration_dir or bool(self._marker_names(module_marks) & INTEGRATION_MARKERS)

    # helpers --------------------------------------------------------------
    def q(self, node: ast.AST) -> str:
        return qualified(node, self.aliases)

    def add(self, node: ast.AST, code: str, detail: str = "") -> None:
        severity, default = CODES[code]
        self.findings.append(Finding(self.path, getattr(node, "lineno", 0), code, detail or default, severity))

    def current_test(self):
        return next((f for f in reversed(self.fn_stack) if is_test_function(f)), None)

    def in_test_or_fixture(self) -> bool:
        return any(is_test_function(f) or self._is_fixture(f) for f in self.fn_stack)

    def _is_fixture(self, node: ast.AST) -> bool:
        return isinstance(node, FUNCTION_NODES) and any(self.q(d).endswith("fixture") for d in node.decorator_list)

    def _marker_names(self, exprs: Iterable[ast.expr]) -> set[str]:
        names = set()
        for expr in exprs:
            name = self.q(expr)
            if name.startswith("pytest.mark."):
                names.add(name.rsplit(".", 1)[-1])
        return names

    def _is_integration(self, test: ast.AST) -> bool:
        if self.integration_module or self._marker_names(getattr(test, "decorator_list", [])) & INTEGRATION_MARKERS:
            return True
        for cls in self.class_stack:
            marks = [*cls.decorator_list, *(v for stmt in cls.body for v in pytestmark_values(stmt))]
            if self._marker_names(marks) & INTEGRATION_MARKERS:
                return True
        return False

    def _patches_environ(self, expr: ast.expr) -> bool:
        if not (isinstance(expr, ast.Call) and self.q(expr) in PATCH_DICT_NAMES and expr.args):
            return False
        target = expr.args[0]
        return self.q(target) == "os.environ" or (isinstance(target, ast.Constant) and target.value == "os.environ")

    # module ---------------------------------------------------------------
    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            for mark in pytestmark_values(stmt):
                self._check_mark(mark)
        self._check_module_state(node)
        self.generic_visit(node)

    def _check_module_state(self, tree: ast.Module) -> None:
        tracked: dict[str, int] = {}
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                targets, value = stmt.targets, stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets, value = [stmt.target], stmt.value
            else:
                continue
            if isinstance(value, (ast.List, ast.Dict, ast.Set)):
                for target in targets:
                    if isinstance(target, ast.Name) and target.id != "pytestmark":
                        tracked[target.id] = stmt.lineno
        if not tracked:
            return
        reported: set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, FUNCTION_NODES):
                self._scan_function_state(stmt, tracked, reported, frozenset())
            elif isinstance(stmt, ast.ClassDef):
                for method in stmt.body:
                    if isinstance(method, FUNCTION_NODES):
                        self._scan_function_state(method, tracked, reported, frozenset())

    def _scan_function_state(self, fn, tracked: dict[str, int], reported: set[str], shadowed: frozenset[str]) -> None:
        """Report the first mutation of each module-level mutable inside fn, unless fn binds the name itself."""
        body_nodes = [n for stmt in fn.body for n in walk_without_defs(stmt)]
        nested = direct_nested_defs(fn)
        declared_global = {name for n in body_nodes if isinstance(n, ast.Global) for name in n.names}
        args = fn.args
        local = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg] if a}
        local |= {n.name for n in nested}
        for n in body_nodes:
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                local.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                local |= {(a.asname or a.name).split(".")[0] for a in n.names}
            elif isinstance(n, ast.ExceptHandler) and n.name:
                local.add(n.name)
        shadow = (shadowed | local) - declared_global
        for n in body_nodes:
            name = self._mutated_name(n, declared_global)
            if name in tracked and name not in shadow and name not in reported:
                reported.add(name)
                self.add(n, "MUTABLE_MODULE_STATE", f"{name!r} (defined line {tracked[name]}) is mutated; move it into a fixture")
        for inner in nested:
            self._scan_function_state(inner, tracked, reported, frozenset(shadow))

    @staticmethod
    def _mutated_name(node: ast.AST, declared_global: set[str]) -> str:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS:
            return subscript_root(node.func.value)
        for target in flatten_targets(node):
            if isinstance(target, ast.Subscript):
                return subscript_root(target)
            if isinstance(node, ast.AugAssign) and isinstance(target, ast.Name) and target.id in declared_global:
                return target.id
        return ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            name = self.q(base)
            if name in UNITTEST_BASES or name in self.unittest_classes:
                self.add(node, "UNITTEST_CLASS", f"{node.name} subclasses {name}")
                self.unittest_classes.add(node.name)
                break
        for dec in node.decorator_list:
            self._check_mark(dec)
        for stmt in node.body:
            for mark in pytestmark_values(stmt):
                self._check_mark(mark)
        if node.name.startswith("Test"):
            scope = self.scopes[-1]
            if node.name in scope:
                self.add(node, "SHADOWED_TEST_NAME", f"class {node.name} already defined at line {scope[node.name]}; that one never runs")
            scope[node.name] = node.lineno
        self.scopes.append({})
        self.class_stack.append(node)
        self.generic_visit(node)
        self.class_stack.pop()
        self.scopes.pop()

    def visit_If(self, node: ast.If) -> None:
        """Each branch starts from the same known test names: one definition per branch is not shadowing."""
        scope = self.scopes[-1]
        before = dict(scope)
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        after_body = dict(scope)
        scope.clear()
        scope.update(before)
        for stmt in node.orelse:
            self.visit(stmt)
        scope.update(after_body)

    # functions ------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node) -> None:
        for dec in node.decorator_list:
            if self.q(dec).rsplit(".", 1)[-1] == "yield_fixture":
                self.add(dec, "YIELD_FIXTURE", f"{node.name} uses yield_fixture; @pytest.fixture supports yield")
        restores = sum(self._patches_environ(d) for d in node.decorator_list)
        self.env_restored += restores
        if is_test_function(node) and not self.fn_stack:
            scope = self.scopes[-1]
            if node.name in scope:
                self.add(node, "SHADOWED_TEST_NAME", f"{node.name} already defined at line {scope[node.name]}; that one never runs")
            scope[node.name] = node.lineno
            for dec in node.decorator_list:
                self._check_mark(dec)
            self._check_assertions(node)
            self._check_branching(node)
            self.seeds_itself[id(node)] = any(isinstance(n, ast.Call) and is_seed_call(n, self.aliases) for n in ast.walk(node))
        self.fn_stack.append(node)
        self.generic_visit(node)
        self.fn_stack.pop()
        self.env_restored -= restores

    # marks ----------------------------------------------------------------
    def _check_mark(self, mark: ast.expr) -> None:
        name = self.q(mark)
        if not name.startswith("pytest.mark."):
            return
        short = name.rsplit(".", 1)[-1]
        call = mark if isinstance(mark, ast.Call) else None
        if short in {"skip", "skipif", "xfail"} and not self._has_reason(short, call):
            self.add(mark, "SKIP_NO_REASON", f"@{name} has no reason")
        if short == "xfail" and self._xfail_not_strict(call):
            self.add(mark, "XFAIL_NOT_STRICT")

    @staticmethod
    def _has_reason(short: str, call: ast.Call | None) -> bool:
        if call is None:
            return False
        if any(kw.arg is None for kw in call.keywords):
            return True  # **kwargs may carry it
        value = kw_value(call, "reason")
        if value is None and short == "skip" and call.args:
            value = call.args[0]
        if value is None:
            return False
        if isinstance(value, ast.Constant):
            return isinstance(value.value, str) and bool(value.value.strip())
        return True

    def _xfail_not_strict(self, call: ast.Call | None) -> bool:
        value = kw_value(call, "strict") if call is not None else None
        if value is not None and not is_none_constant(value):
            return isinstance(value, ast.Constant) and not value.value  # strict=False opts out explicitly
        if call is not None and any(kw.arg is None for kw in call.keywords):
            return False  # **kwargs may carry strict=
        return not self.strict_xfail

    # assertions -----------------------------------------------------------
    @staticmethod
    def _is_mock_typo(tail: str) -> bool:
        return (
            tail.startswith("assert_") and tail not in REAL_MOCK_ASSERTIONS
            and bool(difflib.get_close_matches(tail, REAL_MOCK_ASSERTIONS, n=1, cutoff=MOCK_TYPO_CUTOFF))
        )

    def _check_assertions(self, node) -> None:
        live = live_nodes(node)
        asserts = [n for n in live if isinstance(n, ast.Assert)]
        with_calls = {id(i.context_expr) for n in live if isinstance(n, (ast.With, ast.AsyncWith)) for i in n.items}
        other_assertion = False
        for sub in live:
            if not isinstance(sub, ast.Call):
                continue
            name = self.q(sub)
            tail = name.rsplit(".", 1)[-1]
            if name in CONTEXT_ASSERTS:
                positional_to_run = CONTEXT_ASSERTS[name]
                if id(sub) in with_calls or (positional_to_run is not None and len(sub.args) >= positional_to_run):
                    other_assertion = True
            elif id(sub) in with_calls and any(word in tail.lower() for word in ("raises", "warns")):
                other_assertion = True  # a project helper such as `with expect_raises(...)`
            elif name == "pytest.fail" or tail in REAL_MOCK_ASSERTIONS or (tail.startswith("assert") and not self._is_mock_typo(tail)):
                other_assertion = True
        if not asserts and not other_assertion:
            # Helpers relax NO_ASSERT only: WEAK_ASSERT below still looks at the real asserts alone.
            if not any(isinstance(n, ast.Call) and helper_matches(n, self.aliases, self.assert_helpers) for n in live):
                self.add(node, "NO_ASSERT", f"{node.name} asserts nothing that can fail")
        elif asserts and not other_assertion and all(self._is_weak(a.test) for a in asserts):
            self.add(node, "WEAK_ASSERT", f"{node.name}: every assertion only checks existence or type")

    def _is_weak(self, test: ast.AST) -> bool:
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = test.operand  # `not x is None`
            return (
                isinstance(inner, ast.Compare) and len(inner.ops) == 1 and isinstance(inner.ops[0], ast.Is)
                and is_none_constant(inner.left) != is_none_constant(inner.comparators[0])
            )
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            return all(self._is_weak(v) for v in test.values)
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op, left, right = test.ops[0], test.left, test.comparators[0]
            if isinstance(op, (ast.IsNot, ast.NotEq)) and is_none_constant(left) != is_none_constant(right):
                return True
            if isinstance(left, ast.Call) and dotted(left.func) == "len" and isinstance(right, ast.Constant):
                return (isinstance(op, (ast.Gt, ast.NotEq)) and right.value == 0) or (isinstance(op, ast.GtE) and right.value == 1)
            if isinstance(op, ast.Lt) and isinstance(left, ast.Constant) and left.value == 0:
                return isinstance(right, ast.Call) and dotted(right.func) == "len"  # 0 < len(x)
        return isinstance(test, ast.Call) and self.q(test) in {"isinstance", "hasattr", "callable", "len", "bool"}

    def _check_branching(self, node) -> None:
        for stmt in node.body:
            for inner in ast.walk(stmt):
                if isinstance(inner, (ast.If, ast.While, ast.Match)):
                    self.add(inner, "BRANCHING", "if/while/match in a test body; parametrize the cases")
                    return
                if isinstance(inner, (ast.For, ast.AsyncFor)) and not isinstance(inner.iter, ast.Call):
                    self.add(inner, "BRANCHING", "for-loop over cases; parametrize the cases")
                    return

    # statements -----------------------------------------------------------
    def _bare_comparison(self, value: ast.AST) -> bool:
        if isinstance(value, ast.Compare):
            return True
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
            return self._bare_comparison(value.operand)
        if isinstance(value, ast.BoolOp):
            return all(self._bare_comparison(v) for v in value.values)
        return False

    def visit_Expr(self, node: ast.Expr) -> None:
        if self.current_test() is not None:
            if self._bare_comparison(node.value):
                self.add(node, "BARE_COMPARISON", f"`{ast.unparse(node.value)[:60]}` is not asserted")
            elif isinstance(node.value, ast.Attribute):
                self._check_mock_attribute(node.value)
        self.generic_visit(node)

    def _check_mock_attribute(self, node: ast.Attribute) -> None:
        if node.attr in FAKE_MOCK_ATTRS:
            self.add(node, "FAKE_MOCK_ASSERT", f".{node.attr} is not a Mock assertion: it is truthy on Mock(unsafe=True) and, "
                     "before Python 3.12, on any Mock; on 3.12+ a default Mock raises AttributeError")
        elif node.attr in REAL_MOCK_ASSERTIONS:
            self.add(node, "FAKE_MOCK_ASSERT", f".{node.attr} is referenced but never called (missing parentheses); it asserts nothing")

    def visit_Try(self, node: ast.Try | ast.TryStar) -> None:
        if self.current_test() is not None:
            for handler in node.handlers:
                caught = self._caught_names(handler.type)
                if caught is None or caught & SWALLOWING:
                    reraises = any(
                        isinstance(s, ast.Raise) or (isinstance(s, ast.Call) and self.q(s) == "pytest.fail")
                        for s in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
                    )
                    if not reraises:
                        label = "bare except" if caught is None else "except " + "/".join(sorted(caught & SWALLOWING))
                        self.add(handler, "SWALLOWED_FAILURE", f"{label} without re-raise hides failures")
        self.generic_visit(node)

    visit_TryStar = visit_Try

    @staticmethod
    def _caught_names(type_node) -> set[str] | None:
        if type_node is None:
            return None
        nodes = type_node.elts if isinstance(type_node, ast.Tuple) else [type_node]
        return {dotted(n).rsplit(".", 1)[-1] for n in nodes}

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        restores = sum(self._patches_environ(i.context_expr) for i in node.items)
        if self.current_test() is not None:
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call) and self.q(call) == "contextlib.suppress":
                    swallowed = {dotted(a).rsplit(".", 1)[-1] for a in call.args} & SWALLOWING
                    if swallowed:
                        self.add(node, "SWALLOWED_FAILURE", f"contextlib.suppress({', '.join(sorted(swallowed))}) hides failures")
        self.env_restored += restores
        self.generic_visit(node)
        self.env_restored -= restores

    visit_AsyncWith = visit_With

    def visit_Assert(self, node: ast.Assert) -> None:
        test = node.test
        if const_truth(test) is True:
            if isinstance(test, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
                self.add(node, "ALWAYS_TRUE", "asserting a non-empty literal container always passes")
            else:
                self.add(node, "ALWAYS_TRUE", f"assert {ast.unparse(test)[:40]} always passes")
        stack = [test]
        while stack:
            part = stack.pop()
            if isinstance(part, ast.BoolOp):
                stack.extend(part.values)
            elif isinstance(part, ast.UnaryOp) and isinstance(part.op, ast.Not):
                stack.append(part.operand)
            elif isinstance(part, ast.Attribute):
                self._check_mock_attribute(part)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if self.current_test() is not None:
            operands = [node.left, *node.comparators]
            symbols = {ast.Eq: "==", ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not in"}
            for i, op in enumerate(node.ops):
                left, right = operands[i], operands[i + 1]
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    if is_nan_expr(left, self.aliases) or is_nan_expr(right, self.aliases):
                        self.add(node, "NAN_COMPARE")
                        continue
                    operand = left if has_float(left, self.aliases) else right if has_float(right, self.aliases) else None
                elif isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.List, ast.Tuple, ast.Set)):
                    operand = right if has_float(right, self.aliases) else None
                else:
                    continue
                if operand is not None:
                    self.add(node, "FLOAT_EQ", f"float {ast.unparse(operand)[:40]} compared with {symbols[type(op)]}; use pytest.approx")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self.q(node)
        tail = name.rsplit(".", 1)[-1]
        root = name.split(".", 1)[0]
        test = self.current_test()

        if self._is_mock_typo(tail):
            self.add(node, "FAKE_MOCK_ASSERT", f"{name}() is not a real Mock assertion (misspelt?); a default Mock raises "
                     "AttributeError, Mock(unsafe=True) silently accepts it")
        if tail in FAKE_MOCK_ATTRS and isinstance(node.func, ast.Attribute):
            self.add(node, "FAKE_MOCK_ASSERT", f"{name}() is not a Mock assertion: it returns a truthy Mock on Mock(unsafe=True) and, "
                     "before Python 3.12, on any Mock; on 3.12+ a default Mock raises AttributeError")

        if name in {"pytest.raises", "pytest.RaisesExc"}:
            self._check_broad_raises(node, name)

        if name in SLEEP_NAMES and self.in_test_or_fixture():
            zero = node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0
            if not zero:
                self.add(node, "SLEEP", f"{name}() in a test or fixture")

        if test is not None and name in NETWORK_CALLS and not self._is_integration(test):
            self.add(node, "NETWORK", f"{name}() opens a network connection; stub it or mark the test integration")

        if test is not None and not (self.seeded or self.module_seeds or self.seeds_itself.get(id(test), False)):
            seeded_ctor = name in SEEDED_CONSTRUCTORS and (node.args or node.keywords)
            if not seeded_ctor and not is_seed_call(node, self.aliases) and (root == "random" or name.startswith("numpy.random.")):
                self.add(node, "UNSEEDED_RANDOM", f"{name}() with no seed")

        if test is not None and name == "print":
            self.add(node, "PRINT")

        if name in PATCH_NAMES and not self._patch_has_double(node, name):
            self.add(node, "PATCH_NO_SPEC", f"{name}() without autospec=True")

        if name in SYS_PATH_CALLS:
            self.add(node, "SYS_PATH", f"{name}()")

        if name in ENV_CALLS and not self.env_restored:
            self.add(node, "ENV_MUTATION", f"{name}(); use monkeypatch")
        self.generic_visit(node)

    def _check_broad_raises(self, node: ast.Call, name: str) -> None:
        exc = node.args[0] if node.args else kw_value(node, "expected_exception")
        if exc is None or has_kw(node, "check"):
            return
        elements = exc.elts if isinstance(exc, ast.Tuple) else [exc]
        if not {self.q(e).removeprefix("builtins.") for e in elements} & BROAD_EXCEPTIONS:
            return
        match = kw_value(node, "match")
        if match is None:
            self.add(node, "BROAD_RAISES", f"{name}({ast.unparse(exc)}) with no match=")
        elif isinstance(match, ast.Constant) and (match.value is None or match.value in VACUOUS_MATCH):
            self.add(node, "BROAD_RAISES", f"{name}({ast.unparse(exc)}) with match={ast.unparse(match)}, which matches any message")

    @staticmethod
    def _patch_has_double(node: ast.Call, name: str) -> bool:
        if any(kw.arg is None for kw in node.keywords):
            return True  # **kwargs may carry autospec/new
        for kw in node.keywords:
            if kw.arg in {"new", "new_callable"}:
                return True
            if kw.arg in {"autospec", "spec", "spec_set"} and not (isinstance(kw.value, ast.Constant) and not kw.value.value):
                return True
        needed = 3 if name.endswith(".object") else 2  # positional `new`
        return len(node.args) >= needed

    # assignment-like statements ---------------------------------------------
    def _check_targets(self, node: ast.AST) -> None:
        for target in flatten_targets(node):
            if isinstance(target, ast.Subscript):
                base = self.q(target.value)
            elif isinstance(target, ast.Attribute):
                base = self.q(target)  # os.environ |= ..., sys.path += ..., sys.path = ...
            else:
                continue
            if base == "os.environ" and not self.env_restored:
                verb = "del " if isinstance(node, ast.Delete) else ""
                self.add(node, "ENV_MUTATION", f"{verb}os.environ mutated directly; use monkeypatch.setenv/delenv")
            elif base == "sys.path":
                self.add(node, "SYS_PATH", "sys.path mutated directly")

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_targets(node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_targets(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_targets(node)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._check_targets(node)
        self.generic_visit(node)


# --------------------------------------------------------------------------- driver
def is_test_file_name(name: str) -> bool:
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def collect(root: Path) -> tuple[list[Path], list[str]]:
    """Test files below `root` (a resolved path), and the directories that could not be read.

    Directory names in SKIP_DIRS are skipped only below the root, never in the root's own path.
    """
    if root.is_file():
        return [root], []
    problems: list[str] = []

    def on_error(error: OSError) -> None:
        problems.append(f"cannot read directory {error.filename}: {error.strerror}")

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        files.extend(Path(dirpath) / name for name in sorted(filenames) if is_test_file_name(name))
    return files, problems


def function_ranges(tree: ast.Module) -> list[tuple[int, int, int]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            out.append((first, node.lineno, node.end_lineno or node.lineno))
    return out


def audit_file(path: Path, display: str, *, seeded: bool, strict_xfail: bool, integration_dir: bool,
               assert_helpers: tuple[str, ...] = ()) -> list[Finding]:
    """Audit one test file. Raises OSError if it cannot be read."""
    data = path.read_bytes()
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        source = data.decode(encoding, errors="replace")  # utf-8-sig drops a byte-order mark
        tree = ast.parse(source, filename=display)
    except SyntaxError as exc:
        return [Finding(display, exc.lineno or 0, "SYNTAX", str(exc.msg), ERROR)]
    except ValueError as exc:  # e.g. a null byte in the source
        return [Finding(display, 0, "SYNTAX", str(exc), ERROR)]
    auditor = Auditor(display, tree, seeded=seeded, strict_xfail=strict_xfail, integration_dir=integration_dir,
                      assert_helpers=assert_helpers)
    auditor.visit(tree)
    pragmas = read_pragmas(source)
    line_codes: dict[int, set[str]] = {}
    file_codes: set[str] = set()
    checked: dict[int, tuple[list[str], list[str]]] = {}
    for pragma in pragmas:
        checked[pragma.line] = problems, valid = pragma_code_problems(pragma)
        if pragma.has_reason:  # a pragma without a reason is reported and suppresses nothing
            (file_codes if pragma.file_level else line_codes.setdefault(pragma.line, set())).update(valid)
    ranges = function_ranges(tree)
    used_file: set[str] = set()
    used_line: set[tuple[int, str]] = set()

    def suppressed(f: Finding) -> bool:
        covered = f.code in file_codes
        if covered:
            used_file.add(f.code)
        lines = {f.line}
        for first, def_line, end in ranges:
            if first <= f.line <= end:
                lines.update(range(first, def_line + 1))
        for line in lines:
            if f.code in line_codes.get(line, set()):
                used_line.add((line, f.code))
                covered = True
        return covered

    kept = [f for f in auditor.findings if not suppressed(f)]
    for pragma in pragmas:
        problems, valid = checked[pragma.line]
        if pragma.bracketed and not pragma.has_reason:
            kept.append(Finding(display, pragma.line, "SUPPRESSION_NO_REASON", CODES["SUPPRESSION_NO_REASON"][1], ERROR))
        elif pragma.has_reason:
            where = "in this file" if pragma.file_level else "on this line or in its scope"
            hit = used_file if pragma.file_level else {code for line, code in used_line if line == pragma.line}
            problems = [*problems, *(f"no {code} finding {where}; the pragma is stale" for code in valid if code not in hit)]
        if problems:
            kept.append(Finding(display, pragma.line, "SUPPRESSION_INERT",
                                "audit: ignore pragma suppresses nothing: " + "; ".join(problems), WARNING))
    return kept


def audit_report(root: Path, ignore: frozenset[str] = frozenset(), assert_helpers: tuple[str, ...] = ()) -> Report:
    """Audit `root` (a test file or a directory). Raises AuditError when it cannot be audited at all.

    `assert_helpers` are extra patterns (see helper_matches) added to those in the project's
    [tool.pytest-expert]; the Report carries the patterns that were in effect.
    """
    for pattern in assert_helpers:
        validate_assert_helper(pattern, "--assert-helper")
    if "SYNTAX" in ignore:
        raise AuditError("SYNTAX cannot be ignored: a file that does not parse was not audited")
    if not root.exists():
        raise AuditError(f"no such path: {root}")
    resolved = root.resolve()
    if not (resolved.is_file() or resolved.is_dir()):
        raise AuditError(f"not a file or directory: {root}")
    base = resolved if resolved.is_dir() else resolved.parent
    project = find_project(base)
    helpers = tuple(dict.fromkeys((*project.assert_helpers, *assert_helpers)))
    seeding = SeedingIndex(project.root)
    files, problems = collect(resolved)
    if not files and not problems:
        raise AuditError(
            f"no test files found under {root} (looked for test_*.py, *_test.py and conftest.py, "
            f"skipping {', '.join(sorted(SKIP_DIRS))} below it)"
        )
    findings: list[Finding] = []
    checked = 0
    for path in files:
        display = str(root if resolved.is_file() else root / path.relative_to(resolved))
        directories = {p.lower() for p in path.parent.relative_to(project.root).parts}
        try:
            found = audit_file(path, display, seeded=seeding.seeded(path), strict_xfail=project.strict_xfail,
                               integration_dir=bool(directories & INTEGRATION_MARKERS), assert_helpers=helpers)
        except OSError as exc:
            problems.append(f"cannot read {display}: {exc.strerror or exc}")
            continue
        checked += 1
        findings += [f for f in found if f.code not in ignore]
    ordered = sorted(set(findings), key=lambda f: (f.file, f.line, f.code, f.detail))
    return Report(ordered, checked, problems, list(project.warnings), helpers)


def audit(root: Path, ignore: frozenset[str] = frozenset(), assert_helpers: tuple[str, ...] = ()) -> list[Finding]:
    """Findings for `root`. Raises AuditError if nothing could be checked or any file could not be read."""
    report = audit_report(root, ignore, assert_helpers)
    if report.problems:
        raise AuditError("; ".join(report.problems))
    return report.findings


def to_sarif(findings: list[Finding], assert_helpers: tuple[str, ...] = ()) -> dict:
    rules = [{"id": c, "shortDescription": {"text": d}, "defaultConfiguration": {"level": s}} for c, (s, d) in sorted(CODES.items())]
    results = [{
        "ruleId": f.code, "level": f.severity, "message": {"text": f.detail},
        "locations": [{"physicalLocation": {"artifactLocation": {"uri": quote(Path(f.file).as_posix(), safe="/:")},
                                            "region": {"startLine": max(f.line, 1)}}}],
    } for f in findings]
    return {"version": "2.1.0", "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [{"tool": {"driver": {"name": "pytest-expert-audit", "rules": rules}}, "results": results,
                      "properties": {"assertHelpers": list(assert_helpers)}}]}


def github_escape(text: str, *, prop: bool) -> str:
    """Escape for a GitHub workflow command; property values also escape ':' and ','."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if prop else text


def parse_ignore(values: list[str]) -> frozenset[str]:
    codes = frozenset(c.strip().upper() for value in values for c in value.split(",") if c.strip())
    unknown = sorted(codes - CODES.keys())
    if unknown:
        raise AuditError(f"unknown code(s) in --ignore: {', '.join(unknown)}; valid codes: {', '.join(sorted(CODES))}")
    return codes


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except AuditError as exc:
        print(f"audit_tests: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001  # exit 2, not Python's 1, so a crash is never mistaken for "findings"
        print(f"audit_tests: internal error, nothing was verified: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 2


def _main(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, nargs="?", help="test directory or single test file")
    parser.add_argument("--ignore", action="append", default=[],
                        help="comma-separated codes to suppress everywhere; repeatable; SYNTAX cannot be ignored")
    parser.add_argument("--assert-helper", action="append", default=[], metavar="PATTERN",
                        help="treat a call to a function matching PATTERN as an assertion for NO_ASSERT only; repeatable; "
                             "a pattern without a dot matches the final name of the callee (check_*), one with a dot the "
                             "whole dotted callee (helpers.expect_*); fnmatch, case-sensitive; a pattern that would match "
                             "every call is rejected; adds to [tool.pytest-expert] assert_helpers in pyproject.toml")
    parser.add_argument("--format", choices=("text", "json", "github", "sarif"), default="text")
    parser.add_argument("--warnings-as-errors", action="store_true", help="exit 1 on warning-severity findings too")
    parser.add_argument("--list-codes", action="store_true", help="print every code and exit")
    args = parser.parse_args(argv)

    if args.list_codes:
        for code, (severity, why) in sorted(CODES.items()):
            print(f"{code:23} {severity:8} {why}")
        return 0
    if args.path is None:
        raise AuditError("a test directory or file is required (usage: audit_tests.py PATH)")

    report = audit_report(args.path, parse_ignore(args.ignore), tuple(args.assert_helper))
    findings = report.findings
    for warning in report.warnings:
        print(f"audit_tests: warning: {warning}", file=sys.stderr)

    helpers = report.assert_helpers
    relaxed = f"NO_ASSERT relaxed by assert helpers: {', '.join(helpers)}"
    if args.format == "json":
        rows = [asdict(f) for f in findings]
        # A list stays a list for everyone who parses it; the object appears only when a rule was relaxed.
        print(json.dumps({"findings": rows, "assert_helpers": list(helpers)} if helpers else rows, indent=2))
    elif args.format == "sarif":
        print(json.dumps(to_sarif(findings, helpers), indent=2))
    elif args.format == "github":
        if helpers:
            print(f"::notice title=assert-helpers::{github_escape(relaxed, prop=False)}")
        for f in findings:
            level = "error" if f.severity == ERROR else "warning"
            print(f"::{level} file={github_escape(f.file, prop=True)},line={max(f.line, 1)},"
                  f"title={f.code}::{github_escape(f.detail, prop=False)}")
    else:
        for f in findings:
            print(f.render() + ("" if f.severity == ERROR else "  [warning]"))
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.code] = counts.get(f.code, 0) + 1
        suffix = f"; {relaxed}" if helpers else ""
        if findings:
            print(f"\n{len(findings)} finding(s): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + suffix)
        elif not report.problems:
            print(f"clean: no findings ({report.files_checked} file(s) checked){suffix}")

    for problem in report.problems:
        print(f"audit_tests: {problem}", file=sys.stderr)
    if report.problems:
        print(f"audit_tests: {len(report.problems)} problem(s) above: the audit is incomplete (exit 2)", file=sys.stderr)
        return 2
    failing = [f for f in findings if f.severity == ERROR or args.warnings_as_errors]
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
