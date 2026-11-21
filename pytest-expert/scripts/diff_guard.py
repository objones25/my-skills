#!/usr/bin/env python3
"""Report edits that turn a test suite green without fixing anything.

A coding agent told "make the tests pass" can get there by deleting the failing test,
dropping its assertions, loosening them, adding a skip, deselecting it in config, or
lowering the coverage floor. Every one of those leaves pytest green. This script compares
the test tree in the working directory (committed or not) with a git base and reports
them. It also warns on two source-side cheats: an __eq__ that returns a constant, and new
code that compares an input with a literal the tests use (special-casing).
Two more warnings cover edits with legitimate uses: a new or changed pytest_generate_tests
hook, and a skip reached through a helper or fixture a test newly uses.

Pure stdlib (ast, subprocess, tomllib, configparser). It never imports the code under
test. Git paths are read NUL-separated and file contents as bytes, so non-ASCII names,
binary snapshots and non-UTF-8 test files (PEP 263 coding cookie) are handled; a file
that cannot be decoded or parsed is reported as SYNTAX instead of being skipped.

Usage:
    python diff_guard.py                     compare the working tree with HEAD
    python diff_guard.py --base origin/main  compare with another ref
    python diff_guard.py --allow TEST_DELETED,EXPECTATION_CHANGED   (repeatable)
    python diff_guard.py --format json|github
    python diff_guard.py --list-codes

Exit status: 0 = checked and clean (warnings alone do not fail unless
--warnings-as-errors); 1 = an error-severity finding remains; 2 = could not check (not a
git work tree, unknown base ref, git missing, nothing to compare, unreadable or unparsable
config, bad arguments). Exit 2 is never "clean".
"""

from __future__ import annotations

import argparse
import ast
import configparser
import difflib
import io
import json
import operator
import re
import shlex
import subprocess
import sys
import tokenize
from collections import Counter
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]  # same API, optional backport
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]  # TOML config then ends the run with exit 2, never silently clean

ERROR, WARNING = "error", "warning"

CODES: dict[str, tuple[str, str]] = {
    "TEST_FILE_DELETED": (ERROR, "a test file present in the base is gone (a pure rename or move is followed, not reported)"),
    "TEST_DELETED": (ERROR, "a test function present in the base is gone from every test file"),
    "TEST_EMPTIED": (ERROR, "test body reduced to pass, ..., or an early return"),
    "ASSERTS_REDUCED": (ERROR, "test has fewer live checks (assert, pytest.raises/warns, assert*/self.assert* calls) than in the base; dead code does not count"),
    "ASSERT_WEAKENED": (ERROR, "test kept its checks but lost exact comparisons (== loosened to truthiness, is not None, in, <), or pytest.raises widened to Exception / lost match="),
    "CASES_REDUCED": (ERROR, "a parametrize list lost rows (a failing case can be dropped this way)"),
    "ASSERT_TRIVIALIZED": (ERROR, "test gained an assertion that cannot fail (assert True, assert x == x, x or True, approx with a huge tolerance)"),
    "SKIP_ADDED": (ERROR, "skip, skipif, xfail, pytest.skip(), importorskip or a unittest skip added to a test, class, module or conftest"),
    "PRAGMA_ADDED": (ERROR, "new '# audit: ignore' pragma silences the test audit"),
    "COLLECTION_NARROWED": (ERROR, "conftest gained collect_ignore entries or a collection hook, or changed the body of one"),
    "CONFIG_NARROWED": (ERROR, "pytest config gained --deselect/--ignore/-k/-m/--collect-only/--no-cov/..., or lost testpaths, python_files/classes/functions patterns"),
    "COVERAGE_LOWERED": (ERROR, "effective coverage floor (coverage fail_under or --cov-fail-under, in any config file) lowered or removed (a floor the base never had is not checked), or coverage omit gained entries"),
    "SYNTAX": (ERROR, "a .py file in the test tree (or a changed source file) cannot be decoded or parsed, so it is not compared; a collected test file or conftest then fails to collect, a helper module fails when imported"),
    "EXPECTATION_CHANGED": (WARNING, "a literal on the expected side of an assert, or a parametrize row, changed; confirm the spec changed"),
    "CONSTANT_EQ": (WARNING, "source gained __eq__/__ne__/__hash__ returning a constant: makes any == assertion pass"),
    "SPECIAL_CASE": (WARNING, "source gained a comparison with a literal that also appears in the tests: possible special-casing"),
    "SNAPSHOT_CHANGED": (WARNING, "a stored snapshot (syrupy __snapshots__/*.ambr, or any file under __snapshots__) was rewritten or deleted; review it like code"),
    "PARAM_HOOK_CHANGED": (WARNING, "a pytest_generate_tests hook was added or its body changed (it decides which parameter sets run); confirm no cases were dropped"),
    "SKIP_VIA_HELPER": (WARNING, "a test gained a call to (or a request for) a helper or fixture that can reach pytest.skip/xfail/importorskip, or such a helper gained the skip; confirm the skip condition is real"),
}

# pytest 9.2 reads these, first match per directory wins (_pytest/config/findpaths.py CONFIG_LOADERS_BY_NAME).
PYTEST_CONFIG_ORDER = ("pytest.toml", ".pytest.toml", "pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg")
# coverage 7.17 reads these, first file that carries coverage settings wins (coverage/config.py config_files_to_try).
COVERAGE_CONFIG_ORDER = (".coveragerc", ".coveragerc.toml", "setup.cfg", "tox.ini", "pyproject.toml")
CONFIG_FILES = frozenset(PYTEST_CONFIG_ORDER) | frozenset(COVERAGE_CONFIG_ORDER)

NARROWING_VALUE_FLAGS = ("--deselect", "--ignore", "--ignore-glob", "-k", "-m")
NARROWING_BARE_FLAGS = frozenset({"--collect-only", "--co", "--no-cov", "--noconftest"})
PYTEST_DEFAULTS = {
    "python_files": ["test_*.py", "*_test.py"],
    "python_classes": ["Test"],
    "python_functions": ["test"],
    "norecursedirs": [".*", "build", "dist", "CVS", "_darcs", "{arch}", "*.egg", "venv"],
}

SKIP_MARKERS = frozenset({
    "pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail",
    "unittest.skip", "unittest.skipIf", "unittest.skipUnless", "unittest.expectedFailure",
})
SKIP_FUNCS = frozenset({"pytest.skip", "pytest.xfail", "pytest.importorskip"})
SKIP_NAMES = SKIP_MARKERS | SKIP_FUNCS
SKIP_EXCEPTIONS = frozenset({"SkipTest", "Skipped"})
CHECK_CALL_TAILS = frozenset({"raises", "warns", "deprecated_call", "fail", "RaisesGroup", "RaisesExc", "group_contains"})
RAISES_TAILS = frozenset({"raises", "RaisesExc"})
EXACT_WORDS = ("equal", "close", "almost", "called_with", "called_once_with", "any_call", "has_calls")
COLLECTION_HOOKS = frozenset({"pytest_collection_modifyitems", "pytest_ignore_collect", "pytest_deselected",
                              "pytest_collect_file", "pytest_pycollect_makeitem"})
COLLECT_IGNORE_NAMES = frozenset({"collect_ignore", "collect_ignore_glob"})
PRAGMA = re.compile(r"#\s*audit:\s*ignore(-file)?\[")
BROAD_EXCEPTIONS = frozenset({"Exception", "BaseException"})
MATCH_ANYTHING = frozenset({"", ".*", ".+", "^.*$", "^.+$", "(?s).*"})
RENAME_SIMILARITY_MIN = 0.6
RENAME_PAIR_LIMIT = 400


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    code: str
    detail: str
    severity: str

    def render(self) -> str:
        return f"{self.file}:{self.line}: {self.code}: {self.detail}"


class GitError(RuntimeError):
    """git failed (not a repository, unknown ref, ...). The run ends with exit 2."""


class UnreadableError(RuntimeError):
    """Something the guard needs cannot be read or parsed. The run ends with exit 2."""


# git ---------------------------------------------------------------------------------
def run_git(root: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if proc.returncode != 0:
        raise GitError(proc.stderr.decode("utf-8", "replace").strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def git(root: Path, *args: str) -> str:
    return run_git(root, *args).decode("utf-8", "replace")


def git_paths(root: Path, *args: str) -> list[str]:
    """Paths from a `-z` listing: NUL-separated, never quoted, so non-ASCII names survive."""
    return [p.decode("utf-8", "surrogateescape") for p in run_git(root, *args).split(b"\0") if p]


def repo_root(start: Path) -> Path:
    return Path(git(start, "rev-parse", "--show-toplevel").strip())


def base_tree(root: Path, base: str) -> list[str]:
    """Blob paths in the base commit (submodules are not blobs and are left out)."""
    names = []
    for entry in run_git(root, "ls-tree", "-r", "-z", base).split(b"\0"):
        header, _, path = entry.partition(b"\t")
        if path and header.split()[1:2] == [b"blob"]:
            names.append(path.decode("utf-8", "surrogateescape"))
    return names


def base_files(root: Path, base: str, names: list[str]) -> dict[str, bytes]:
    return {n: run_git(root, "cat-file", "blob", f"{base}:{n}") for n in names if is_relevant(n)}


def current_files(root: Path) -> dict[str, bytes]:
    out = {}
    for n in git_paths(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"):
        path = root / n
        if is_relevant(n) and path.is_file():
            out[n] = path.read_bytes()
    return out


# classification ----------------------------------------------------------------------
def is_test_file(name: str) -> bool:
    p = PurePosixPath(name)
    if p.suffix != ".py":
        return False
    return p.name == "conftest.py" or p.name.startswith("test_") or p.name.endswith("_test.py") or "tests" in p.parts


def is_snapshot_file(name: str) -> bool:
    p = PurePosixPath(name)
    return "__snapshots__" in p.parts or p.suffix == ".ambr"


def is_config_file(name: str) -> bool:
    return PurePosixPath(name).name in CONFIG_FILES


def is_relevant(name: str) -> bool:
    return is_test_file(name) or is_snapshot_file(name) or is_config_file(name)


# reading Python the way Python does ---------------------------------------------------
class SourceUnreadable(ValueError):
    pass


def decode_source(data: bytes) -> str:
    """Decode with the PEP 263 coding cookie / BOM, like the interpreter (not strict UTF-8)."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        return data.decode(encoding)
    except (SyntaxError, LookupError, UnicodeDecodeError) as exc:
        raise SourceUnreadable(f"cannot decode: {exc}") from exc


def source_text(data: bytes) -> str:
    """Best-effort text for scans that tolerate damage (pragma counting, similarity)."""
    try:
        return decode_source(data)
    except SourceUnreadable:
        return data.decode("utf-8", "replace")


def parse_python(data: bytes | str) -> tuple[ast.Module | None, str]:
    try:
        text = decode_source(data) if isinstance(data, bytes) else data
        return ast.parse(text), ""
    except SourceUnreadable as exc:
        return None, str(exc)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return None, str(exc) or type(exc).__name__


def count_pragmas(data: bytes) -> int:
    text = source_text(data)
    try:
        return sum(1 for t in tokenize.generate_tokens(io.StringIO(text).readline)
                   if t.type == tokenize.COMMENT and PRAGMA.search(t.string))
    except (tokenize.TokenError, SyntaxError):
        return len(PRAGMA.findall(text))


# AST facts ---------------------------------------------------------------------------
def dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def call_name(node: ast.AST) -> str:
    return dotted(node.func if isinstance(node, ast.Call) else node)


def tail_of(node: ast.AST) -> str:
    return call_name(node).rsplit(".", 1)[-1]


def build_aliases(tree: ast.Module) -> dict[str, str]:
    """Local names for pytest/unittest objects: `import pytest as pt`, `from pytest import skip`, `m = pytest.mark`."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}".replace("unittest.case.", "unittest.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            is_call = isinstance(node.value, ast.Call)
            target = resolve(dotted(node.value.func if isinstance(node.value, ast.Call) else node.value), aliases)
            # `np = pytest.importorskip(...)` binds a module, not the skip function; only markers are aliased through a call
            if target in (SKIP_MARKERS if is_call else SKIP_NAMES | {"pytest.mark"}):
                aliases[node.targets[0].id] = target
    return aliases


def resolve(name: str, aliases: dict[str, str]) -> str:
    parts = name.split(".")
    for _ in range(4):
        target = aliases.get(parts[0])
        if target is None or target == parts[0]:
            break
        parts = target.split(".") + parts[1:]
    return ".".join(parts)


def count_skips(node: ast.AST, aliases: dict[str, str]) -> int:
    """Skip/xfail occurrences under `node`: markers (bare or called), skip calls, SkipTest raises.

    A marker written as a call is counted once (its function is not visited again), so
    adding reason= to an existing bare skip does not look like a new skip.
    """
    total = 0
    stack: list[ast.AST] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Call):
            func = cur.func
            if resolve(dotted(func), aliases) in SKIP_NAMES or (isinstance(func, ast.Attribute) and func.attr == "skipTest"):
                total += 1
            stack.extend(cur.args)
            stack.extend(k.value for k in cur.keywords)
            if isinstance(func, ast.Attribute):
                stack.append(func.value)
            elif not isinstance(func, ast.Name):
                stack.append(func)
        elif isinstance(cur, (ast.Name, ast.Attribute)):
            if isinstance(cur.ctx, ast.Load) and resolve(dotted(cur), aliases) in SKIP_NAMES:
                total += 1
            elif isinstance(cur, ast.Attribute):
                stack.append(cur.value)
        else:
            if isinstance(cur, ast.Raise) and cur.exc is not None:
                raised = cur.exc.func if isinstance(cur.exc, ast.Call) else cur.exc
                if dotted(raised).rsplit(".", 1)[-1] in SKIP_EXCEPTIONS:
                    total += 1
            stack.extend(ast.iter_child_nodes(cur))
    return total


def child_blocks(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """Statement lists nested in if/for/while/with/try/match (not in def or class)."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return []
    blocks: list[list[ast.stmt]] = []
    for attr in ("body", "orelse", "finalbody"):
        value = getattr(stmt, attr, None)
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            blocks.append(value)
    blocks.extend(h.body for h in getattr(stmt, "handlers", []))
    blocks.extend(c.body for c in getattr(stmt, "cases", []))
    return blocks


def plain_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Statements that run at import / class-creation time: nested blocks flattened, defs and classes left out."""
    out: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        blocks = child_blocks(stmt)
        if blocks:
            for block in blocks:
                out.extend(plain_statements(block))
        else:
            out.append(stmt)
    return out


def is_marker_definition(stmt: ast.stmt, aliases: dict[str, str]) -> bool:
    """`slow = pytest.mark.skip(...)`: counted where `@slow` is used, not where it is defined."""
    if not (isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) and t.id != "pytestmark" for t in stmt.targets)):
        return False
    value = stmt.value.func if isinstance(stmt.value, ast.Call) else stmt.value
    return resolve(dotted(value), aliases) in SKIP_MARKERS


def pytestmark_values(stmt: ast.stmt) -> list[ast.expr]:
    """Expressions assigned to / appended to pytestmark."""
    if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        if any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets) and stmt.value is not None:
            return list(stmt.value.elts) if isinstance(stmt.value, (ast.List, ast.Tuple)) else [stmt.value]
    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        func = stmt.value.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "pytestmark":
            args = stmt.value.args
            if func.attr == "append" and args:
                return [args[0]]
            if func.attr == "insert" and len(args) > 1:
                return [args[1]]
            if func.attr == "extend" and args and isinstance(args[0], (ast.List, ast.Tuple)):
                return list(args[0].elts)
    return []


def is_autouse_fixture(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call) and tail_of(dec) in {"fixture", "yield_fixture"} and any(
            k.arg == "autouse" and isinstance(k.value, ast.Constant) and k.value.value is True for k in dec.keywords
        ):
            return True
    return False


_SELF_OPS = (ast.Eq, ast.Is, ast.LtE, ast.GtE)
_CONST_OPS: dict[type, Any] = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not,
}


def _number(node: ast.AST) -> float | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def is_loose_approx(node: ast.AST) -> bool:
    """pytest.approx(x, abs=1e9) / rel>=1: a tolerance wide enough that the comparison cannot fail."""
    if not (isinstance(node, ast.Call) and tail_of(node) == "approx"):
        return False
    expected = _number(node.args[0]) if node.args else None
    for kw in node.keywords:
        tol = _number(kw.value)
        if tol is None:
            continue
        if kw.arg == "rel" and tol >= 1:
            return True
        if kw.arg == "abs" and tol >= (max(abs(expected), 1.0) if expected is not None else 1e9):
            return True
    return False


def _trivial(test: ast.expr) -> bool:
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return bool(getattr(test, "elts", None) or getattr(test, "keys", None))
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and isinstance(test.operand, ast.Constant):
        return not test.operand.value
    if isinstance(test, ast.BoolOp):
        combine = any if isinstance(test.op, ast.Or) else all
        return combine(_trivial(v) for v in test.values)
    if isinstance(test, ast.Call):
        return tail_of(test) == "isinstance" and len(test.args) == 2 and dotted(test.args[1]) == "object"
    if isinstance(test, ast.Compare):
        operands = [test.left, *test.comparators]
        pairs = list(zip(operands, operands[1:], test.ops, strict=False))  # n operands, n-1 pairs
        if all(isinstance(o, ast.Constant) for o in operands):
            try:
                return all(_CONST_OPS[type(op)](a.value, b.value) for a, b, op in pairs)  # type: ignore[attr-defined]  # operands narrowed above
            except (KeyError, TypeError):
                return False
        if all(isinstance(op, _SELF_OPS) and ast.dump(a) == ast.dump(b) for a, b, op in pairs):
            return True
        return any(isinstance(op, ast.Eq) and is_loose_approx(b) for _, b, op in pairs)
    return False


def is_trivial_assert(node: ast.Assert) -> bool:
    return _trivial(node.test)


def literal_text(node: ast.AST) -> str | None:
    """Normalised text of a literal on the expected side (snapshot(...) and approx(...) unwrapped)."""
    if isinstance(node, ast.Call) and tail_of(node) in {"snapshot", "approx"} and node.args:
        node = node.args[0]
    if isinstance(node, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.UnaryOp)):
        try:
            return repr(ast.literal_eval(node))
        except TypeError:  # a literal python cannot build, e.g. {[1]: 2}: compare its source text
            return ast.unparse(node)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    return None


def row_text(row: ast.expr) -> str:
    if isinstance(row, ast.Call) and tail_of(row) == "param" and row.args:
        return ast.unparse(row.args[0]) if len(row.args) == 1 else ast.unparse(ast.Tuple(elts=list(row.args), ctx=ast.Load()))
    return ast.unparse(row)


def parametrize_rows(decorators: list[ast.expr], consts: dict[str, ast.expr]) -> list[str]:
    """Rows of literal parametrize lists (positional or argvalues=, directly or via a module constant)."""
    rows: list[str] = []
    for dec in decorators:
        if not (isinstance(dec, ast.Call) and call_name(dec).endswith("parametrize")):
            continue
        values: ast.expr | None = dec.args[1] if len(dec.args) >= 2 else next((k.value for k in dec.keywords if k.arg == "argvalues"), None)
        if isinstance(values, ast.Name):
            values = consts.get(values.id)
        if isinstance(values, (ast.List, ast.Tuple)):
            rows.extend(row_text(e) for e in values.elts)
    return rows


def body_is_empty(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        return True
    first = body[0]
    if isinstance(first, ast.Return):
        return True
    return all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) for s in body)


@dataclass
class Scan:
    """What a test body checks, counting only code that can run."""

    checks: int = 0
    exact: int = 0
    trivial: int = 0
    skips: int = 0

    def __post_init__(self) -> None:
        self.raises: list[tuple[str, str | None]] = []
        self.asserts: list[ast.Assert] = []
        self.exact_calls: list[ast.Call] = []


def static_truth(test: ast.expr) -> bool | None:
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = static_truth(test.operand)
        return None if inner is None else not inner
    return None


def _record_call(call: ast.Call, scan: Scan) -> None:
    tail = tail_of(call)
    if tail in CHECK_CALL_TAILS or tail.startswith("assert"):
        scan.checks += 1
        if tail.startswith("assert") and any(w in tail.lower() for w in EXACT_WORDS):
            scan.exact += 1
            scan.exact_calls.append(call)
    if tail in RAISES_TAILS:
        exc = ast.unparse(call.args[0]) if call.args else ""
        match = next((ast.unparse(k.value) for k in call.keywords if k.arg == "match"), None)
        scan.raises.append((exc, match))


def _visit_expr(node: ast.AST, scan: Scan, aliases: dict[str, str]) -> None:
    scan.skips += count_skips(node, aliases)
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Lambda):
            continue
        if isinstance(cur, ast.Call):
            _record_call(cur, scan)
        stack.extend(ast.iter_child_nodes(cur))


def _record_assert(node: ast.Assert, scan: Scan, aliases: dict[str, str]) -> None:
    test = node.test
    scan.checks += len(test.values) if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else 1
    scan.trivial += is_trivial_assert(node)
    for sub in ast.walk(test):  # == / != comparisons, and closeness helpers such as math.isclose, np.allclose
        if isinstance(sub, ast.Compare):
            scan.exact += sum(isinstance(op, (ast.Eq, ast.NotEq)) for op in sub.ops)
        elif isinstance(sub, ast.Call) and any(w in tail_of(sub).lower() for w in ("equal", "close", "approx")):
            scan.exact += 1
    scan.asserts.append(node)
    _visit_expr(test, scan, aliases)


def scan_block(stmts: list[ast.stmt], scan: Scan, aliases: dict[str, str], called: set[str]) -> bool:
    """Count checks in the statements that can run. Returns True when the block always exits."""
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name in called:  # a nested helper only counts if the test refers to it
                scan_block(stmt.body, scan, aliases, called)
        elif isinstance(stmt, ast.ClassDef):
            continue
        elif isinstance(stmt, ast.If):
            _visit_expr(stmt.test, scan, aliases)
            truth = static_truth(stmt.test)
            body_exits = scan_block(stmt.body, scan, aliases, called) if truth is not False else False
            else_exits = scan_block(stmt.orelse, scan, aliases, called) if truth is not True else False
            exits = body_exits if truth is True else else_exits if truth is False else body_exits and else_exits
            if exits:
                return True
        elif isinstance(stmt, ast.While):
            _visit_expr(stmt.test, scan, aliases)
            if static_truth(stmt.test) is not False:
                scan_block(stmt.body, scan, aliases, called)
            scan_block(stmt.orelse, scan, aliases, called)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            _visit_expr(stmt.iter, scan, aliases)
            scan_block(stmt.body, scan, aliases, called)
            scan_block(stmt.orelse, scan, aliases, called)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                _visit_expr(item.context_expr, scan, aliases)
            if scan_block(stmt.body, scan, aliases, called):
                return True
        elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            for block in child_blocks(stmt):
                scan_block(block, scan, aliases, called)
        elif isinstance(stmt, ast.Match):
            _visit_expr(stmt.subject, scan, aliases)
            for case_ in stmt.cases:
                scan_block(case_.body, scan, aliases, called)
        elif isinstance(stmt, ast.Assert):
            _record_assert(stmt, scan, aliases)
        elif isinstance(stmt, (ast.Return, ast.Raise)):
            _visit_expr(stmt, scan, aliases)
            return True
        elif isinstance(stmt, (ast.Break, ast.Continue)):
            return True
        else:
            _visit_expr(stmt, scan, aliases)
    return False


def expected_literals(scan: Scan) -> list[str]:
    """Literals compared with in the live asserts (and assertEqual-style calls), in source order."""
    found: list[tuple[tuple[int, int], str]] = []

    def take(node: ast.expr) -> None:
        text = literal_text(node)
        if text is not None:
            found.append(((node.lineno, node.col_offset), text))

    for node in scan.asserts:
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Compare):
                for operand in (sub.left, *sub.comparators):
                    take(operand)
            elif isinstance(sub, ast.Call) and any(w in tail_of(sub).lower() for w in ("equal", "close")):
                for arg in sub.args:
                    take(arg)
    for call in scan.exact_calls:
        for arg in call.args:
            take(arg)
    return [text for _, text in sorted(found, key=lambda item: item[0])]


@dataclass
class CaseFacts:
    key: str
    line: int
    rows: list[str]
    checks: int
    exact: int
    trivial: int
    skips: int
    empty: bool
    expected: list[str]
    raises: list[tuple[str, str | None]]

    @property
    def cases(self) -> int:
        return len(self.rows)


def _is_test_class(node: ast.ClassDef) -> bool:
    return node.name.startswith("Test") or any(dotted(b).rsplit(".", 1)[-1] == "TestCase" for b in node.bases)


def facts_for_tests(data: bytes | str) -> tuple[dict[str, CaseFacts], int, ast.Module | None]:
    """Facts per test function (key 'Class.test_x'), plus module-level skip count, plus the tree."""
    tree, _ = parse_python(data)
    if tree is None:
        return {}, 0, None
    aliases = build_aliases(tree)
    consts: dict[str, ast.expr] = {}
    module_skips = 0
    module_rows: list[str] = []
    for stmt in plain_statements(tree.body):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            consts[stmt.targets[0].id] = stmt.value
        if not is_marker_definition(stmt, aliases):
            module_skips += count_skips(stmt, aliases)
        module_rows += parametrize_rows(pytestmark_values(stmt), consts)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_autouse_fixture(node) and node in tree.body:
            module_skips += sum(count_skips(s, aliases) for s in node.body)
    facts: dict[str, CaseFacts] = {}

    def add_test(fn: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str, inherited_skips: int, inherited_rows: list[str]) -> None:
        scan = Scan()
        called = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        scan_block(fn.body, scan, aliases, called)
        key = prefix + fn.name
        facts[key] = CaseFacts(
            key=key, line=fn.lineno, rows=inherited_rows + parametrize_rows(fn.decorator_list, consts),
            checks=scan.checks, exact=scan.exact, trivial=scan.trivial,
            skips=scan.skips + inherited_skips + sum(count_skips(d, aliases) for d in fn.decorator_list),
            empty=body_is_empty(fn), expected=expected_literals(scan), raises=scan.raises,
        )

    def visit(body: list[ast.stmt], prefix: str, inherited_skips: int, inherited_rows: list[str]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef) and _is_test_class(node):
                body_stmts = plain_statements(node.body)
                skips = inherited_skips + sum(count_skips(d, aliases) for d in node.decorator_list)
                skips += sum(count_skips(s, aliases) for s in body_stmts if not is_marker_definition(s, aliases))
                rows = inherited_rows + parametrize_rows(node.decorator_list, consts)
                for s in body_stmts:
                    rows += parametrize_rows(pytestmark_values(s), consts)
                visit(node.body, f"{prefix}{node.name}.", skips, rows)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                add_test(node, prefix, inherited_skips, inherited_rows)
            else:
                for block in child_blocks(node):
                    visit(block, prefix, inherited_skips, inherited_rows)

    visit(tree.body, "", 0, module_rows)
    return facts, module_skips, tree


@dataclass
class ConftestFacts:
    entries: Counter
    hooks: dict[str, str]
    skips: int


def conftest_facts(data: bytes | str) -> ConftestFacts:
    """collect_ignore entries (by source text), collection hooks (by AST dump), skip count."""
    tree, _ = parse_python(data)
    if tree is None:
        return ConftestFacts(Counter(), {}, 0)
    entries: Counter = Counter()
    hooks: dict[str, str] = {}

    def entry_texts(value: ast.expr | None) -> list[str]:
        if value is None:
            return []
        return [ast.unparse(e) for e in value.elts] if isinstance(value, (ast.List, ast.Tuple)) else [ast.unparse(value)]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in COLLECT_IGNORE_NAMES for t in targets):
                entries.update(entry_texts(node.value))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id in COLLECT_IGNORE_NAMES and node.args:
                if node.func.attr == "append":
                    entries.update([ast.unparse(node.args[0])])
                elif node.func.attr == "extend":
                    entries.update(entry_texts(node.args[0]))
                elif node.func.attr == "insert" and len(node.args) > 1:
                    entries.update([ast.unparse(node.args[1])])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in COLLECTION_HOOKS:
            hooks[node.name] = ast.dump(node)
    return ConftestFacts(entries, hooks, count_skips(tree, build_aliases(tree)))


def generate_hooks(data: bytes | str) -> list[tuple[str, int]]:
    """pytest_generate_tests functions as (normalised AST text, line).

    Comments never reach the AST and the docstring is left out, so only a change that
    can alter which parameter sets run (signature, decorators, statements) differs.
    """
    tree, _ = parse_python(data)
    if tree is None:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "pytest_generate_tests":
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            parts = [ast.dump(d) for d in node.decorator_list] + [ast.dump(node.args)] + [ast.dump(stmt) for stmt in body]
            out.append(("\n".join(parts), node.lineno))
    return out


# skips reached through helpers and fixtures -------------------------------------------
Func = ast.FunctionDef | ast.AsyncFunctionDef
CallableId = tuple[str, str]  # (file, function name)


def module_functions(body: list[ast.stmt]) -> list[Func]:
    """Functions defined at module level, including under if/try/with (not in classes or functions)."""
    out: list[Func] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(node)
        elif not isinstance(node, ast.ClassDef):
            for block in child_blocks(node):
                out.extend(module_functions(block))
    return out


def own_nodes(fn: Func) -> list[ast.AST]:
    """Nodes of the body that run when fn runs: nested defs, lambdas and classes are not entered."""
    found: list[ast.AST] = []
    stack: list[ast.AST] = list(fn.body)
    while stack:
        cur = stack.pop()
        found.append(cur)
        if not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(cur))
    return found


def skips_directly(fn: Func, aliases: dict[str, str]) -> bool:
    for node in own_nodes(fn):
        if isinstance(node, ast.Call) and (resolve(dotted(node.func), aliases) in SKIP_FUNCS
                                           or (isinstance(node.func, ast.Attribute) and node.func.attr == "skipTest")):
            return True
        if isinstance(node, ast.Raise) and node.exc is not None:
            raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if dotted(raised).rsplit(".", 1)[-1] in SKIP_EXCEPTIONS:
                return True
    return False


def fixture_name(fn: Func) -> tuple[str, bool] | None:
    """(name tests request it by, autouse) for a fixture function, else None."""
    for dec in fn.decorator_list:
        if tail_of(dec) not in {"fixture", "yield_fixture"}:
            continue
        keywords = dec.keywords if isinstance(dec, ast.Call) else []
        name = next((k.value.value for k in keywords if k.arg == "name" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)), fn.name)
        return name, any(k.arg == "autouse" and isinstance(k.value, ast.Constant) and k.value.value is True for k in keywords)
    return None


def called_names(fn: Func) -> set[str]:
    return {n.func.id for n in own_nodes(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def requested_fixtures(fn: Func) -> set[str]:
    """Fixtures named in the parameters or in @pytest.mark.usefixtures("a", "b")."""
    args = fn.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)} - {"self", "cls"}
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call) and tail_of(dec) == "usefixtures":
            names |= {a.value for a in dec.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
    return names


def iter_tests(body: list[ast.stmt], prefix: str = "") -> list[tuple[str, Func]]:
    """Test functions as ('Class.test_x', node), found the way facts_for_tests finds them."""
    out: list[tuple[str, Func]] = []
    for node in body:
        if isinstance(node, ast.ClassDef) and _is_test_class(node):
            out.extend(iter_tests(node.body, f"{prefix}{node.name}."))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            out.append((prefix + node.name, node))
        else:
            for block in child_blocks(node):
                out.extend(iter_tests(block, prefix))
    return out


@dataclass
class _Module:
    file: str
    aliases: dict[str, str]
    helpers: dict[str, Func]  # ordinary functions: not a test, a fixture or a pytest_ hook
    fixtures: dict[str, Func]  # non-autouse fixtures by the name tests request (autouse is SKIP_ADDED's job)
    conftest_imports: dict[str, str]  # local name -> name in conftest, from `from conftest import x [as y]`
    tests: dict[str, Func]
    skips: int


def _read_module(file: str, data: bytes) -> _Module | None:
    tree, _ = parse_python(data)
    if tree is None:
        return None
    module = _Module(file, build_aliases(tree), {}, {}, {}, dict(iter_tests(tree.body)), count_skips(tree, build_aliases(tree)))
    for fn in module_functions(tree.body):
        fixture = fixture_name(fn)
        if fixture is not None:
            if not fixture[1]:
                module.fixtures[fixture[0]] = fn
        elif not fn.name.startswith(("test", "pytest_")):
            module.helpers[fn.name] = fn
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and (node.module == "conftest" or node.module.endswith(".conftest")):
            module.conftest_imports.update({a.asname or a.name: a.name for a in node.names})
    return module


class SkipReach:
    """Which helpers and fixtures of a test tree can reach a skip, and which of them each test uses.

    Resolution is deliberately shallow: a plain name in the same module, a name imported
    with `from conftest import x`, and fixtures of the same module or of a conftest.py in
    the same or a parent directory. Nothing else is followed.
    """

    def __init__(self, files: dict[str, bytes]) -> None:
        self.modules = {f: m for f, d in files.items() if (m := _read_module(f, d)) is not None}
        self.nodes: dict[CallableId, tuple[Func, _Module]] = {}
        edges: dict[CallableId, list[CallableId]] = {}
        for m in self.modules.values():
            for name, fn in m.helpers.items():
                self.nodes[(m.file, name)] = (fn, m)
            for name, fn in m.fixtures.items():
                self.nodes[(m.file, name)] = (fn, m)
        for cid, (fn, m) in self.nodes.items():
            requests = requested_fixtures(fn) if fixture_name(fn) is not None else set()
            edges[cid] = [i for i in (*(self.helper_id(m.file, n) for n in called_names(fn)),
                                      *(self.fixture_id(m.file, n) for n in requests)) if i is not None]
        self.reach = {cid: skips_directly(fn, m.aliases) for cid, (fn, m) in self.nodes.items()}
        for _ in range(len(self.reach) + 1):  # reach only ever flips to True, so this ends in at most one pass per callable
            changed = False
            for cid, outs in edges.items():
                if not self.reach[cid] and any(self.reach[o] for o in outs):
                    self.reach[cid] = changed = True
            if not changed:
                break

    def conftests(self, file: str) -> list[str]:
        parent = PurePosixPath(file).parent
        return [path for d in (parent, *parent.parents) if (path := _join(d.as_posix(), "conftest.py")) in self.modules]

    def helper_id(self, file: str, name: str) -> CallableId | None:
        module = self.modules[file]
        if name in module.helpers:
            return (file, name)
        original = module.conftest_imports.get(name)
        if original is not None:
            return next(((c, original) for c in self.conftests(file) if original in self.modules[c].helpers), None)
        return None

    def fixture_id(self, file: str, name: str) -> CallableId | None:
        return next(((c, name) for c in dict.fromkeys([file, *self.conftests(file)]) if name in self.modules[c].fixtures), None)

    def test_uses(self, file: str, key: str) -> tuple[set[str], dict[str, CallableId]]:
        """(every name the test calls or requests, those that resolve to a callable that can skip)."""
        fn = self.modules[file].tests[key]
        calls, requests = called_names(fn), requested_fixtures(fn)
        resolved = {n: i for n in calls if (i := self.helper_id(file, n)) is not None}
        resolved.update({n: i for n in requests if (i := self.fixture_id(file, n)) is not None})
        return calls | requests, {n: i for n, i in resolved.items() if self.reach[i]}


# config ------------------------------------------------------------------------------
@dataclass
class PytestCfg:
    file: str
    addopts: list[str]
    options: dict[str, list[str] | None]


@dataclass
class CoverageCfg:
    file: str
    fail_under: float | None
    omit: set[str]


def _decode_config(name: str, data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnreadableError(f"cannot read {name}: not valid UTF-8 ({exc})") from exc


def load_toml(name: str, data: bytes) -> dict[str, Any]:
    if tomllib is None:
        raise UnreadableError(
            f"cannot parse {name}: this Python has no tomllib. Run diff_guard with Python 3.11 or newer "
            "(or install the tomli package); TOML config is never skipped silently"
        )
    try:
        return tomllib.loads(_decode_config(name, data))
    except tomllib.TOMLDecodeError as exc:
        raise UnreadableError(f"cannot parse {name}: {exc}") from exc


def load_ini(name: str, data: bytes) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(_decode_config(name, data))
    except configparser.Error as exc:
        raise UnreadableError(f"cannot parse {name}: {exc}") from exc
    return parser


def _join(directory: str, filename: str) -> str:
    return filename if directory in ("", ".") else f"{directory}/{filename}"


def _words(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return raw.split()
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _args(name: str, raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            return shlex.split(raw)
        except ValueError as exc:
            raise UnreadableError(f"cannot split addopts in {name}: {exc}") from exc
    return [str(x) for x in raw] if isinstance(raw, list) else []


def _pytest_cfg(name: str, table: dict[str, Any]) -> PytestCfg:
    options: dict[str, list[str] | None] = {}
    for key in ("testpaths", *PYTEST_DEFAULTS):
        options[key] = _words(table[key]) if key in table else None
    return PytestCfg(name, _args(name, table.get("addopts", [])), options)


def pytest_config(files: dict[str, bytes], directory: str) -> PytestCfg | None:
    """The pytest config of one directory: the first file in pytest's own order that applies."""
    for fname in PYTEST_CONFIG_ORDER:
        path = _join(directory, fname)
        data = files.get(path)
        if data is None:
            continue
        if fname in ("pytest.toml", ".pytest.toml"):
            table = load_toml(path, data).get("pytest", {})
            return _pytest_cfg(path, table if isinstance(table, dict) else {})
        if fname == "pyproject.toml":
            tool = load_toml(path, data).get("tool", {}).get("pytest")
            if not isinstance(tool, dict):
                continue
            table = {k: v for k, v in tool.items() if k != "ini_options"}
            if isinstance(tool.get("ini_options"), dict):
                table.update(tool["ini_options"])
            return _pytest_cfg(path, table)
        parser = load_ini(path, data)
        section = {"tox.ini": "pytest", "setup.cfg": "tool:pytest"}.get(fname, "pytest")
        if parser.has_section(section):
            return _pytest_cfg(path, dict(parser[section]))
        if fname in ("pytest.ini", ".pytest.ini"):  # always the source of config, even without a [pytest] section
            return _pytest_cfg(path, {})
    return None


def _float(name: str, raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise UnreadableError(f"cannot read fail_under {raw!r} in {name}") from exc


def _omit(raw: Any) -> set[str]:
    if isinstance(raw, str):
        return {p.strip() for line in raw.splitlines() for p in line.split(",") if p.strip()}
    return {str(x) for x in raw} if isinstance(raw, list) else set()


def _coverage_from_tables(name: str, report: dict[str, Any], run: dict[str, Any]) -> CoverageCfg:
    return CoverageCfg(name, _float(name, report.get("fail_under")), _omit(report.get("omit")) | _omit(run.get("omit")))


def coverage_config(files: dict[str, bytes], directory: str) -> CoverageCfg | None:
    """The coverage config of one directory: the first file in coverage's order that carries settings."""
    for fname in COVERAGE_CONFIG_ORDER:
        path = _join(directory, fname)
        data = files.get(path)
        if data is None:
            continue
        if fname.endswith(".toml"):
            doc = load_toml(path, data)
            tool = doc.get("tool", {}).get("coverage") if fname == "pyproject.toml" else doc.get("tool", {}).get("coverage", doc)
            if not isinstance(tool, dict) or (fname == "pyproject.toml" and not tool):
                continue
            report, run = tool.get("report", {}), tool.get("run", {})
            return _coverage_from_tables(path, report if isinstance(report, dict) else {}, run if isinstance(run, dict) else {})
        parser = load_ini(path, data)
        prefixes = ("coverage:", "") if fname == ".coveragerc" else ("coverage:",)
        if fname != ".coveragerc" and not any(s.startswith("coverage:") for s in parser.sections()):
            continue

        def table(section: str, parser: configparser.ConfigParser = parser, prefixes: tuple[str, ...] = prefixes) -> dict[str, Any]:
            for prefix in prefixes:
                if parser.has_section(prefix + section):
                    return dict(parser[prefix + section])
            return {}

        return _coverage_from_tables(path, table("report"), table("run"))
    return None


def narrowing_tokens(addopts: list[str]) -> Counter:
    """Options that make pytest collect or run less, normalised to '<flag> <value>'."""
    out: Counter = Counter()
    for i, opt in enumerate(addopts):
        if opt in NARROWING_BARE_FLAGS:
            out[opt] += 1
            continue
        for flag in NARROWING_VALUE_FLAGS:
            if opt == flag:
                value = addopts[i + 1] if i + 1 < len(addopts) else ""
            elif flag.startswith("--") and opt.startswith(flag + "="):
                value = opt[len(flag) + 1:]
            elif not flag.startswith("--") and not opt.startswith("--") and opt.startswith(flag) and len(opt) > len(flag):
                value = opt[len(flag):]  # attached short form: -k'expr', -kexpr
            else:
                continue
            out[f"{flag} {value}".strip()] += 1
            break
    return out


def cov_fail_under_flag(name: str, addopts: list[str]) -> float | None:
    """--cov-fail-under=N or '--cov-fail-under N' (the last one wins; it overrides the coverage file)."""
    value = None
    for i, opt in enumerate(addopts):
        if opt.startswith("--cov-fail-under="):
            value = _float(name, opt.split("=", 1)[1])
        elif opt == "--cov-fail-under" and i + 1 < len(addopts):
            value = _float(name, addopts[i + 1])
    return value


def effective_floor(pytest_cfg: PytestCfg | None, cov_cfg: CoverageCfg | None) -> tuple[float, str] | None:
    if pytest_cfg is not None:
        flag = cov_fail_under_flag(pytest_cfg.file, pytest_cfg.addopts)
        if flag is not None:
            return flag, pytest_cfg.file
    if cov_cfg is not None and cov_cfg.fail_under is not None:
        return cov_cfg.fail_under, cov_cfg.file
    return None


# source-side cheats ------------------------------------------------------------------
TRIVIAL_LITERALS = frozenset({0, 1, -1, 2, "", None})


def _signed_number(node: ast.AST) -> int | float | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = node.operand
        if isinstance(inner, ast.Constant) and isinstance(inner.value, (int, float)) and not isinstance(inner.value, bool):
            return -inner.value if isinstance(node.op, ast.USub) else inner.value
    return None


def literal_values(data: bytes | str) -> set:
    """Non-trivial int/float/str constants (negatives folded) in a test file; docstrings excluded."""
    tree, _ = parse_python(data)
    if tree is None:
        return set()
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out = set()
    folded: set[int] = set()  # the operand of a folded -41 must not also count as 41
    for node in ast.walk(tree):
        value: Any
        negative = _signed_number(node)
        if negative is not None:
            value = negative
            folded.add(id(node.operand))  # type: ignore[attr-defined]  # _signed_number only accepts a UnaryOp
        elif isinstance(node, ast.Constant) and id(node) in folded:
            continue
        elif isinstance(node, ast.Constant) and id(node) not in docstrings and type(node.value) in (int, float, str):
            value = node.value
        else:
            continue
        if value not in TRIVIAL_LITERALS and not (isinstance(value, str) and len(value) < 2):
            out.add(value)
    return out


def _constant_like(node: ast.AST | None) -> bool:
    if node is None or isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _constant_like(node.operand)
    if isinstance(node, ast.BoolOp):
        return all(_constant_like(v) for v in node.values)
    if isinstance(node, ast.IfExp):
        return _constant_like(node.body) and _constant_like(node.orelse)
    if isinstance(node, ast.Call):
        return tail_of(node) == "bool" and len(node.args) == 1 and _constant_like(node.args[0])
    return False


def constant_dunders(tree: ast.Module) -> set[str]:
    out = set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name in {"__eq__", "__ne__", "__hash__"}:
                body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
                prelude, last = body[:-1], body[-1:]
                if (last and isinstance(last[0], ast.Return) and _constant_like(last[0].value)
                        and all(isinstance(s, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Pass)) for s in prelude)):
                    out.add(f"{cls.name}.{fn.name}")
            elif (isinstance(fn, ast.Assign) and len(fn.targets) == 1 and isinstance(fn.targets[0], ast.Name)
                  and fn.targets[0].id in {"__eq__", "__ne__", "__hash__"}
                  and isinstance(fn.value, ast.Lambda) and _constant_like(fn.value.body)):
                out.add(f"{cls.name}.{fn.targets[0].id}")
    return out


def literal_comparisons(tree: ast.Module) -> Counter:
    """Comparisons with a literal on either side, keyed by AST dump, with the literal values and line."""
    out: Counter = Counter()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Is, ast.IsNot))):
            continue
        for operand in (node.comparators[0], node.left):
            try:
                value = ast.literal_eval(operand)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            values = tuple(value) if isinstance(value, (tuple, list, set, frozenset)) else (value,)
            try:
                out[(ast.dump(node), values, node.lineno)] += 1
            except TypeError:  # unhashable values: not a special-casing shape
                pass
            break
    return out


def source_findings(base_src: dict[str, bytes], cur_src: dict[str, bytes], test_literals: set) -> list[Finding]:
    findings = []
    for name, data in sorted(cur_src.items()):
        new_tree, reason = parse_python(data)
        if new_tree is None:
            findings.append(Finding(name, 1, "SYNTAX", f"{name} does not parse: {reason}", CODES["SYNTAX"][0]))
            continue
        old_tree = parse_python(base_src.get(name, b""))[0] or ast.parse("")
        for dunder in sorted(constant_dunders(new_tree) - constant_dunders(old_tree)):
            findings.append(Finding(name, 1, "CONSTANT_EQ", f"{dunder} returns a constant", CODES["CONSTANT_EQ"][0]))
        old_dumps = Counter(key[0] for key in literal_comparisons(old_tree))
        seen: Counter = Counter()
        for (dump, values, line), _n in sorted(literal_comparisons(new_tree).items(), key=lambda kv: kv[0][2]):
            seen[dump] += 1
            if seen[dump] <= old_dumps[dump]:
                continue
            hits = [v for v in values if v in test_literals and v not in TRIVIAL_LITERALS]
            if hits:
                findings.append(Finding(name, line, "SPECIAL_CASE", f"compares with {hits[0]!r}, which a test also uses", CODES["SPECIAL_CASE"][0]))
    return findings


def changed_source(root: Path, base: str, base_names: list[str]) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Non-test .py files changed since base (tracked diffs plus untracked), as (base, current) bytes."""
    names = set(git_paths(root, "diff", "--name-only", "-z", "--no-renames", base, "--"))
    names |= set(git_paths(root, "ls-files", "-z", "--others", "--exclude-standard"))
    in_base = set(base_names)
    old, new = {}, {}
    for n in sorted(names):
        if not n.endswith(".py") or is_test_file(n):
            continue
        if n in in_base:
            old[n] = run_git(root, "cat-file", "blob", f"{base}:{n}")
        if (root / n).is_file():
            new[n] = (root / n).read_bytes()
    return old, new


def pair_renames(removed: dict[str, bytes], added: dict[str, bytes]) -> dict[str, str]:
    """Match removed files to added ones: identical content first, then similar text. {old: new}."""
    pairs: dict[str, str] = {}
    free = dict(added)
    for old_name, old_data in sorted(removed.items()):
        for new_name in sorted(free):
            if free[new_name] == old_data:
                pairs[old_name] = new_name
                del free[new_name]
                break
    left = {n: d for n, d in removed.items() if n not in pairs}
    if len(left) * len(free) > RENAME_PAIR_LIMIT:
        return pairs
    scored = []
    for old_name, old_data in left.items():
        old_lines = source_text(old_data).splitlines()
        for new_name, new_data in free.items():
            matcher = difflib.SequenceMatcher(None, old_lines, source_text(new_data).splitlines(), autojunk=False)
            ratio = matcher.ratio()
            if ratio >= RENAME_SIMILARITY_MIN:
                scored.append((ratio, old_name, new_name))
    for _, old_name, new_name in sorted(scored, reverse=True):
        if old_name not in pairs and new_name not in pairs.values():
            pairs[old_name] = new_name
    return pairs


# comparison --------------------------------------------------------------------------
def github_annotation(f: Finding) -> str:
    def esc(text: str, prop: bool = False) -> str:
        text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        return text.replace(":", "%3A").replace(",", "%2C") if prop else text

    return f"::{f.severity} file={esc(f.file, True)},line={f.line}::{esc(f.code + ' ' + f.detail)}"


def _compare_tests(base_tests: dict[str, bytes], cur_tests: dict[str, bytes], add) -> None:
    base_facts = {n: facts_for_tests(t) for n, t in base_tests.items()}
    cur_facts = {n: facts_for_tests(t) for n, t in cur_tests.items()}
    for n, facts in list(cur_facts.items()):
        if facts[2] is None:
            add(n, 1, "SYNTAX", f"{n} does not parse: {parse_python(cur_tests[n])[1]}")
            cur_facts[n] = base_facts.get(n, ({}, 0, None))
    base_count = Counter(k for f in base_facts.values() for k in f[0])
    cur_count = Counter(k for f in cur_facts.values() for k in f[0])
    cur_index: dict[str, list[tuple[str, CaseFacts]]] = {}
    for n, (facts, _, _) in sorted(cur_facts.items()):
        for k, fact in facts.items():
            cur_index.setdefault(k, []).append((n, fact))

    for name in sorted(base_tests):
        if name not in cur_tests:
            add(name, 1, "TEST_FILE_DELETED", f"{name} ({len(base_facts[name][0])} tests) deleted")

    reported_deleted: Counter = Counter()
    for name, (facts, base_module_skips, _) in sorted(base_facts.items()):
        cur = cur_facts.get(name)
        if cur is not None and cur[1] > base_module_skips and PurePosixPath(name).name != "conftest.py":
            add(name, 1, "SKIP_ADDED", "module-level skip/xfail (pytestmark or pytest.skip at import) added")
        for key, old in facts.items():
            missing = base_count[key] - cur_count[key] - reported_deleted[key]
            new_file = name
            new = cur[0].get(key) if cur else None
            if new is None and missing > 0:
                reported_deleted[key] += 1
                if name in cur_tests:
                    add(name, old.line, "TEST_DELETED", f"{key} removed (had {old.checks} checks)")
                continue
            if new is None:
                candidates = cur_index.get(key, [])
                if not candidates:
                    continue
                new_file, new = candidates[0]
            if new.empty and not old.empty:
                add(name, new.line, "TEST_EMPTIED", f"{key} body is now empty or returns immediately")
                continue
            _compare_test(add, name, new_file, key, old, new, base_module_skips, cur_facts[new_file][1])

    for name, data in sorted(cur_tests.items()):
        if count_pragmas(data) > count_pragmas(base_tests.get(name, b"")):
            add(name, 1, "PRAGMA_ADDED", f"{count_pragmas(data) - count_pragmas(base_tests.get(name, b''))} new audit pragma(s)")
        if PurePosixPath(name).name == "conftest.py":
            _compare_conftest(add, name, base_tests.get(name, b""), data)
        _compare_generate_hooks(add, name, base_tests.get(name, b""), data)
    _compare_helper_skips(add, base_tests, cur_tests)


def _compare_test(add, name: str, new_file: str, key: str, old: CaseFacts, new: CaseFacts, old_module_skips: int, new_module_skips: int) -> None:
    if new.cases < old.cases:
        add(new_file, new.line, "CASES_REDUCED", f"{key}: {old.cases} parametrize rows -> {new.cases}")
    else:
        replaced = list((Counter(old.rows) - Counter(new.rows)).elements())
        if replaced:
            add(new_file, new.line, "EXPECTATION_CHANGED", f"{key}: {len(replaced)} parametrize row(s) replaced: " + "; ".join(replaced[:3]))
    if new.checks < old.checks:
        add(new_file, new.line, "ASSERTS_REDUCED", f"{key}: {old.checks} checks -> {new.checks}")
    elif new.exact < old.exact and new.trivial <= old.trivial:
        add(new_file, new.line, "ASSERT_WEAKENED", f"{key}: {old.exact} equality comparisons -> {new.exact}")
    if len(old.raises) == len(new.raises):
        for (old_exc, old_match), (new_exc, new_match) in zip(old.raises, new.raises, strict=True):
            if new_exc in BROAD_EXCEPTIONS and old_exc not in BROAD_EXCEPTIONS:
                add(new_file, new.line, "ASSERT_WEAKENED", f"{key}: pytest.raises({old_exc}) widened to {new_exc}")
            elif old_match is not None and new_match is None:
                add(new_file, new.line, "ASSERT_WEAKENED", f"{key}: pytest.raises(match={old_match}) lost its match=")
            elif old_match != new_match and new_match is not None and _literal_str(new_match) in MATCH_ANYTHING:
                add(new_file, new.line, "ASSERT_WEAKENED", f"{key}: pytest.raises match= loosened to {new_match}")
    if new.trivial > old.trivial:
        add(new_file, new.line, "ASSERT_TRIVIALIZED", f"{key}: {new.trivial - old.trivial} assertion(s) that cannot fail")
    moved = new_file != name  # a moved test also inherits the module-level skips of its new file
    if new.skips + (new_module_skips if moved else 0) > old.skips + (old_module_skips if moved else 0):
        add(new_file, new.line, "SKIP_ADDED", f"{key}: skip/xfail added")
    if new.exact < old.exact or new.trivial > old.trivial:
        return  # already reported as weakened/trivialized; a changed literal would only repeat it
    if old.expected and new.expected and old.expected != new.expected and len(old.expected) == len(new.expected):
        changed = [f"{a} -> {b}" for a, b in zip(old.expected, new.expected, strict=True) if a != b]
        add(new_file, new.line, "EXPECTATION_CHANGED", f"{key}: " + "; ".join(changed[:3]))
    elif new.checks >= old.checks and (dropped := list((Counter(old.expected) - Counter(new.expected)).elements())):
        add(new_file, new.line, "EXPECTATION_CHANGED", f"{key}: expected literal(s) no longer asserted: " + "; ".join(dropped[:3]))


def _literal_str(text: str) -> str | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def _compare_generate_hooks(add, name: str, old_data: bytes, new_data: bytes) -> None:
    """A pytest_generate_tests that is new or whose normalised body differs from every one in the base."""
    unmatched = Counter(dump for dump, _ in generate_hooks(old_data))
    had_hook = bool(unmatched)
    for dump, line in generate_hooks(new_data):
        if unmatched[dump] > 0:
            unmatched[dump] -= 1
            continue
        what = "body changed" if had_hook else "added"
        add(name, line, "PARAM_HOOK_CHANGED", f"pytest_generate_tests {what}: it decides which parameter sets run; confirm no cases were dropped")


def _compare_helper_skips(add, base_tests: dict[str, bytes], cur_tests: dict[str, bytes]) -> None:
    base, cur = SkipReach(base_tests), SkipReach(cur_tests)
    # a conftest that gained a skip is already SKIP_ADDED; its helpers and fixtures are not reported again
    exempt = {f for f, m in cur.modules.items() if PurePosixPath(f).name == "conftest.py" and m.skips > (base.modules[f].skips if f in base.modules else 0)}
    base_names: dict[str, set[str]] = {}
    for f, m in base.modules.items():
        for key in m.tests:
            base_names.setdefault(key, set()).update(base.test_uses(f, key)[0])
    callers: dict[CallableId, list[tuple[str, bool]]] = {}  # helper -> (test key, did the base test already use it)
    for f in sorted(cur.modules):
        for key, fn in cur.modules[f].tests.items():
            if key not in base_names:
                continue  # a new test may use a skipping helper
            _, reaching = cur.test_uses(f, key)
            reaching = {n: i for n, i in reaching.items() if i[0] not in exempt}
            for name, cid in reaching.items():
                callers.setdefault(cid, []).append((key, name in base_names[key]))
            gained = sorted(n for n in reaching if n not in base_names[key])
            if gained:
                add(f, fn.lineno, "SKIP_VIA_HELPER", f"{key} now uses {', '.join(gained[:3])}, which can reach pytest.skip/xfail/importorskip; confirm the skip condition is real")
    for cid, users in sorted(callers.items()):
        was_skipping = base.reach.get(cid)
        if was_skipping is False and any(already for _, already in users):
            n = len(users)
            add(cid[0], cur.nodes[cid][0].lineno, "SKIP_VIA_HELPER",
                f"{cid[1]} can now reach pytest.skip/xfail/importorskip and {n} test{'s' if n != 1 else ''} use it; confirm the skip condition is real")


def _compare_conftest(add, name: str, old_data: bytes, new_data: bytes) -> None:
    old, new = conftest_facts(old_data), conftest_facts(new_data)
    gained = sorted((new.entries - old.entries).elements())
    if gained:
        add(name, 1, "COLLECTION_NARROWED", f"collect_ignore gained {gained}")
    for hook in sorted(new.hooks):
        if hook not in old.hooks:
            add(name, 1, "COLLECTION_NARROWED", f"{hook} added; check it does not deselect failing tests")
        elif new.hooks[hook] != old.hooks[hook]:
            add(name, 1, "COLLECTION_NARROWED", f"{hook} body changed; check it does not deselect failing tests")
    if new.skips > old.skips:
        add(name, 1, "SKIP_ADDED", "conftest gained a skip/xfail (a hook or autouse fixture can skip every test)")


def _compare_configs(base: dict[str, bytes], current: dict[str, bytes], add) -> None:
    directories = {PurePosixPath(n).parent.as_posix() for n in (*base, *current) if is_config_file(n)}
    for directory in sorted(directories):
        old_p, new_p = pytest_config(base, directory), pytest_config(current, directory)
        old_c, new_c = coverage_config(base, directory), coverage_config(current, directory)
        if old_p is not None or new_p is not None:
            label = (new_p or old_p).file  # type: ignore[union-attr]  # one of them is not None
            old_opts = old_p.addopts if old_p else []
            new_opts = new_p.addopts if new_p else []
            for token in sorted(narrowing_tokens(new_opts) - narrowing_tokens(old_opts)):
                add(label, 1, "CONFIG_NARROWED", f"addopts gained '{token}'")
            old_tp = old_p.options["testpaths"] if old_p and old_p.options["testpaths"] else []
            new_tp = new_p.options["testpaths"] if new_p and new_p.options["testpaths"] else []
            if sorted(set(old_tp) - set(new_tp)):
                add(label, 1, "CONFIG_NARROWED", f"testpaths lost {sorted(set(old_tp) - set(new_tp))}")
            _compare_collection_options(add, label, old_p, new_p)
        old_floor, new_floor = effective_floor(old_p, old_c), effective_floor(new_p, new_c)
        if old_floor is not None and (new_floor is None or new_floor[0] < old_floor[0]):
            where = (new_floor or old_floor)[1]
            now = f"{new_floor[0]:g}" if new_floor else "removed"
            add(where, 1, "COVERAGE_LOWERED", f"fail_under {old_floor[0]:g} -> {now}")
        if new_c is not None:
            gained = sorted(new_c.omit - (old_c.omit if old_c else set()))
            if gained:
                add(new_c.file, 1, "COVERAGE_LOWERED", f"coverage omit gained {gained}")


def _compare_collection_options(add, label: str, old_p: PytestCfg | None, new_p: PytestCfg | None) -> None:
    if new_p is None:
        return
    for key, default in PYTEST_DEFAULTS.items():
        new_value = new_p.options[key]
        if new_value is None:
            continue
        old_value = (old_p.options[key] if old_p else None) or default
        if key == "norecursedirs":
            gained = sorted(set(new_value) - set(old_value))
            if gained:
                add(label, 1, "CONFIG_NARROWED", f"norecursedirs gained {gained}")
        else:
            lost = [p for p in old_value if not any(fnmatchcase(p, q) for q in new_value)]
            if lost:
                add(label, 1, "CONFIG_NARROWED", f"{key} lost {lost}")


def compare(base: dict[str, bytes], current: dict[str, bytes]) -> list[Finding]:
    findings: list[Finding] = []

    def add(file: str, line: int, code: str, detail: str) -> None:
        findings.append(Finding(file, line, code, detail, CODES[code][0]))

    _compare_tests({n: t for n, t in base.items() if is_test_file(n)}, {n: t for n, t in current.items() if is_test_file(n)}, add)

    for name in sorted(n for n in base if is_snapshot_file(n)):
        if name not in current:
            add(name, 1, "SNAPSHOT_CHANGED", "snapshot file deleted")
        elif current[name] != base[name]:
            add(name, 1, "SNAPSHOT_CHANGED", "snapshot file rewritten (--snapshot-update?)")

    _compare_configs({n: d for n, d in base.items() if is_config_file(n)}, {n: d for n, d in current.items() if is_config_file(n)}, add)
    return findings


def guard(start: Path, base: str = "HEAD", allow: frozenset[str] = frozenset()) -> list[Finding]:
    root = repo_root(start)
    try:
        git(root, "rev-parse", "--verify", f"{base}^{{commit}}")
    except GitError as exc:
        raise GitError(f"--base {base!r} is not a commit in this repository ({exc})") from exc
    base_names = base_tree(root, base)
    base_rel = base_files(root, base, base_names)
    current = current_files(root)
    if not base_rel and not current:
        raise UnreadableError(f"nothing to compare: no test files, snapshots or pytest/coverage config in {base} or the working tree")

    renames = pair_renames({n: d for n, d in base_rel.items() if n not in current and is_test_file(n)},
                           {n: d for n, d in current.items() if n not in base_rel and is_test_file(n)})
    base_rel = {renames.get(n, n): d for n, d in base_rel.items()}
    findings = compare(base_rel, current)

    test_literals: set = set().union(*(literal_values(d) for n, d in current.items() if is_test_file(n)))
    old_src, new_src = changed_source(root, base, base_names)
    src_renames = pair_renames({n: d for n, d in old_src.items() if n not in new_src}, {n: d for n, d in new_src.items() if n not in old_src})
    old_src = {src_renames.get(n, n): d for n, d in old_src.items()}
    findings += source_findings(old_src, new_src, test_literals)
    return [f for f in findings if f.code not in allow]


def emit(stream, text: str) -> None:
    """Print text that may hold surrogate-escaped (non-UTF-8) file names without crashing."""
    stream.write(text.encode("utf-8", "backslashreplace").decode("utf-8") + "\n")


def main(argv: list[str] | None = None) -> int:
    assert __doc__ is not None
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", default=".", help="any directory inside the repository")
    ap.add_argument("--base", default="HEAD", help="git ref to compare against (default HEAD)")
    ap.add_argument("--allow", action="append", default=[], help="comma-separated codes to accept, e.g. TEST_DELETED (repeatable)")
    ap.add_argument("--format", choices=("text", "json", "github"), default="text")
    ap.add_argument("--warnings-as-errors", action="store_true")
    ap.add_argument("--list-codes", action="store_true")
    args = ap.parse_args(argv)
    if args.list_codes:
        for code, (sev, text) in CODES.items():
            print(f"{code:20} {sev:8} {text}")
        return 0
    allow = frozenset(c.strip() for value in args.allow for c in value.split(",") if c.strip())
    unknown = allow - set(CODES)
    if unknown:
        emit(sys.stderr, f"diff_guard: unknown codes in --allow: {sorted(unknown)}")
        return 2
    try:
        findings = guard(Path(args.path), args.base, allow)
    except (GitError, UnreadableError) as exc:
        emit(sys.stderr, f"diff_guard: {exc}")
        return 2
    except FileNotFoundError as exc:
        emit(sys.stderr, f"diff_guard: cannot run git or read a file: {exc}")
        return 2
    except OSError as exc:
        emit(sys.stderr, f"diff_guard: I/O error: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001  # last resort: a bug here must be exit 2 with a message, never a traceback that reads like exit 1
        emit(sys.stderr, f"diff_guard: internal error, nothing was checked: {type(exc).__name__}: {exc}")
        return 2
    if args.format == "json":
        emit(sys.stdout, json.dumps([asdict(f) for f in findings], indent=2))
    elif args.format == "github":
        for f in findings:
            emit(sys.stdout, github_annotation(f))
    elif findings:
        emit(sys.stdout, "\n".join(f.render() for f in findings))
    else:
        emit(sys.stdout, f"clean: no test-weakening edits since {args.base}")
    failing = [f for f in findings if f.severity == ERROR or args.warnings_as_errors]
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
