#!/usr/bin/env python3
"""Report edits that turn a test suite green without fixing anything.

A coding agent told "make the tests pass" can get there by deleting the failing test,
dropping its assertions, adding a skip, deselecting it in config, or lowering the
coverage floor. Every one of those leaves pytest green. This script compares the test
tree in the working directory (committed or not) with a git base and reports them.
It also warns on two source-side cheats: an __eq__ that returns a constant, and new
code that compares an input with a literal the tests use (special-casing).

Pure stdlib (ast, subprocess, tomllib, configparser). It never imports the code under
test.

Usage:
    python diff_guard.py                     compare the working tree with HEAD
    python diff_guard.py --base origin/main  compare with another ref
    python diff_guard.py --allow TEST_DELETED,EXPECTATION_CHANGED
    python diff_guard.py --format json|github
    python diff_guard.py --list-codes

Exit status: 1 if an error-severity finding remains (or any warning, with
--warnings-as-errors), 2 if the directory is not a git work tree or the base ref is
unknown, 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

ERROR, WARNING = "error", "warning"

CODES: dict[str, tuple[str, str]] = {
    "TEST_FILE_DELETED": (ERROR, "a test file present in the base is gone"),
    "TEST_DELETED": (ERROR, "a test function present in the base is gone from every test file"),
    "TEST_EMPTIED": (ERROR, "test body reduced to pass, ..., or an early return"),
    "ASSERTS_REDUCED": (ERROR, "test has fewer checks (assert, pytest.raises/warns, assert_* calls) than in the base"),
    "CASES_REDUCED": (ERROR, "a parametrize list lost rows (a failing case can be dropped this way)"),
    "ASSERT_TRIVIALIZED": (ERROR, "test gained an assertion that cannot fail (assert True, assert x == x)"),
    "SKIP_ADDED": (ERROR, "skip, skipif, xfail, pytest.skip() or importorskip added to a test, class, or module"),
    "PRAGMA_ADDED": (ERROR, "new '# audit: ignore' pragma silences the test audit"),
    "COLLECTION_NARROWED": (ERROR, "conftest gained collect_ignore entries or a collection/deselection hook"),
    "CONFIG_NARROWED": (ERROR, "pytest config gained --deselect, --ignore, --ignore-glob, -k, or lost testpaths"),
    "COVERAGE_LOWERED": (ERROR, "coverage fail_under / --cov-fail-under lowered"),
    "SYNTAX": (ERROR, "test file does not parse; its tests are not compared (pytest will fail to collect it)"),
    "EXPECTATION_CHANGED": (WARNING, "a literal on the expected side of an assert changed; confirm the spec changed"),
    "CONSTANT_EQ": (WARNING, "source gained __eq__/__ne__/__hash__ returning a constant: makes any == assertion pass"),
    "SPECIAL_CASE": (WARNING, "source gained a comparison with a literal that also appears in the tests: possible special-casing"),
    "SNAPSHOT_CHANGED": (WARNING, "a stored snapshot (syrupy __snapshots__/*.ambr) was rewritten or deleted; review it like code"),
}

CONFIG_FILES = ("pyproject.toml", "pytest.toml", "pytest.ini", "tox.ini", "setup.cfg", ".coveragerc")
NARROWING_FLAGS = ("--deselect", "--ignore", "--ignore-glob", "-k")
SKIP_MARKS = frozenset({"skip", "skipif", "xfail"})
SKIP_CALLS = frozenset({"skip", "xfail", "importorskip"})
CHECK_CALL_TAILS = frozenset({"raises", "warns", "deprecated_call", "fail", "RaisesGroup", "RaisesExc", "group_contains"})
COLLECTION_HOOKS = frozenset({"pytest_collection_modifyitems", "pytest_ignore_collect", "pytest_deselected"})
PRAGMA = re.compile(r"#\s*audit:\s*ignore(-file)?\[")


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    code: str
    detail: str
    severity: str

    def render(self) -> str:
        return f"{self.file}:{self.line}: {self.code}: {self.detail}"


# git ---------------------------------------------------------------------------------
class GitError(RuntimeError):
    pass


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise GitError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def repo_root(start: Path) -> Path:
    return Path(git(start, "rev-parse", "--show-toplevel").strip())


def base_files(root: Path, base: str) -> dict[str, str]:
    names = git(root, "ls-tree", "-r", "--name-only", base).splitlines()
    return {n: git(root, "show", f"{base}:{n}") for n in names if is_relevant(n)}


def current_files(root: Path) -> dict[str, str]:
    names = git(root, "ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    out = {}
    for n in names:
        path = root / n
        if is_relevant(n) and path.is_file():
            out[n] = path.read_text(encoding="utf-8", errors="replace")
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


def is_relevant(name: str) -> bool:
    return is_test_file(name) or is_snapshot_file(name) or PurePosixPath(name).name in CONFIG_FILES


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


def marker_skips(decorators: list[ast.expr]) -> int:
    return sum(1 for d in decorators if call_name(d).rsplit(".", 1)[-1] in SKIP_MARKS and "mark" in call_name(d))


def is_trivial_assert(node: ast.Assert) -> bool:
    test = node.test
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return bool(getattr(test, "elts", None) or getattr(test, "keys", None))
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Eq, ast.Is, ast.LtE, ast.GtE)):
        return ast.dump(test.left) == ast.dump(test.comparators[0])
    return False


def expected_literals(fn: ast.AST) -> list[str]:
    """Constants on the right of `assert x == <literal>`, in order."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            for comp in node.test.comparators:
                if isinstance(comp, ast.Call) and call_name(comp).rsplit(".", 1)[-1] == "snapshot" and comp.args:
                    comp = comp.args[0]  # inline-snapshot: snapshot(<literal>)
                if isinstance(comp, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.UnaryOp)):
                    try:
                        out.append(ast.literal_eval(comp).__repr__())
                    except ValueError:
                        continue
    return out


def param_cases(decorators: list[ast.expr]) -> int:
    """Rows in literal @pytest.mark.parametrize lists (0 when none or not a literal)."""
    total = 0
    for dec in decorators:
        if isinstance(dec, ast.Call) and call_name(dec).endswith("parametrize") and len(dec.args) >= 2:
            values = dec.args[1]
            if isinstance(values, (ast.List, ast.Tuple)):
                total += len(values.elts)
    return total


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
class CaseFacts:
    key: str
    line: int
    cases: int
    checks: int
    trivial: int
    skips: int
    empty: bool
    expected: list[str]


def facts_for_tests(source: str) -> tuple[dict[str, CaseFacts], int, ast.Module | None]:
    """Facts per test function (key 'Class.test_x'), plus module-level skip count."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}, 0, None
    module_skips = 0
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets):
            values = stmt.value.elts if isinstance(stmt.value, (ast.List, ast.Tuple)) else [stmt.value]
            module_skips += marker_skips(values)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and call_name(stmt.value).rsplit(".", 1)[-1] in SKIP_CALLS:
            module_skips += 1
    facts: dict[str, CaseFacts] = {}

    def visit(body: list[ast.stmt], prefix: str, inherited: int) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                visit(node.body, f"{prefix}{node.name}.", inherited + marker_skips(node.decorator_list))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                checks = trivial = skips = 0
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assert):
                        checks += 1
                        trivial += is_trivial_assert(sub)
                    elif isinstance(sub, ast.Call):
                        tail = call_name(sub).rsplit(".", 1)[-1]
                        if tail in CHECK_CALL_TAILS or tail.startswith("assert_"):
                            checks += 1
                        if tail in SKIP_CALLS and call_name(sub).startswith("pytest"):
                            skips += 1
                facts[prefix + node.name] = CaseFacts(
                    key=prefix + node.name, line=node.lineno, cases=param_cases(node.decorator_list),
                    checks=checks, trivial=trivial,
                    skips=skips + inherited + marker_skips(node.decorator_list),
                    empty=body_is_empty(node), expected=expected_literals(node),
                )

    visit(tree.body, "", 0)
    return facts, module_skips, tree


def conftest_facts(source: str) -> tuple[int, set[str]]:
    """(number of collect_ignore entries, collection hooks defined)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0, set()
    entries, hooks = 0, set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in {"collect_ignore", "collect_ignore_glob"} for t in targets):
                entries += len(node.value.elts) if isinstance(node.value, (ast.List, ast.Tuple)) else 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"append", "extend"}:
            if isinstance(node.func.value, ast.Name) and node.func.value.id in {"collect_ignore", "collect_ignore_glob"}:
                entries += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in COLLECTION_HOOKS:
            hooks.add(node.name)
    return entries, hooks


# config ------------------------------------------------------------------------------
@dataclass
class ConfigFacts:
    addopts: list[str]
    testpaths: list[str]
    fail_under: float | None


def config_facts(name: str, text: str) -> ConfigFacts:
    addopts: list[str] = []
    testpaths: list[str] = []
    fail_under = None
    base = PurePosixPath(name).name
    if base.endswith(".toml") and tomllib is not None:
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return ConfigFacts([], [], None)
        sections = [data.get("tool", {}).get("pytest", {}).get("ini_options", {}), data.get("tool", {}).get("pytest", {}), data.get("pytest", {})]
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            raw = sec.get("addopts", [])
            addopts += shlex.split(raw) if isinstance(raw, str) else [str(x) for x in raw]
            tp = sec.get("testpaths", [])
            testpaths += shlex.split(tp) if isinstance(tp, str) else [str(x) for x in tp]
        cov = data.get("tool", {}).get("coverage", {}).get("report", {})
        if "fail_under" in cov:
            fail_under = float(cov["fail_under"])
    else:
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read_string(text)
        except configparser.Error:
            return ConfigFacts([], [], None)
        for sect in ("pytest", "tool:pytest"):
            if parser.has_section(sect):
                addopts += shlex.split(parser[sect].get("addopts", ""))
                testpaths += parser[sect].get("testpaths", "").split()
        for sect in ("report", "coverage:report"):
            if parser.has_section(sect) and parser[sect].get("fail_under"):
                fail_under = float(parser[sect]["fail_under"])
    for opt in addopts:
        m = re.fullmatch(r"--cov-fail-under=(\d+(?:\.\d+)?)", opt)
        if m:
            fail_under = float(m.group(1)) if fail_under is None else min(fail_under, float(m.group(1)))
    return ConfigFacts(addopts, testpaths, fail_under)


def narrowing_tokens(addopts: list[str]) -> Counter:
    out: Counter = Counter()
    for i, opt in enumerate(addopts):
        for flag in NARROWING_FLAGS:
            if opt == flag:
                nxt = addopts[i + 1] if i + 1 < len(addopts) else ""
                out[f"{flag} {nxt}".strip()] += 1
            elif opt.startswith(flag + "="):
                out[opt.replace("=", " ", 1)] += 1
    return out


# source-side cheats ------------------------------------------------------------------
TRIVIAL_LITERALS = frozenset({0, 1, -1, 2, "", None})


def literal_values(source: str) -> set:
    """Non-trivial int/float/str constants in a test file (docstrings excluded)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out = set()
    for node in ast.walk(tree):
        is_value = isinstance(node, ast.Constant) and id(node) not in docstrings and type(node.value) in (int, float, str)
        if is_value and node.value not in TRIVIAL_LITERALS and not (isinstance(node.value, str) and len(node.value) < 2):
            out.add(node.value)
    return out


def constant_dunders(tree: ast.Module) -> set[str]:
    out = set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name in {"__eq__", "__ne__", "__hash__"}:
                body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
                if len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Constant):
                    out.add(f"{cls.name}.{fn.name}")
    return out


def literal_comparisons(tree: ast.Module) -> Counter:
    """Comparisons `<expr> ==/!=/in <literal>` keyed by their AST dump, with the literal values."""
    out: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.In, ast.Is)):
            comp = node.comparators[0]
            try:
                value = ast.literal_eval(comp)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            values = tuple(value) if isinstance(value, (tuple, list, set, frozenset)) else (value,)
            out[(ast.dump(node), values, node.lineno)] += 1
    return out


def source_findings(base_src: dict[str, str], cur_src: dict[str, str], test_literals: set) -> list[Finding]:
    findings = []
    for name, text in sorted(cur_src.items()):
        try:
            new_tree = ast.parse(text)
        except SyntaxError:
            continue
        try:
            old_tree = ast.parse(base_src.get(name, ""))
        except SyntaxError:
            old_tree = ast.parse("")
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


def changed_source(root: Path, base: str) -> tuple[dict[str, str], dict[str, str]]:
    """Non-test .py files changed since base (tracked diffs plus untracked), as (base, current) texts."""
    names = set(git(root, "diff", "--name-only", base, "--").splitlines())
    names |= set(git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    base_names = set(git(root, "ls-tree", "-r", "--name-only", base).splitlines())
    old, new = {}, {}
    for n in sorted(names):
        if not n.endswith(".py") or is_test_file(n):
            continue
        if n in base_names:
            old[n] = git(root, "show", f"{base}:{n}")
        if (root / n).is_file():
            new[n] = (root / n).read_text(encoding="utf-8", errors="replace")
    return old, new


# comparison --------------------------------------------------------------------------
def compare(base: dict[str, str], current: dict[str, str]) -> list[Finding]:
    findings: list[Finding] = []

    def add(file: str, line: int, code: str, detail: str) -> None:
        findings.append(Finding(file, line, code, detail, CODES[code][0]))

    base_tests = {n: t for n, t in base.items() if is_test_file(n)}
    cur_tests = {n: t for n, t in current.items() if is_test_file(n)}

    base_facts = {n: facts_for_tests(t) for n, t in base_tests.items()}
    cur_facts = {n: facts_for_tests(t) for n, t in cur_tests.items()}
    for n, facts in list(cur_facts.items()):
        if facts[2] is None:
            add(n, 1, "SYNTAX", f"{n} does not parse")
            cur_facts[n] = base_facts.get(n, ({}, 0, None))
    base_count = Counter(k for f in base_facts.values() for k in f[0])
    cur_count = Counter(k for f in cur_facts.values() for k in f[0])
    cur_index: dict[str, list[tuple[str, CaseFacts]]] = {}
    for n, (facts, _, _) in cur_facts.items():
        for k, fact in facts.items():
            cur_index.setdefault(k, []).append((n, fact))

    for name in sorted(base_tests):
        if name not in cur_tests:
            n_tests = len(base_facts[name][0])
            add(name, 1, "TEST_FILE_DELETED", f"{name} ({n_tests} tests) deleted")

    reported_deleted: Counter = Counter()
    for name, (facts, base_module_skips, _) in sorted(base_facts.items()):
        cur = cur_facts.get(name)
        if cur is not None and cur[1] > base_module_skips:
            add(name, 1, "SKIP_ADDED", "module-level skip/xfail (pytestmark or pytest.skip at import) added")
        for key, old in facts.items():
            missing = base_count[key] - cur_count[key] - reported_deleted[key]
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
                new = candidates[0][1]
            if new.empty and not old.empty:
                add(name, new.line, "TEST_EMPTIED", f"{key} body is now empty or returns immediately")
                continue
            if new.cases < old.cases:
                add(name, new.line, "CASES_REDUCED", f"{key}: {old.cases} parametrize rows -> {new.cases}")
            if new.checks < old.checks:
                add(name, new.line, "ASSERTS_REDUCED", f"{key}: {old.checks} checks -> {new.checks}")
            if new.trivial > old.trivial:
                add(name, new.line, "ASSERT_TRIVIALIZED", f"{key}: {new.trivial - old.trivial} assertion(s) that cannot fail")
            if new.skips > old.skips:
                add(name, new.line, "SKIP_ADDED", f"{key}: skip/xfail added")
            if old.expected and new.expected and old.expected != new.expected and len(old.expected) == len(new.expected):
                changed = [f"{a} -> {b}" for a, b in zip(old.expected, new.expected, strict=True) if a != b]
                add(name, new.line, "EXPECTATION_CHANGED", f"{key}: " + "; ".join(changed[:3]))

    for name, text in sorted(cur_tests.items()):
        old_pragmas = len(PRAGMA.findall(base_tests.get(name, "")))
        new_pragmas = len(PRAGMA.findall(text))
        if new_pragmas > old_pragmas:
            add(name, 1, "PRAGMA_ADDED", f"{new_pragmas - old_pragmas} new audit pragma(s)")
        if PurePosixPath(name).name == "conftest.py":
            old_entries, old_hooks = conftest_facts(base_tests.get(name, ""))
            new_entries, new_hooks = conftest_facts(text)
            if new_entries > old_entries:
                add(name, 1, "COLLECTION_NARROWED", f"collect_ignore entries {old_entries} -> {new_entries}")
            for hook in sorted(new_hooks - old_hooks):
                add(name, 1, "COLLECTION_NARROWED", f"{hook} added; check it does not deselect failing tests")

    for name in sorted(n for n in base if is_snapshot_file(n)):
        if name not in current:
            add(name, 1, "SNAPSHOT_CHANGED", "snapshot file deleted")
        elif current[name] != base[name]:
            add(name, 1, "SNAPSHOT_CHANGED", "snapshot file rewritten (--snapshot-update?)")

    for name in sorted(n for n in current if not is_test_file(n) and not is_snapshot_file(n)):
        new = config_facts(name, current[name])
        old = config_facts(name, base.get(name, ""))
        added = narrowing_tokens(new.addopts) - narrowing_tokens(old.addopts)
        for token in sorted(added):
            add(name, 1, "CONFIG_NARROWED", f"addopts gained '{token}'")
        lost = sorted(set(old.testpaths) - set(new.testpaths))
        if lost:
            add(name, 1, "CONFIG_NARROWED", f"testpaths lost {lost}")
        if old.fail_under is not None and (new.fail_under is None or new.fail_under < old.fail_under):
            add(name, 1, "COVERAGE_LOWERED", f"fail_under {old.fail_under:g} -> " + (f"{new.fail_under:g}" if new.fail_under is not None else "removed"))
    return findings


def guard(start: Path, base: str = "HEAD", allow: frozenset[str] = frozenset()) -> list[Finding]:
    root = repo_root(start)
    git(root, "rev-parse", "--verify", f"{base}^{{commit}}")
    current = current_files(root)
    findings = compare(base_files(root, base), current)
    test_literals = set().union(*(literal_values(t) for n, t in current.items() if is_test_file(n)))
    findings += source_findings(*changed_source(root, base), test_literals)
    return [f for f in findings if f.code not in allow]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", default=".", help="any directory inside the repository")
    ap.add_argument("--base", default="HEAD", help="git ref to compare against (default HEAD)")
    ap.add_argument("--allow", default="", help="comma-separated codes to accept, e.g. TEST_DELETED")
    ap.add_argument("--format", choices=("text", "json", "github"), default="text")
    ap.add_argument("--warnings-as-errors", action="store_true")
    ap.add_argument("--list-codes", action="store_true")
    args = ap.parse_args(argv)
    if args.list_codes:
        for code, (sev, text) in CODES.items():
            print(f"{code:20} {sev:8} {text}")
        return 0
    allow = frozenset(c.strip() for c in args.allow.split(",") if c.strip())
    unknown = allow - set(CODES)
    if unknown:
        print(f"unknown codes in --allow: {sorted(unknown)}", file=sys.stderr)
        return 2
    try:
        findings = guard(Path(args.path), args.base, allow)
    except GitError as exc:
        print(f"diff_guard: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps([asdict(f) for f in findings], indent=2))
    elif args.format == "github":
        for f in findings:
            print(f"::{f.severity} file={f.file},line={f.line}::{f.code} {f.detail}")
    elif findings:
        print("\n".join(f.render() for f in findings))
    else:
        print(f"clean: no test-weakening edits since {args.base}")
    failing = [f for f in findings if f.severity == ERROR or args.warnings_as_errors]
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
