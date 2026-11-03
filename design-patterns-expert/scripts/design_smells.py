#!/usr/bin/env python3
"""Scan Python source for structural signals that suggest (or argue against) a design pattern.

These are LEADS, NOT VERDICTS. A five-branch if/elif that will never grow a sixth
branch is fine as it is. The value of this script is that it finds the mechanical
signals fast and consistently, so the judgement call is made against a complete
list rather than whatever caught the eye first.

Usage:
    python design_smells.py path/to/file.py path/to/package/
    python design_smells.py src/ --json
    python design_smells.py src/ --only dispatch_chain,concrete_construction
    python design_smells.py src/ --threshold max_params=7 --domain-dirs core,rules
    python design_smells.py src/ --fail-on-findings        # exit 1 if anything is found (CI)

Stdlib only. Exit codes: 0 report produced, 1 findings with --fail-on-findings,
2 bad arguments (for example an unknown kind passed to --only).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Modules whose presence inside business logic usually means infrastructure has
# leaked inward. Not exhaustive, and an adapter module is *supposed* to import these.
INFRA_MODULES = frozenset({
    "sqlalchemy", "psycopg", "psycopg2", "pymongo", "redis", "boto3", "botocore",
    "requests", "httpx", "aiohttp", "urllib3", "smtplib", "sqlite3", "flask",
    "fastapi", "django", "starlette", "kafka", "pika", "celery", "elasticsearch",
    "google", "azure", "openai", "anthropic", "stripe",
})

# Package or file-name tokens that suggest a module is meant to hold pure logic.
# Matched as whole tokens ("scorecard" does not match "core").
DOMAIN_HINTS = frozenset({"domain", "core", "model", "models", "entities", "usecase",
                          "usecases", "use_cases", "business", "rules", "services"})

# Class-name suffixes that usually mean "talks to the outside world".
INFRA_SUFFIXES = ("Client", "Session", "Repository", "Repo", "Gateway", "Emailer",
                  "Mailer", "Connection", "Engine", "Producer", "Consumer", "Publisher",
                  "Adapter", "Api", "API", "SMTP")
# Factory functions on infra modules that create a connection-like object.
INFRA_FACTORIES = frozenset({"client", "resource", "connect", "create_engine",
                             "Session", "Client", "AsyncClient", "SMTP", "Redis",
                             "MongoClient", "KafkaProducer", "KafkaConsumer"})
# Functions allowed to construct concrete things: the composition root.
COMPOSITION_ROOT = re.compile(
    r"^(main|create_app|make_app|bootstrap|setup|configure|wire|compose|lifespan)$"
    r"|^(build|make|create|wire|compose|setup|configure|get)_"
    r"|factory|_factory$"
)
TEST_FILE = re.compile(r"(^test_.*|.*_test|conftest)\.py$")
# Packages whose job is to talk to infrastructure: construction there is expected.
ADAPTER_HINTS = frozenset({"adapters", "adapter", "infra", "infrastructure", "gateways", "drivers"})

DEFAULT_THRESHOLDS = {
    "dispatch_branches": 3,     # branches on one subject before flagging
    "isinstance_branches": 2,
    "max_params": 5,            # excluding self/cls
    "max_methods": 10,          # public methods per class
    "max_function_lines": 60,
    "max_nesting": 4,
    "max_bool_flags": 3,        # keyword-only bool params before 2^N becomes a smell
}

KINDS = {
    "dispatch_chain": "if/elif or match on one value selecting behaviour",
    "isinstance_chain": "if/elif isinstance or match on class patterns",
    "long_param_list": "many parameters threaded through",
    "boolean_positional": "bool parameter callers pass positionally (f(x, True))",
    "boolean_flags": "many keyword-only bool flags (2^N behaviours)",
    "long_function": "function body longer than the threshold",
    "deep_nesting": "control flow nested past the threshold (elif does not count)",
    "god_class": "class with many public methods",
    "singleton": "hand-rolled Singleton",
    "infra_in_domain": "domain-looking module imports infrastructure",
    "concrete_construction": "infrastructure object constructed inside logic",
    "import_time_io": "I/O performed when the module is imported",
    "cache_on_method": "@cache/@lru_cache on an instance method (pins every self)",
}


@dataclass
class Finding:
    file: str
    line: int
    kind: str
    detail: str
    candidates: str
    snippet: str = ""


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<expr>"


def _subject_of_test(test: ast.expr) -> str | None:
    """`if kind == "pdf"`, `if "pdf" == kind` and `if kind in (...)` all have subject `kind`."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op, right = test.ops[0], test.comparators[0]
        if isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.Is)):
            if isinstance(test.left, ast.Constant) and not isinstance(right, ast.Constant):
                return _unparse(right)
            return _unparse(test.left)
    return None


def _isinstance_subject(test: ast.expr) -> str | None:
    if (isinstance(test, ast.Call) and isinstance(test.func, ast.Name)
            and test.func.id == "isinstance" and test.args):
        return _unparse(test.args[0])
    return None


def _chain(node: ast.If) -> list[ast.If]:
    """Flatten an if/elif/elif chain into its `If` nodes."""
    out, cur = [node], node
    while len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
        cur = cur.orelse[0]
        out.append(cur)
    return out


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    a = fn.args
    return [p for p in (*a.posonlyargs, *a.args, *a.kwonlyargs) if p.arg not in ("self", "cls")]


def _is_bool_annotation(node: ast.expr | None) -> bool:
    return (isinstance(node, ast.Name) and node.id == "bool") or (
        isinstance(node, ast.Constant) and node.value == "bool")


def _is_bool_constant(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, bool)


_NESTING = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try,
            ast.Match, *((ast.TryStar,) if hasattr(ast, "TryStar") else ()))


def _nesting_depth(node: ast.AST, depth: int = 0) -> int:
    """Max control-flow depth under `node`. An `elif` is a sibling branch, not a level."""
    best = depth
    if isinstance(node, ast.If):
        for child in node.body:
            best = max(best, _nesting_depth(child, depth + 1 if isinstance(child, _NESTING) else depth))
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            best = max(best, _nesting_depth(node.orelse[0], depth))      # elif: same level
        else:
            for child in node.orelse:
                best = max(best, _nesting_depth(child, depth + 1 if isinstance(child, _NESTING) else depth))
        return best
    for sub in ast.iter_child_nodes(node):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue  # nested defs are measured on their own
        best = max(best, _nesting_depth(sub, depth + 1 if isinstance(sub, _NESTING) else depth))
    return best


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def _tokens(part: str) -> set[str]:
    low = part.lower()
    return {low, *filter(None, re.split(r"[^a-z0-9]+", low))}


def _module_parts(path: Path, scan_root: Path) -> list[str]:
    """Package path of a file: the chain of parent dirs that hold __init__.py, plus the stem.
    Falls back to the path relative to the scan target when there is no package."""
    parts: list[str] = [path.stem]
    cur = path.parent
    while (cur / "__init__.py").exists():
        parts.insert(0, cur.name)
        if cur.parent == cur:
            break
        cur = cur.parent
    if len(parts) == 1:
        try:
            rel = path.parent.resolve().relative_to(scan_root.resolve())
            parts = [*rel.parts, path.stem]
        except ValueError:
            pass
    return parts


def _walk_own_body(fn: ast.AST) -> Iterator[ast.AST]:
    """ast.walk that does not descend into nested functions or classes (they are checked on their own)."""
    stack: list[ast.AST] = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _decorator_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    names = []
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        names.append(_unparse(target))
    return names


# --------------------------------------------------------------------------- #
# the scanner
# --------------------------------------------------------------------------- #

class Scanner(ast.NodeVisitor):
    def __init__(self, path: Path, source: str, thresholds: dict[str, int],
                 scan_root: Path, domain_hints: frozenset[str]) -> None:
        self.path = path
        self.lines = source.splitlines()
        self.t = thresholds
        self.findings: list[Finding] = []
        self._seen_if: set[int] = set()
        self._reported_infra: set[str] = set()
        self._is_test = bool(TEST_FILE.match(path.name))
        parts = _module_parts(path, scan_root)
        self._looks_like_domain = any(_tokens(p) & domain_hints for p in parts)
        self._is_adapter = any(_tokens(p) & ADAPTER_HINTS for p in parts)
        self._infra_names: set[str] = set()      # names bound to infra imports in this module
        self._class_stack: list[ast.ClassDef] = []
        self._func_stack: list[str] = []

    def add(self, node: ast.AST, kind: str, detail: str, candidates: str) -> None:
        line = getattr(node, "lineno", 0)
        snippet = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        if len(snippet) > 96:
            snippet = snippet[:93] + "..."
        self.findings.append(Finding(str(self.path), line, kind, detail, candidates, snippet))

    # -- module level ------------------------------------------------------ #

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self._check_import_time_io(stmt)
        self.generic_visit(node)

    def _check_import_time_io(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Import, ast.ImportFrom, ast.If)):
            return  # `if __name__ == "__main__":` and other guarded blocks are fine
        for call in (n for n in ast.walk(stmt) if isinstance(n, ast.Call)):
            name = _unparse(call.func)
            root = _root_module(name)
            last = name.rsplit(".", 1)[-1]
            if name == "open" or (root in INFRA_MODULES | self._infra_names and "." in name
                                  and last in INFRA_FACTORIES | {"get", "post", "urlopen"}) \
                    or last in {"connect", "create_engine"}:
                self.add(stmt, "import_time_io",
                         f"module-level call {name}() runs on import",
                         "Move into a function called from the composition root; "
                         "or @cache / PEP 562 __getattr__ (both can run twice under threads)")
                return

    # -- imports ----------------------------------------------------------- #

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = _root_module(alias.name)
            if root in INFRA_MODULES:
                self._infra_names.add(alias.asname or root)
            self._check_infra_import(node, root)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.level == 0:
            root = _root_module(node.module)
            if root in INFRA_MODULES:
                self._infra_names.update(a.asname or a.name for a in node.names)
            self._check_infra_import(node, root)
        self.generic_visit(node)

    def _check_infra_import(self, node: ast.AST, root: str) -> None:
        if root in INFRA_MODULES and self._looks_like_domain and root not in self._reported_infra:
            self._reported_infra.add(root)
            self.add(node, "infra_in_domain",
                     f"module path looks like domain logic but imports {root!r}",
                     "Ports & Adapters, Repository, Dependency Injection")

    # -- conditionals ------------------------------------------------------ #

    def visit_If(self, node: ast.If) -> None:
        if id(node) not in self._seen_if:
            chain = _chain(node)
            self._seen_if.update(id(n) for n in chain)
            self._check_dispatch(chain)
            self._check_isinstance(chain)
        self.generic_visit(node)

    def _check_dispatch(self, chain: list[ast.If]) -> None:
        concrete = [s for s in (_subject_of_test(n.test) for n in chain) if s is not None]
        if len(concrete) >= self.t["dispatch_branches"] and len(set(concrete)) == 1:
            self.add(chain[0], "dispatch_chain",
                     f"{len(concrete)}-branch if/elif dispatching on {concrete[0]!r}",
                     "Strategy (dict of callables); Registry if variants live in other files")

    def _check_isinstance(self, chain: list[ast.If]) -> None:
        concrete = [s for s in (_isinstance_subject(n.test) for n in chain) if s is not None]
        if len(concrete) >= self.t["isinstance_branches"] and len(set(concrete)) == 1:
            self.add(chain[0], "isinstance_chain",
                     f"{len(concrete)}-branch dispatch on the type of {concrete[0]!r}",
                     "match + Union + assert_never if the set of types is closed; "
                     "functools.singledispatch if it is open")

    def visit_Match(self, node: ast.Match) -> None:
        values = sum(isinstance(c.pattern, (ast.MatchValue, ast.MatchSingleton)) for c in node.cases)
        classes = sum(isinstance(c.pattern, ast.MatchClass) for c in node.cases)
        subject = _unparse(node.subject)
        if values >= self.t["dispatch_branches"]:
            self.add(node, "dispatch_chain", f"{values}-case match dispatching on {subject!r}",
                     "Strategy (dict of callables) if each case selects behaviour; "
                     "fine as is if the cases are data, not behaviour")
        elif classes >= max(self.t["isinstance_branches"], 3):
            self.add(node, "isinstance_chain", f"{classes}-case match on the type of {subject!r}",
                     "Fine for a closed Union with assert_never; singledispatch if the set is open")
        self.generic_visit(node)

    # -- functions --------------------------------------------------------- #

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._check_function(fn)
        self._check_cache_on_method(fn)
        self._check_construction(fn)
        self._func_stack.append(fn.name)
        self.generic_visit(fn)
        self._func_stack.pop()

    def _check_function(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        params = _params(fn)
        if len(params) > self.t["max_params"]:
            self.add(fn, "long_param_list", f"{fn.name}() takes {len(params)} parameters",
                     "Context Object (group related ones into a dataclass); or split the function")

        is_dunder = fn.name.startswith("__") and fn.name.endswith("__")
        is_setter = any(n.endswith(".setter") for n in _decorator_names(fn))
        positional = [*fn.args.posonlyargs, *fn.args.args]
        defaults: list[ast.expr | None] = [None] * (len(positional) - len(fn.args.defaults)) + list(fn.args.defaults)
        pos_flags = [a.arg for a, d in zip(positional, defaults, strict=True)
                     if a.arg not in ("self", "cls") and (_is_bool_annotation(a.annotation) or _is_bool_constant(d))]
        kw_flags = [a.arg for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True)
                    if _is_bool_annotation(a.annotation) or _is_bool_constant(d)]
        if pos_flags and not (is_dunder or is_setter):
            self.add(fn, "boolean_positional",
                     f"{fn.name}() takes bool positionally: {', '.join(pos_flags)}",
                     "Make it keyword-only (*, flag: bool), or split into two functions, or an Enum "
                     "(ruff FBT001 flags the same thing)")
        elif len(kw_flags) >= self.t["max_bool_flags"]:
            self.add(fn, "boolean_flags",
                     f"{fn.name}() has {len(kw_flags)} bool flags: {', '.join(kw_flags)}",
                     f"2^{len(kw_flags)} behaviours to reason about: split the function, "
                     "or a Strategy/Policy object if the flags select behaviour")

        end = getattr(fn, "end_lineno", None)
        if end is not None and (end - fn.lineno) > self.t["max_function_lines"]:
            self.add(fn, "long_function", f"{fn.name}() spans {end - fn.lineno} lines",
                     "Extract steps; Template Method or a pipeline if the steps vary")

        depth = _nesting_depth(fn)
        if depth >= self.t["max_nesting"]:
            self.add(fn, "deep_nesting", f"{fn.name}() nests control flow {depth} levels deep",
                     "Guard clauses; State machine if the nesting encodes states")

    def _check_cache_on_method(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not self._class_stack or self._func_stack:
            return
        names = _decorator_names(fn)
        if any(n in ("staticmethod", "classmethod") for n in names):
            return
        if not (fn.args.args and fn.args.args[0].arg == "self"):
            return
        if any(n.split(".")[-1] in ("cache", "lru_cache") for n in names):
            self.add(fn, "cache_on_method",
                     f"{fn.name}() is cached per call, and the cache keeps every self alive",
                     "functools.cached_property (per instance), or cache a module-level function "
                     "keyed on plain values")

    def _check_construction(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self._is_test or self._is_adapter or any(
                COMPOSITION_ROOT.search(n) for n in (*self._func_stack, fn.name)):
            return
        built: list[str] = []
        for node in _walk_own_body(fn):
            if not isinstance(node, ast.Call):
                continue
            name = _unparse(node.func)
            last = name.rsplit(".", 1)[-1]
            root = _root_module(name)
            via_module = "." in name and root in INFRA_MODULES | self._infra_names and last in INFRA_FACTORIES
            by_name = ("." not in name and last[:1].isupper()
                       and (last.endswith(INFRA_SUFFIXES) or last in self._infra_names))
            if (via_module or by_name) and name not in built:
                built.append(name)
        if built:
            self.add(fn, "concrete_construction",
                     f"{fn.name}() constructs {', '.join(built)} itself",
                     "Dependency Injection: take it as a parameter and construct it in the "
                     "composition root (main/create_app)")

    # -- classes ----------------------------------------------------------- #

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        public = [m for m in methods if not m.name.startswith("_")]
        if len(public) > self.t["max_methods"]:
            self.add(node, "god_class", f"class {node.name} exposes {len(public)} public methods",
                     "Split by responsibility; Facade only if callers still need the parts")

        names = {m.name for m in methods}
        attrs = {t.id for n in node.body if isinstance(n, ast.Assign)
                 for t in n.targets if isinstance(t, ast.Name)}
        attrs |= {n.target.id for n in node.body
                  if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        meta = [_unparse(kw.value) for kw in node.keywords if kw.arg == "metaclass"]
        bases = [_unparse(b) for b in node.bases]
        if "__new__" in names and attrs & {"_instance", "_instances", "instance", "__instance"}:
            self.add(node, "singleton", f"class {node.name} looks like a hand-rolled Singleton",
                     "A Python module is already a singleton; or a function guarded by a lock. "
                     "See python-idioms.md")
        elif any("Singleton" in m for m in meta) or any(b.split(".")[-1].startswith("Singleton") for b in bases):
            self.add(node, "singleton", f"class {node.name} uses a singleton metaclass or base",
                     "A Python module is already a singleton. See python-idioms.md")

        self._class_stack.append(node)
        saved, self._func_stack = self._func_stack, []
        self.generic_visit(node)
        self._func_stack = saved
        self._class_stack.pop()


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

SKIP_DIRS = {".venv", "venv", "node_modules", "__pycache__", ".git", "build", "dist",
             ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


def iter_python_files(targets: list[str]) -> list[tuple[Path, Path]]:
    """(file, scan_root) pairs."""
    out: list[tuple[Path, Path]] = []
    for target in targets:
        p = Path(target)
        if p.is_file() and p.suffix == ".py":
            out.append((p, p.parent))
        elif p.is_dir():
            out.extend((f, p) for f in sorted(p.rglob("*.py"))
                       if not any(part in SKIP_DIRS for part in f.relative_to(p).parts))
    return out


def scan(targets: list[str], thresholds: dict[str, int] | None = None,
         domain_hints: frozenset[str] = DOMAIN_HINTS) -> Report:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    report = Report()
    for path, root in iter_python_files(targets):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            report.errors.append(f"{path}: {exc}")
            continue
        report.files_scanned += 1
        scanner = Scanner(path, source, t, root, domain_hints)
        scanner.visit(tree)
        report.findings.extend(scanner.findings)
    report.findings.sort(key=lambda f: (f.file, f.line, f.kind))
    return report


def render(report: Report, filtered: set[str] | None = None) -> str:
    if not report.findings:
        if filtered:
            return (f"No findings of kind {', '.join(sorted(filtered))} in "
                    f"{report.files_scanned} file(s). Other kinds were not checked.")
        return (f"No structural smells found in {report.files_scanned} file(s).\n"
                "That is a real result: recommend no pattern.")
    out: list[str] = []
    by_file: dict[str, list[Finding]] = {}
    for f in report.findings:
        by_file.setdefault(f.file, []).append(f)
    for filename, findings in by_file.items():
        out += [f"\n{filename}", "-" * len(filename)]
        for f in findings:
            out.append(f"  L{f.line}  [{f.kind}] {f.detail}")
            if f.snippet:
                out.append(f"         | {f.snippet}")
            out.append(f"         -> {f.candidates}")
    counts: dict[str, int] = {}
    for f in report.findings:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    out.append(f"\n{len(report.findings)} finding(s) across {report.files_scanned} file(s): "
               + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    out.append("Leads, not verdicts. Confirm each against how the code actually changes "
               "before recommending a pattern.")
    if report.errors:
        out += ["\nCould not parse:", *(f"  {e}" for e in report.errors)]
    return "\n".join(out)


def _parse_threshold(text: str) -> tuple[str, int]:
    name, sep, value = text.partition("=")
    if not sep or name not in DEFAULT_THRESHOLDS or not value.isdigit():
        raise argparse.ArgumentTypeError(
            f"expected NAME=INT with NAME in {', '.join(DEFAULT_THRESHOLDS)}; got {text!r}")
    return name, int(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find structural signals that suggest a design pattern.")
    parser.add_argument("paths", nargs="*", help="files or directories to scan")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--only", default="", help=f"comma-separated kinds: {', '.join(KINDS)}")
    parser.add_argument("--threshold", action="append", type=_parse_threshold, default=[],
                        metavar="NAME=INT", help=f"override a threshold ({', '.join(DEFAULT_THRESHOLDS)})")
    parser.add_argument("--max-params", type=int, help="shorthand for --threshold max_params=N")
    parser.add_argument("--dispatch-branches", type=int, help="shorthand for --threshold dispatch_branches=N")
    parser.add_argument("--domain-dirs", default="", help="comma-separated package names that hold pure logic")
    parser.add_argument("--fail-on-findings", action="store_true", help="exit 1 when anything is reported")
    parser.add_argument("--list-kinds", action="store_true", help="print the finding kinds and exit")
    args = parser.parse_args(argv)

    if args.list_kinds:
        for k, v in KINDS.items():
            print(f"{k:22s} {v}")
        return 0
    if not args.paths:
        parser.error("the following arguments are required: paths")
    wanted = {k.strip() for k in args.only.split(",") if k.strip()}
    unknown = wanted - KINDS.keys()
    if unknown:
        print(f"error: unknown kind(s) {', '.join(sorted(unknown))}; valid: {', '.join(KINDS)}",
              file=sys.stderr)
        return 2
    thresholds = dict(args.threshold)
    if args.max_params is not None:
        thresholds["max_params"] = args.max_params
    if args.dispatch_branches is not None:
        thresholds["dispatch_branches"] = args.dispatch_branches
    hints = frozenset(h.strip().lower() for h in args.domain_dirs.split(",") if h.strip()) or DOMAIN_HINTS

    report = scan(args.paths, thresholds, hints)
    if wanted:
        report.findings = [f for f in report.findings if f.kind in wanted]
    if args.json:
        print(json.dumps({**asdict(report), "thresholds": {**DEFAULT_THRESHOLDS, **thresholds}}, indent=2))
    else:
        print(render(report, wanted or None))
    return 1 if args.fail_on_findings and report.findings else 0


if __name__ == "__main__":
    sys.exit(main())
