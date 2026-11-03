#!/usr/bin/env python3
"""Static audit for Python async code. Stdlib only, no third-party imports.

Scans .py files with `ast` for patterns that make asyncio code silently wrong:
blocking calls on the event loop, fire-and-forget tasks that can be garbage
collected, `gather` used without deciding what happens on failure, unbounded
fan-out, swallowed cancellation, and legacy event-loop management.

Exit codes: 0 clean, 1 findings, 2 a target path does not exist. Works as a CI
or pre-commit gate.

    python audit_async.py src/
    python audit_async.py src/ --format json
    python audit_async.py src/ --ignore SEQUENTIAL_AWAIT,NO_CLIENT_TIMEOUT
    python audit_async.py --list-codes .
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

CODES: dict[str, str] = {
    "BLOCKING_SLEEP": "time.sleep() inside async def. Blocks the whole event loop; use await asyncio.sleep().",
    "BLOCKING_HTTP": "Synchronous HTTP call inside async def (requests, urllib, sync httpx). Use httpx.AsyncClient.",
    "BLOCKING_OPEN": "Builtin open()/Path.read_text()/write_text() inside async def. Use asyncio.to_thread() or aiofiles.",
    "BLOCKING_SUBPROCESS": "Blocking subprocess/os.system inside async def. Use asyncio.create_subprocess_exec().",
    "BLOCKING_SYNC_DB": "Synchronous DB driver inside async def (sqlite3, psycopg2, pymysql). Use the async driver or to_thread.",
    "BLOCKING_INPUT": "input() inside async def. Blocks the loop until a human types.",
    "BARE_GATHER": "asyncio.gather() without an explicit return_exceptions=. Decide the failure policy, or use TaskGroup.",
    "GATHER_LOOP_ARG": "asyncio.gather(loop=...): the loop argument was removed in Python 3.10.",
    "FLOATING_TASK": "create_task()/ensure_future() result discarded. The loop holds only a weak reference; the task may be GC'd mid-flight.",
    "UNAWAITED_COROUTINE": "Coroutine function called without await and without create_task. It never runs.",
    "WAIT_ON_COROUTINES": "asyncio.wait() passed raw coroutines. Forbidden since 3.11; wrap them in tasks first.",
    "SWALLOWED_CANCEL": "except CancelledError/BaseException without re-raising. Breaks cancellation and timeouts.",
    "LEGACY_LOOP": "get_event_loop()/new_event_loop()/run_until_complete(). get_event_loop() raises RuntimeError in 3.14.",
    "NO_CLIENT_TIMEOUT": "httpx client built without an explicit timeout=. The 5s default is rarely what you want.",
    "CLIENT_IN_LOOP": "HTTP client constructed inside a loop body. Rebuilds the connection pool every iteration.",
    "UNBOUNDED_FANOUT": "Fan-out over a comprehension with no Semaphore/CapacityLimiter in scope. Caps at 'however many items exist'.",
    "SEQUENTIAL_AWAIT": "await of a local coroutine function inside a for loop. Runs serially; you probably wanted a fan-out.",
    "ASYNC_GEN_YIELD_IN_CM": "yield inside a with/async with block in an async generator. Cleanup is deferred until the generator is closed, possibly never.",
    "BLOCKING_LOCK": "threading.Lock/threading.Event used inside async def. Use asyncio.Lock/asyncio.Event.",
}

BLOCKING_ATTR_CALLS: dict[tuple[str, ...], str] = {
    ("time", "sleep"): "BLOCKING_SLEEP",
    ("os", "system"): "BLOCKING_SUBPROCESS",
    ("subprocess", "run"): "BLOCKING_SUBPROCESS",
    ("subprocess", "call"): "BLOCKING_SUBPROCESS",
    ("subprocess", "Popen"): "BLOCKING_SUBPROCESS",
    ("subprocess", "check_call"): "BLOCKING_SUBPROCESS",
    ("subprocess", "check_output"): "BLOCKING_SUBPROCESS",
    ("sqlite3", "connect"): "BLOCKING_SYNC_DB",
    ("psycopg2", "connect"): "BLOCKING_SYNC_DB",
    ("pymysql", "connect"): "BLOCKING_SYNC_DB",
    ("MySQLdb", "connect"): "BLOCKING_SYNC_DB",
    ("threading", "Lock"): "BLOCKING_LOCK",
    ("threading", "RLock"): "BLOCKING_LOCK",
    ("threading", "Event"): "BLOCKING_LOCK",
    ("threading", "Semaphore"): "BLOCKING_LOCK",
}

SYNC_HTTP_ROOTS = {"requests", "urllib3"}
HTTPX_MODULES = {"httpx", "httpx2"}
SYNC_HTTP_VERBS = {"get", "post", "put", "patch", "delete", "head", "options", "request", "send"}
HTTPX_CLIENT_NAMES = {"AsyncClient", "Client"}
ASYNCIO_ALIASABLE = {"gather", "wait", "create_task", "ensure_future", "get_event_loop", "new_event_loop"}
ASYNC_CM_DECORATORS = {"asynccontextmanager", "fixture"}
BLOCKING_PATH_METHODS = {"read_text", "write_text", "read_bytes", "write_bytes"}


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    line: int
    col: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.code} {self.message}"


def dotted(node: ast.AST) -> tuple[str, ...]:
    """Flatten a.b.c into ('a', 'b', 'c'). Returns () for anything else."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return tuple(reversed(parts))
    return ()


def contains_reraise(handler: ast.ExceptHandler) -> bool:
    """True if the handler re-raises, propagates, or deliberately returns."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call) and dotted(node.func)[-1:] == ("cancel",):
            return True
    return False


def handler_catches_cancel(handler: ast.ExceptHandler) -> bool:
    types = handler.type
    if types is None:
        return True  # bare `except:` catches BaseException
    candidates = types.elts if isinstance(types, ast.Tuple) else [types]
    for c in candidates:
        name = dotted(c)[-1:] or (c.id,) if isinstance(c, ast.Name) else dotted(c)[-1:]
        if name and name[0] in {"CancelledError", "BaseException"}:
            return True
    return False


def _is_own_call(path: tuple[str, ...]) -> bool:
    """`fetch(...)`, `self.fetch(...)`, `cls.fetch(...)`: a call that resolves to a
    coroutine function defined in this module. `d.update(...)` on some other object
    is not, even when the module also has an `async def update`."""
    return len(path) == 1 or (len(path) == 2 and path[0] in {"self", "cls"})


def _walk_same_function(node: ast.AST):
    """ast.walk that does not descend into nested functions, lambdas, or classes."""
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        for child in ast.iter_child_nodes(cur):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                stack.append(child)


def _is_unmanaged_async_generator(node: ast.AsyncFunctionDef) -> bool:
    """An `async def` that yields, and is not an @asynccontextmanager or a pytest
    fixture (both of which close the generator deterministically)."""
    for deco in node.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if dotted(target)[-1:] and dotted(target)[-1] in ASYNC_CM_DECORATORS:
            return False
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in _walk_same_function(node))


class AsyncAuditor(ast.NodeVisitor):
    def __init__(self, path: str, source: str) -> None:
        self.path = path
        self.findings: list[Finding] = []
        self.async_depth = 0
        self.loop_depth = 0
        self.coroutine_names: set[str] = set()
        self.taskgroup_names: set[str] = set()
        self.asyncio_aliases: dict[str, str] = {}  # local name -> asyncio function name
        self.awaited_calls: set[int] = set()
        self.async_gen_stack: list[bool] = []
        self.has_limiter = "Semaphore" in source or "CapacityLimiter" in source
        self._prepass(source)

    def _prepass(self, source: str) -> None:
        """Collect coroutine function names and names bound to a task group.

        `tg.create_task(...)` inside `async with asyncio.TaskGroup() as tg` is
        correct — the group holds a strong reference and awaits the task. Only
        `asyncio.create_task`/`loop.create_task` leak.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "asyncio":
                for alias in node.names:
                    if alias.name in ASYNCIO_ALIASABLE:
                        self.asyncio_aliases[alias.asname or alias.name] = alias.name
            if isinstance(node, ast.AsyncFunctionDef):
                self.coroutine_names.add(node.name)
            elif isinstance(node, ast.AsyncWith):
                for item in node.items:
                    called = dotted(item.context_expr.func) if isinstance(item.context_expr, ast.Call) else ()
                    if called and called[-1] in {"TaskGroup", "create_task_group"}:
                        if isinstance(item.optional_vars, ast.Name):
                            self.taskgroup_names.add(item.optional_vars.id)

    def add(self, code: str, node: ast.AST, extra: str = "") -> None:
        msg = CODES[code] + (f" ({extra})" if extra else "")
        self.findings.append(
            Finding(code, self.path, getattr(node, "lineno", 0), getattr(node, "col_offset", 0), msg)
        )

    # -- scope tracking -------------------------------------------------

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.async_depth += 1
        outer_loop = self.loop_depth
        self.loop_depth = 0
        self.async_gen_stack.append(_is_unmanaged_async_generator(node))
        self.generic_visit(node)
        self.async_gen_stack.pop()
        self.loop_depth = outer_loop
        self.async_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        outer_async, outer_loop = self.async_depth, self.loop_depth
        self.async_depth = 0  # a sync def nested in an async def is not on the loop
        self.loop_depth = 0
        self.async_gen_stack.append(False)
        self.generic_visit(node)
        self.async_gen_stack.pop()
        self.async_depth, self.loop_depth = outer_async, outer_loop

    # A lambda inside an async def is almost always a callback handed elsewhere
    # (to_thread, run_in_executor, a done-callback), so its body is not on the loop.
    visit_Lambda = visit_FunctionDef  # type: ignore[assignment]

    def _visit_loop(self, node: ast.AST) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_For = _visit_loop
    visit_AsyncFor = _visit_loop
    visit_While = _visit_loop

    # -- checks ---------------------------------------------------------

    def resolve(self, node: ast.AST) -> tuple[str, ...]:
        """dotted(), with `from asyncio import gather` mapped back to asyncio.gather."""
        path = dotted(node)
        if len(path) == 1 and path[0] in self.asyncio_aliases:
            return ("asyncio", self.asyncio_aliases[path[0]])
        return path

    def visit_Call(self, node: ast.Call) -> None:
        path = self.resolve(node.func)
        tail = path[-1] if path else ""
        head = path[0] if path else ""
        kwargs = {kw.arg for kw in node.keywords if kw.arg}
        has_double_star = any(kw.arg is None for kw in node.keywords)

        if self.async_depth:
            self._check_blocking(node, path, tail, head)

        if path[-2:] == ("asyncio", "gather") or (tail == "gather" and head == "asyncio"):
            if "loop" in kwargs:
                self.add("GATHER_LOOP_ARG", node)
            if "return_exceptions" not in kwargs:
                self.add("BARE_GATHER", node)
            self._check_fanout(node)

        if head == "asyncio" and tail == "wait":
            if node.args and self._is_coroutine_expr(node.args[0]):
                self.add("WAIT_ON_COROUTINES", node)

        if head == "asyncio" and tail in {"get_event_loop", "new_event_loop"}:
            self.add("LEGACY_LOOP", node, f"asyncio.{tail}()")
        if tail == "run_until_complete":
            self.add("LEGACY_LOOP", node, "loop.run_until_complete()")

        if tail in HTTPX_CLIENT_NAMES and (head in HTTPX_MODULES or head == tail):
            if "timeout" not in kwargs and not has_double_star:
                self.add("NO_CLIENT_TIMEOUT", node, f"{tail}(...)")
            if self.loop_depth:
                self.add("CLIENT_IN_LOOP", node, f"{tail}(...)")
        if tail == "ClientSession" and self.loop_depth:
            self.add("CLIENT_IN_LOOP", node, "aiohttp.ClientSession(...)")

        self.generic_visit(node)

    def _check_blocking(self, node: ast.Call, path, tail: str, head: str) -> None:
        if len(path) >= 2 and (path[-2], path[-1]) in BLOCKING_ATTR_CALLS:
            self.add(BLOCKING_ATTR_CALLS[(path[-2], path[-1])], node, ".".join(path))
            return
        if head in SYNC_HTTP_ROOTS and tail in SYNC_HTTP_VERBS | {"Session"}:
            self.add("BLOCKING_HTTP", node, ".".join(path))
            return
        if head in HTTPX_MODULES and tail in SYNC_HTTP_VERBS | {"Client", "stream"}:
            self.add("BLOCKING_HTTP", node, f"{head}.{tail}() is the sync API")
            return
        if head == "urllib" and tail in {"urlopen", "urlretrieve"}:
            self.add("BLOCKING_HTTP", node, ".".join(path))
            return
        if tail == "urlopen" and head in {"request", "urlopen"}:
            self.add("BLOCKING_HTTP", node, ".".join(path))
            return
        if isinstance(node.func, ast.Name):
            if node.func.id == "open":
                self.add("BLOCKING_OPEN", node, "open()")
            elif node.func.id == "input":
                self.add("BLOCKING_INPUT", node, "input()")
        if tail in BLOCKING_PATH_METHODS and len(path) >= 1 and id(node) not in self.awaited_calls:
            self.add("BLOCKING_OPEN", node, f".{tail}()")

    def _is_taskgroup_call(self, path: tuple[str, ...]) -> bool:
        """True for `tg.create_task(...)` where tg is a TaskGroup/nursery."""
        if len(path) < 2:
            return False
        receiver = path[-2]
        return receiver in self.taskgroup_names or receiver in {"tg", "task_group", "nursery", "group"}

    def _is_coroutine_expr(self, node: ast.AST) -> bool:
        targets = [node]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            targets = list(node.elts)
        elif isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            targets = [node.elt]
        for t in targets:
            if isinstance(t, ast.Call):
                name = dotted(t.func)
                if name and name[-1] in self.coroutine_names:
                    return True
        return False

    def _check_fanout(self, node: ast.Call) -> None:
        if self.has_limiter:
            return
        for arg in node.args:
            inner = arg.value if isinstance(arg, ast.Starred) else arg
            if isinstance(inner, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
                self.add("UNBOUNDED_FANOUT", node)
                return

    def visit_Expr(self, node: ast.Expr) -> None:
        value = node.value
        if isinstance(value, ast.Await):
            self._check_sequential_await(value)
        if isinstance(value, ast.Call):
            name = self.resolve(value.func)
            tail = name[-1] if name else ""
            if tail in {"create_task", "ensure_future"} and not self._is_taskgroup_call(name):
                self.add("FLOATING_TASK", node, f"{tail}(...) not assigned")
            elif tail in self.coroutine_names and _is_own_call(name):
                self.add("UNAWAITED_COROUTINE", node, f"{tail}(...) result discarded")
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        if isinstance(node.value, ast.Call):
            self.awaited_calls.add(id(node.value))
        self._check_sequential_await(node)
        self.generic_visit(node)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        if self.async_gen_stack and self.async_gen_stack[-1]:
            for child in node.body:
                for sub in _walk_same_function(child):
                    if isinstance(sub, (ast.Yield, ast.YieldFrom)):
                        self.add("ASYNC_GEN_YIELD_IN_CM", sub)
        self.generic_visit(node)

    visit_With = _visit_with
    visit_AsyncWith = _visit_with

    def _check_sequential_await(self, node: ast.Await) -> None:
        if not (self.loop_depth and self.async_depth):
            return
        if isinstance(node.value, ast.Call):
            name = dotted(node.value.func)
            if name and name[-1] in self.coroutine_names:
                self.add("SEQUENTIAL_AWAIT", node, f"await {name[-1]}(...) in a loop")

    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            if handler_catches_cancel(handler) and not contains_reraise(handler):
                self.add("SWALLOWED_CANCEL", handler)
        self.generic_visit(node)

    visit_TryStar = visit_Try  # `except* CancelledError:` swallows just the same (3.11+)


def audit_source(path: str, source: str) -> list[Finding]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [Finding("SYNTAX", path, exc.lineno or 0, exc.offset or 0, f"could not parse: {exc.msg}")]
    auditor = AsyncAuditor(path, source)
    auditor.visit(tree)
    return auditor.findings


def iter_python_files(targets: list[str]):
    for target in targets:
        p = Path(target)
        if not p.exists():
            raise FileNotFoundError(target)
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if any(part in {".venv", "venv", "__pycache__", ".git", "node_modules"} for part in f.parts):
                    continue
                yield f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", default=["."], help="files or directories to scan")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--ignore", default="", help="comma-separated codes to suppress")
    ap.add_argument("--list-codes", action="store_true", help="print every code and what it means, then exit")
    args = ap.parse_args(argv)

    if args.list_codes:
        width = max(len(c) for c in CODES)
        for code, desc in CODES.items():
            print(f"{code:<{width}}  {desc}")
        return 0

    ignored = {c.strip().upper() for c in args.ignore.split(",") if c.strip()}
    missing = [t for t in (args.targets or ["."]) if not Path(t).exists()]
    if missing:
        # A typo'd path in CI must not read as "no findings".
        print(f"audit_async: no such file or directory: {', '.join(missing)}", file=sys.stderr)
        return 2
    findings: list[Finding] = []
    for f in iter_python_files(args.targets or ["."]):
        try:
            source = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "async def" not in source and "asyncio" not in source:
            continue
        findings.extend(x for x in audit_source(str(f), source) if x.code not in ignored)

    if args.format == "json":
        print(json.dumps([asdict(x) for x in findings], indent=2))
    else:
        for x in findings:
            print(x.render())
        counts: dict[str, int] = {}
        for x in findings:
            counts[x.code] = counts.get(x.code, 0) + 1
        if findings:
            print(f"\n{len(findings)} finding(s): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        else:
            print("audit_async: no findings")
    return 1 if findings else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # `audit_async.py src/ | head` closes the pipe early; that is not an error.
        sys.stderr.close()
        sys.exit(0)
