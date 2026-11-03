#!/usr/bin/env python3
"""Static audit of logging and W&B instrumentation against the observability-expert gates.

Pure stdlib (ast + pathlib). Nothing is imported from the code under audit, so it
is safe to run on any repo and fast enough for a pre-commit hook.

Usage:
    python audit_observability.py src/
    python audit_observability.py a.py b.py c.py        # several paths (pre-commit)
    python audit_observability.py src/ --entrypoint src/main.py --entrypoint src/cli.py
    python audit_observability.py src/ --ignore LOG004,WB002
    python audit_observability.py src/ --format json
    python audit_observability.py --list-codes .

Exit code is 1 if any finding is reported, 0 otherwise. That is the gate.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

LOG_METHODS = frozenset(
    {"debug", "info", "warning", "error", "critical", "exception", "log", "warn"}
)
# Secret-looking names are matched word by word (snake_case, camelCase, dotted),
# not by substring: a substring match flags `tokens_per_sec`, `tokenizer`,
# `max_tokens` and `classname`, which are ordinary ML names. Keep in sync with
# assets/obslog/redact.py (this script is stdlib-only and imports nothing).
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
TOKEN_NEIGHBORS = frozenset(
    {"pad", "eos", "bos", "unk", "sep", "cls", "mask", "id", "ids", "count", "len", "type"}
)
WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

# High-confidence credential shapes, used only for literal strings.
SECRET_VALUE_RE = re.compile(
    r"(?i)(\bbearer\s+[A-Za-z0-9._\-]{8,}|\bsk-[A-Za-z0-9._\-]{8,}"
    r"|\bAKIA[0-9A-Z]{12,}|\bghp_[A-Za-z0-9]{20,}|\b[0-9a-f]{40}\b)"
)

CODES: dict[str, str] = {
    "LOG001": "Eager message building (f-string, .format(), % or + concat) inside a log call",
    "LOG002": "Module-level logging.info/debug/... -- that is the root logger",
    "LOG003": "logging.getLogger() with no name returns the root logger",
    "LOG004": "print() used for diagnostics instead of a logger",
    "LOG005": "Logging configured at import time / outside an entry point",
    "LOG006": "Exception handled and logged without a traceback (use .exception or exc_info=True)",
    "LOG007": "Exception swallowed with no log record and no re-raise",
    "LOG008": "Secret-looking value passed to a log or W&B call",
    "LOG009": "logger.warn() is deprecated; use logger.warning()",
    "LOG010": "dictConfig without disable_existing_loggers: False (absent or True)",
    "LOG011": "Logging the exception's str() instead of its traceback",
    "WB001": "wandb.init() without project=",
    "WB002": "wandb.init() without config=",
    "WB003": "wandb.init() with no finish() and not used as a context manager",
    "WB004": "Module-level wandb.log/summary/run.* instead of the run object",
    "WB005": "W&B credential hardcoded in source",
    "WB006": "run.log(..., step=...) -- out-of-order steps are dropped; use define_metric",
}

DEFAULT_ENTRYPOINT_GLOBS = (
    "*/main.py",
    "main.py",
    "*/__main__.py",
    "__main__.py",
    "*/cli.py",
    "cli.py",
    "*/app.py",
    "app.py",
    "*/manage.py",
    "manage.py",
    "*/train.py",
    "train.py",
    "*/conftest.py",
    "conftest.py",
    "*/settings.py",
    "settings.py",
    "*/logging_setup.py",
    "*/setup.py",
    "*/log_config.py",
    "*/logging_config.py",
)

SKIP_DIR_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "dist",
        "site-packages",
    }
)


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    line: int
    col: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.code} {self.message}"


def attr_chain(node: ast.AST) -> str:
    """Dotted source text of an attribute/name chain, e.g. 'logging.config.dictConfig'."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        parts.append("()")
    return ".".join(reversed(parts))


def looks_secret(name: str) -> bool:
    words = [w.lower() for w in WORD_RE.findall(name)]
    for i, word in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else ""
        prev = words[i - 1] if i > 0 else ""
        if (word, nxt) in SECRET_BIGRAMS:
            return True
        if word not in SECRET_WORDS:
            continue
        if word == "token" and (prev in TOKEN_NEIGHBORS or nxt in TOKEN_NEIGHBORS):
            continue
        return True
    return False


def is_logger_call(func: ast.AST) -> tuple[bool, str]:
    """(is a logger.<method> call, method name).

    Matches any receiver whose name mentions 'log' -- logger, log, self.logger,
    LOGGER, access_logger -- which is how loggers are named in practice.
    """
    if not isinstance(func, ast.Attribute) or func.attr not in LOG_METHODS:
        return False, ""
    # logging.getLogger(__name__).info(...): the receiver is a call, not a name.
    if isinstance(func.value, ast.Call) and attr_chain(func.value.func).endswith(
        "getLogger"
    ):
        return True, func.attr
    receiver = attr_chain(func.value).split(".")[-1].lower()
    if receiver in ("logging",):
        return False, ""
    if "log" in receiver:
        return True, func.attr
    return False, ""


def is_stdlib_logging_call(func: ast.AST) -> tuple[bool, str]:
    """(is logging.<method>(...) module-level convenience call, method name)."""
    if not isinstance(func, ast.Attribute) or func.attr not in LOG_METHODS:
        return False, ""
    return attr_chain(func.value) == "logging", func.attr


class Auditor(ast.NodeVisitor):
    def __init__(self, path: Path, source: str, is_entrypoint: bool) -> None:
        self.path = path
        self.source = source
        self.is_entrypoint = is_entrypoint
        self.findings: list[Finding] = []
        # Stack of handler-name -> exception-var for enclosing `except` blocks.
        self._handler_stack: list[str | None] = []
        self._func_depth = 0
        self._has_wandb_finish = False
        self._wandb_init_nodes: list[ast.Call] = []
        self._wandb_init_in_with: set[int] = set()

    # -- helpers ---------------------------------------------------------
    def add(self, code: str, node: ast.AST, message: str) -> None:
        self.findings.append(
            Finding(
                code=code,
                path=str(self.path),
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=message,
            )
        )

    # -- structure -------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._handler_stack.append(node.name)
        logged = self._handler_logs(node)
        if not logged and not self._handler_reraises(node):
            self.add(
                "LOG007",
                node,
                "except block neither logs nor re-raises -- the failure is invisible",
            )
        self.generic_visit(node)
        self._handler_stack.pop()

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call) and attr_chain(call.func).endswith(
                "wandb.init"
            ):
                self._wandb_init_in_with.add(id(call))
        self.generic_visit(node)

    visit_AsyncWith = visit_With  # type: ignore[assignment]

    # -- calls -----------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        dotted = attr_chain(node.func)

        is_logger, method = is_logger_call(node.func)
        is_stdlib, stdlib_method = is_stdlib_logging_call(node.func)

        if is_logger or is_stdlib:
            self._check_log_call(node, method or stdlib_method, via_module=is_stdlib)

        if is_stdlib:
            self.add(
                "LOG002",
                node,
                f"logging.{stdlib_method}() logs to the root logger; use "
                "logging.getLogger(__name__)",
            )

        if dotted == "logging.getLogger" and not node.args and not node.keywords:
            self.add("LOG003", node, "getLogger() with no name returns the root logger")

        if dotted in ("print",) and self._looks_diagnostic(node):
            self.add("LOG004", node, "print() has no level, no timestamp, no routing")

        if dotted.endswith("basicConfig") or dotted.endswith("dictConfig"):
            self._check_config_call(node, dotted)

        if dotted.endswith("wandb.init"):
            self._check_wandb_init(node)

        if dotted in ("wandb.log", "wandb.log_artifact") or dotted.startswith(
            "wandb.run."
        ):
            self.add(
                "WB004",
                node,
                f"{dotted}() uses the implicit global run; it raises after "
                "finish() and silently targets the wrong run when several exist. "
                "Hold the object returned by wandb.init().",
            )

        if dotted.endswith("wandb.finish") or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "finish"
        ):
            self._has_wandb_finish = True

        if dotted.endswith("wandb.login"):
            for kw in node.keywords:
                if kw.arg == "key" and isinstance(kw.value, ast.Constant):
                    self.add("WB005", node, "API key hardcoded in wandb.login(key=...)")

        if isinstance(node.func, ast.Attribute) and node.func.attr == "log":
            receiver = attr_chain(node.func.value).split(".")[-1].lower()
            if "run" in receiver or receiver == "wandb":
                for kw in node.keywords:
                    if kw.arg == "step":
                        self.add(
                            "WB006",
                            node,
                            "explicit step= : a step lower than the current one is "
                            "dropped without an exception (a terminal warning at "
                            "most). Log the counter as a metric and set it as the "
                            "x-axis with define_metric().",
                        )

        self._check_secret_args(node, dotted)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # wandb.summary["best"] = x / wandb.summary.update(...): neither is a
        # call to wandb.summary itself, so visit_Call never sees it.
        if attr_chain(node) == "wandb.summary":
            self.add(
                "WB004",
                node,
                "wandb.summary is the implicit global run's summary; use "
                "run.summary on the object returned by wandb.init().",
            )
        self.generic_visit(node)

    # -- assignments (hardcoded keys, handler installs) -------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            name = attr_chain(target)
            if "WANDB_API_KEY" in name.upper() and isinstance(node.value, ast.Constant):
                self.add("WB005", node, "WANDB_API_KEY assigned a literal in source")
        # os.environ["WANDB_API_KEY"] = "..."
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and str(target.slice.value).upper() == "WANDB_API_KEY"
                and isinstance(node.value, ast.Constant)
            ):
                self.add("WB005", node, "WANDB_API_KEY written into os.environ as a literal")
        self.generic_visit(node)

    # -- checks ----------------------------------------------------------
    def _looks_diagnostic(self, node: ast.Call) -> bool:
        """print() in a __main__ guard or a CLI is output; elsewhere it is a log."""
        return not self.is_entrypoint

    def _check_log_call(self, node: ast.Call, method: str, via_module: bool) -> None:
        if method == "warn":
            self.add("LOG009", node, "logger.warn() is deprecated; use .warning()")

        if not node.args:
            return
        # For logger.log(level, msg, ...) the message is the second arg.
        msg = node.args[1] if method == "log" and len(node.args) > 1 else node.args[0]

        if isinstance(msg, ast.JoinedStr):
            self.add(
                "LOG001",
                node,
                "f-string builds the message before the level check; pass "
                "%-style args instead: logger.info('x=%s', x)",
            )
        elif isinstance(msg, ast.BinOp) and isinstance(msg.op, (ast.Mod, ast.Add)):
            op = "%" if isinstance(msg.op, ast.Mod) else "+"
            self.add(
                "LOG001",
                node,
                f"message built with `{op}` at the call site; pass the values as "
                "arguments so formatting is deferred",
            )
        elif (
            isinstance(msg, ast.Call)
            and isinstance(msg.func, ast.Attribute)
            and msg.func.attr == "format"
        ):
            self.add(
                "LOG001",
                node,
                ".format() builds the message before the level check; pass "
                "%-style args instead",
            )

        if self._handler_stack:
            exc_name = self._handler_stack[-1]
            has_exc_info = any(kw.arg == "exc_info" for kw in node.keywords)
            if method in ("error", "critical", "warning") and not has_exc_info:
                self.add(
                    "LOG006",
                    node,
                    f"logging .{method}() inside except without the traceback; "
                    "use logger.exception(...) or exc_info=True",
                )
            if exc_name and self._mentions_only_exception(msg, exc_name):
                self.add(
                    "LOG011",
                    node,
                    f"logging str({exc_name}) discards the traceback -- the message "
                    "is the one part you can always reconstruct, the traceback is not",
                )

    def _mentions_only_exception(self, msg: ast.AST, exc_name: str) -> bool:
        if isinstance(msg, ast.Call) and attr_chain(msg.func) == "str":
            return bool(msg.args) and attr_chain(msg.args[0]) == exc_name
        if isinstance(msg, ast.Name):
            return msg.id == exc_name
        return False

    def _check_config_call(self, node: ast.Call, dotted: str) -> None:
        at_import_time = self._func_depth == 0
        if at_import_time and not self.is_entrypoint:
            self.add(
                "LOG005",
                node,
                f"{dotted}() runs at import time in a non-entry-point module; "
                "importing it silently reconfigures the whole process",
            )
        elif not self.is_entrypoint and dotted.endswith("basicConfig"):
            self.add(
                "LOG005",
                node,
                "library code calling basicConfig() overrides the application's "
                "own configuration",
            )

        if dotted.endswith("dictConfig") and node.args:
            config = node.args[0]
            if isinstance(config, ast.Dict):
                keys = {
                    k.value
                    for k in config.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
                value = next(
                    (
                        v
                        for k, v in zip(config.keys, config.values)
                        if isinstance(k, ast.Constant)
                        and k.value == "disable_existing_loggers"
                    ),
                    None,
                )
                explicit_true = isinstance(value, ast.Constant) and value.value is True
                if "disable_existing_loggers" not in keys or explicit_true:
                    self.add(
                        "LOG010",
                        node,
                        "disable_existing_loggers is True (the default when the "
                        "key is absent), silencing every logger created before "
                        "this call",
                    )

    def _check_wandb_init(self, node: ast.Call) -> None:
        self._wandb_init_nodes.append(node)
        kwargs = {kw.arg for kw in node.keywords}
        # Positional order is (entity, project, ...): a single positional
        # argument is the entity, not the project.
        if "project" not in kwargs and len(node.args) < 2:
            self.add(
                "WB001",
                node,
                "wandb.init() without project= : W&B guesses from the git repo or "
                "script name, else 'uncategorized'",
            )
        if "config" not in kwargs:
            self.add(
                "WB002",
                node,
                "wandb.init() without config= : the run records metrics but not "
                "the hyperparameters that produced them",
            )

    def _check_secret_args(self, node: ast.Call, dotted: str) -> None:
        is_logger, _ = is_logger_call(node.func)
        is_stdlib, _ = is_stdlib_logging_call(node.func)
        is_wandb = dotted.endswith(("wandb.init", "run.log", "wandb.log", "config.update"))
        if not (is_logger or is_stdlib or is_wandb):
            return
        for child in ast.walk(node):
            name = ""
            if isinstance(child, ast.Name):
                name = child.id
            elif isinstance(child, ast.Attribute):
                name = child.attr
            elif isinstance(child, ast.keyword) and child.arg:
                name = child.arg
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                # Literal strings are judged by shape, not by wording, so
                # logger.info("token refreshed") is not a finding but a pasted
                # `sk-...` key is.
                if SECRET_VALUE_RE.search(child.value):
                    self.add(
                        "LOG008",
                        node,
                        "a literal that looks like a credential reaches a log/W&B call",
                    )
                    return
                continue
            if name and looks_secret(name):
                self.add(
                    "LOG008",
                    node,
                    f"secret-looking value `{name}` reaches a log/W&B call; logs and "
                    "run configs are readable by everyone with project access",
                )
                return

    # -- except-block helpers -------------------------------------------
    @staticmethod
    def _handler_logs(node: ast.ExceptHandler) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if is_logger_call(child.func)[0] or is_stdlib_logging_call(child.func)[0]:
                    return True
                if attr_chain(child.func) in ("print", "traceback.print_exc"):
                    return True
        return False

    @staticmethod
    def _handler_reraises(node: ast.ExceptHandler) -> bool:
        # Optional-dependency idiom: `except ImportError: torch = None`. The
        # fallback assignment is the handling, like a `return fallback`.
        if (
            node.type is not None
            and attr_chain(node.type) in ("ImportError", "ModuleNotFoundError")
            and all(isinstance(stmt, (ast.Assign, ast.AnnAssign)) for stmt in node.body)
        ):
            return True
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                return True
            if isinstance(child, ast.Return) and child.value is not None:
                return True  # deliberate fallback value, not a swallow
        return False

    # -- end of module ---------------------------------------------------
    def finalize(self) -> None:
        for call in self._wandb_init_nodes:
            if id(call) in self._wandb_init_in_with:
                continue
            if not self._has_wandb_finish:
                self.add(
                    "WB003",
                    call,
                    "wandb.init() without finish() or `with`: an exception caught "
                    "further up leaves the run open and it is later recorded as "
                    "finished. Use `with wandb.init(...) as run:` so any exception "
                    "inside the block marks the run failed.",
                )


def iter_python_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        p
        for p in root.rglob("*.py")
        if not SKIP_DIR_PARTS.intersection(p.parts)
    )


def is_entrypoint(path: Path, extra_globs: tuple[str, ...]) -> bool:
    text = str(path)
    return any(
        fnmatch.fnmatch(text, pattern)
        for pattern in DEFAULT_ENTRYPOINT_GLOBS + extra_globs
    )


def has_main_guard(tree: ast.Module) -> bool:
    """True if the module has an `if __name__ == "__main__":` block.

    Such a module can be executed directly, which makes it an entry point: it is
    allowed to configure logging and to print to stdout as its actual output.
    """
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(
                isinstance(c, ast.Constant) and c.value == "__main__"
                for c in test.comparators
            )
        ):
            return True
    return False


# `# obs: allow LOG003` (or `# noqa: LOG003`) on the offending line silences one
# code there. Deliberate exceptions should be visible in the diff, not in a
# config file nobody reads.
SUPPRESS_RE = re.compile(r"#\s*(?:obs:\s*allow|noqa:?)\s+([A-Z]{2,4}\d{3}(?:\s*,\s*[A-Z]{2,4}\d{3})*)")


def suppressed_codes(source_lines: list[str], line: int) -> set[str]:
    if not 1 <= line <= len(source_lines):
        return set()
    match = SUPPRESS_RE.search(source_lines[line - 1])
    if not match:
        return set()
    return {code.strip() for code in match.group(1).split(",")}


def audit_file(path: Path, entry_globs: tuple[str, ...]) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [
            Finding(
                code="SYNTAX",
                path=str(path),
                line=exc.lineno or 0,
                col=exc.offset or 0,
                message=f"could not parse: {exc.msg}",
            )
        ]
    entry = is_entrypoint(path, entry_globs) or has_main_guard(tree)
    auditor = Auditor(path, source, entry)
    auditor.visit(tree)
    auditor.finalize()
    lines = source.splitlines()
    return [
        f for f in auditor.findings if f.code not in suppressed_codes(lines, f.line)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", type=Path, nargs="+", metavar="path", help="files or directories to audit"
    )
    parser.add_argument(
        "--entrypoint",
        action="append",
        default=[],
        help="glob for a module allowed to configure logging (repeatable)",
    )
    parser.add_argument("--ignore", default="", help="comma-separated codes to skip")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--list-codes", action="store_true", help="print every code and exit 0"
    )
    args = parser.parse_args(argv)

    if args.list_codes:
        for code, description in CODES.items():
            print(f"{code}  {description}")
        return 0

    ignored = {c.strip().upper() for c in args.ignore.split(",") if c.strip()}
    entry_globs = tuple(args.entrypoint)

    findings: list[Finding] = []
    seen: set[Path] = set()
    for root in args.paths:
        for path in iter_python_files(root):
            if path in seen:
                continue
            seen.add(path)
            findings.extend(
                f for f in audit_file(path, entry_globs) if f.code not in ignored
            )

    findings.sort(key=lambda f: (f.path, f.line, f.code))

    if args.format == "json":
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        for finding in findings:
            print(finding.render())
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding.code] = counts.get(finding.code, 0) + 1
        if findings:
            summary = ", ".join(f"{c}:{n}" for c, n in sorted(counts.items()))
            print(f"\n{len(findings)} finding(s) [{summary}]", file=sys.stderr)
        else:
            print("no findings", file=sys.stderr)

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
