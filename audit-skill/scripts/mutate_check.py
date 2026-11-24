#!/usr/bin/env python3
"""Targeted mutation check: prove that a test command fails when the code is wrong.

Copied from pytest-expert/scripts/mutate_check.py (2026-11-20, resynced 2026-11-24) so audit-skill stays
self-contained. audit-skill runs it only inside the run's workspace copy of a skill.

Stdlib only. Applies one small AST mutation at a time to ONE source file (optionally
one function), runs your test command, and reports which mutants survived. A
survivor is a change in behavior that no test noticed.

Sites, by kind name (use the names with --skip-kinds / --only-kinds):
    cmp        comparison operators          < <= > >= == != is is-not in not-in
    binop      arithmetic operators          + - * // / % **     (binary form)
    augassign  the same, augmented           += -= *= //= /= %= **=
    boolop     and <-> or
    not        `not x` -> `x`
    bool       True <-> False
    int        n -> n + 1
    float      c -> c + 1.0  (skipped when that changes nothing, e.g. inf, 1e300)
    name       builtin calls max <-> min, any <-> all
    return     `return value` -> `return None`
    matmul     @ -> *  (binary and @=)
    shift      << <-> >>  (binary and augmented)
    bitop      | -> &, & -> |, ^ -> &  (binary and augmented)
    neg        unary minus: `-x` -> `x`
`|` is mutated wherever it is not in an annotation, except when an operand is the constant
None (that can only be a type union). Other runtime type unions such as
`isinstance(x, int | str)` become `int & str`, which raises TypeError: a "kill" that says
little about your tests. Set and dict `|` / `|=` cannot be told apart from integer bitwise
or, so they are mutated too. Drop the bitop kind if that noise outweighs the signal.
Only code inside a function or class body is mutated. Never mutated: docstrings (strings are
not a site kind), annotations, bare constant statements, module-level code. Default arguments
and decorators run at import time and are mutated like any other code; a mutant that breaks
the import shows up as "errored", not as a kill (see below).

Usage:
    python mutate_check.py src/wallet/money.py -- pytest -x -q -p no:randomly --no-cov tests/unit
    python mutate_check.py src/wallet/money.py --function transfer_fee_cents -- pytest -x -q tests/unit
    python mutate_check.py src/pkg/mod.py --format json -- pytest -x -q

--function NAME: a bare name matches any function or class with that name at any depth and
everything nested inside it (closures included); a dotted name (Class.method) is matched as a
path from the top of the module and also includes everything nested inside.

How each mutant run is classified (pytest exit codes; change them with --kill-codes):
    exit 0                  SURVIVED: the tests did not notice the change.
    exit in --kill-codes    killed (default 1: pytest "tests failed").
    timeout                 "killed (timeout)": counts as a kill but is labelled and counted
                            separately, because a slow or loaded machine also causes it.
    any other exit          "errored (exit N)": e.g. pytest exit 2 when the mutant breaks the
                            import or collection, or a crash by signal (negative N). The tests
                            did not catch a behaviour change, they could not run, so these are
                            excluded from the score and reported on their own line.
Which test killed a mutant: the test command's stdout+stderr is captured (only the last 256 KiB
is kept) and, for a plain "killed" mutant, failing test ids are read from pytest's short summary
(`FAILED path::test - msg`, `ERROR path::test`; pytest 9.2.0 prints these by default, with -q
and with -x -q, but not with `-p no:terminal` or a truncated tail) and from unittest's
`FAIL: test (mod.Class.test)` lines. The text report shows the first 3 ("by a, b (+N more)");
JSON has a `killed_by` list per mutant, empty when nothing was recognised. The output never
affects the classification above. --show-output prints the last 10 lines of the output of each
errored mutant (text) or adds an `output_tail` string to every mutant (json).
--skip-kinds / --only-kinds (comma-separated kind names, mutually exclusive) choose the kinds.

The score is killed / (killed + survived). If nothing could be evaluated the exit status is 2.

The mutant timeout is max(--timeout-floor, baseline * --timeout-factor). The floor defaults to
2 s, so a --timeout-factor that gives less than 2 s has no effect unless you lower the floor.
The baseline test command runs twice and must pass both times (a flaky suite cannot be
scored); the slower run sets the timeout. The test command runs in its own process group, and
the whole group is killed on timeout, on SIGINT/SIGTERM and when it exits (POSIX).

The file is edited in place and always restored (a .mutate_check.bak copy exists
while it runs; if one is left behind after a crash, the script refuses to start
until you restore it). Test runs get PYTHONDONTWRITEBYTECODE=1, so no mutant is ever cached as a .pyc
and a stale one cannot make a later mutant look like it survived; the existing cache for
pytest and the stdlib is still read, so each run does not recompile them.

Exit status: 0 if no mutant survived (or survivors <= --max-survivors), 1 otherwise, 130 after
SIGINT/SIGTERM (the source is restored first), 2 when the check could not be done: usage
errors, a source that does not read or parse, a test command that cannot be started, a baseline
that fails or is flaky, no mutation sites, or no mutant that could be evaluated.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, cast

COMPARE_SWAPS = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}
BINOP_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
    ast.Div: ast.Mult, ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult,
}
MATMUL_SWAPS = {ast.MatMult: ast.Mult}
SHIFT_SWAPS = {ast.LShift: ast.RShift, ast.RShift: ast.LShift}
BIT_SWAPS = {ast.BitOr: ast.BitAnd, ast.BitAnd: ast.BitOr, ast.BitXor: ast.BitAnd}
# operator class -> (site kind, replacement class); the first seven keep their original kinds
OP_SWAPS: dict[type, tuple[str | None, type]] = {
    **{op: (None, new) for op, new in BINOP_SWAPS.items()},  # None: "binop" or "augassign"
    **{op: ("matmul", new) for op, new in MATMUL_SWAPS.items()},
    **{op: ("shift", new) for op, new in SHIFT_SWAPS.items()},
    **{op: ("bitop", new) for op, new in BIT_SWAPS.items()},
}
KIND_NAMES = ("cmp", "binop", "augassign", "boolop", "not", "bool", "int", "float", "name", "return",
              "matmul", "shift", "bitop", "neg")
BOOLOP_SWAPS = {ast.And: ast.Or, ast.Or: ast.And}
NAME_SWAPS = {"max": "min", "min": "max", "any": "all", "all": "any"}
SYMBOLS = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
    ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in", ast.Add: "+",
    ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//", ast.Div: "/", ast.Mod: "%", ast.Pow: "**",
    ast.And: "and", ast.Or: "or", ast.MatMult: "@", ast.LShift: "<<", ast.RShift: ">>",
    ast.BitOr: "|", ast.BitAnd: "&", ast.BitXor: "^",
}
MAX_IDS_SHOWN = 3
OUTPUT_TAIL_BYTES = 256 * 1024
OUTPUT_LINES_SHOWN = 10


BASELINE_TIMEOUT_S = 600.0
MUTANT_TIMEOUT_FLOOR_S = 2.0  # infinite-loop mutants stop after max(floor, baseline * factor)


@dataclass
class Mutant:
    index: int
    line: int
    description: str
    status: str = "pending"
    seconds: float = 0.0
    exit_code: int | None = None
    killed_by: list[str] = dataclass_field(default_factory=list)
    output_tail: str = ""


@dataclass(frozen=True)
class RunResult:
    status: str  # "passed" | "failed" | "timeout"
    returncode: int | None
    seconds: float
    output: str = ""  # the last OUTPUT_TAIL_BYTES of stdout+stderr; never used for classification


class SiteFinder(ast.NodeVisitor):
    """Collect mutation sites as (node id, line, kind, description)."""

    def __init__(self, function: str | None) -> None:
        self.function = function
        self.qualname: list[str] = []
        self.sites: list[tuple[int, int, str, str]] = []
        self.parents: dict[ast.AST, ast.AST] = {}

    def in_scope(self) -> bool:
        if self.function is None:
            return True
        if "." not in self.function:
            return self.function in self.qualname
        joined = ".".join(self.qualname)
        return joined == self.function or joined.startswith(self.function + ".")

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
        super().generic_visit(node)

    def _scoped(self, node) -> None:
        self.qualname.append(node.name)
        self.generic_visit(node)
        self.qualname.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def _in_annotation(self, node: ast.AST) -> bool:
        child, parent = node, self.parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.arg, ast.AnnAssign)) and parent.annotation is child:
                return True
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and parent.returns is child:
                return True
            child, parent = parent, self.parents.get(parent)
        return False

    def add(self, node: ast.AST, kind: str, description: str) -> None:
        if self.in_scope() and self.qualname and not self._in_annotation(node):
            self.sites.append((id(node), getattr(node, "lineno", 0), kind, description))

    def visit_Compare(self, node: ast.Compare) -> None:
        for i, op in enumerate(node.ops):
            if type(op) in COMPARE_SWAPS:
                self.add(node, f"cmp:{i}", f"{SYMBOLS[type(op)]} -> {SYMBOLS[COMPARE_SWAPS[type(op)]]}")
        self.generic_visit(node)

    def _op_site(self, node: ast.BinOp | ast.AugAssign, plain_kind: str) -> None:
        entry = OP_SWAPS.get(type(node.op))
        if entry is None:
            return
        kind, new_op = entry
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr) and any(
            isinstance(side, ast.Constant) and side.value is None for side in (node.left, node.right)
        ):
            return  # `X | None` can only be a type union
        suffix = "=" if isinstance(node, ast.AugAssign) else ""
        self.add(node, kind or plain_kind, f"{SYMBOLS[type(node.op)]}{suffix} -> {SYMBOLS[new_op]}{suffix}")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._op_site(node, "binop")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._op_site(node, "augassign")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.add(node, "boolop", f"{SYMBOLS[type(node.op)]} -> {SYMBOLS[BOOLOP_SWAPS[type(node.op)]]}")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            self.add(node, "not", "drop `not`")
        elif isinstance(node.op, ast.USub):
            self.add(node, "neg", "drop unary minus")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(self.parents.get(node), ast.Expr):
            return  # a bare constant statement has no effect, so any mutant of it is equivalent
        if isinstance(node.value, bool):
            self.add(node, "bool", f"{node.value} -> {not node.value}")
        elif isinstance(node.value, int):
            self.add(node, "int", f"{node.value} -> {node.value + 1}")
        elif isinstance(node.value, float) and math.isfinite(node.value) and node.value + 1.0 != node.value:
            self.add(node, "float", f"{node.value!r} -> {node.value + 1.0!r}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in NAME_SWAPS:
            self.add(node, "name", f"{node.func.id}() -> {NAME_SWAPS[node.func.id]}()")
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and not (isinstance(node.value, ast.Constant) and node.value.value is None):
            self.add(node, "return", "return value -> return None")
        self.generic_visit(node)


def mutate(tree: ast.Module, target_id: int, kind: str) -> ast.Module:
    """Return a deep copy of tree with exactly one site mutated."""
    clone = copy.deepcopy(tree)
    for orig, new in zip(ast.walk(tree), ast.walk(clone), strict=True):
        if id(orig) != target_id:
            continue
        # The site kind says which node type this is; ast.AST has no typed fields to check.
        new = cast(Any, new)
        if kind.startswith("cmp:"):
            i = int(kind.split(":")[1])
            new.ops[i] = COMPARE_SWAPS[type(new.ops[i])]()
        elif kind in {"binop", "augassign", "matmul", "shift", "bitop"}:
            new.op = OP_SWAPS[type(new.op)][1]()
        elif kind == "boolop":
            new.op = BOOLOP_SWAPS[type(new.op)]()
        elif kind in {"not", "neg"}:
            return _replace(clone, new, new.operand)
        elif kind == "bool":
            new.value = not new.value
        elif kind in {"int", "float"}:
            new.value = new.value + (1 if kind == "int" else 1.0)
        elif kind == "name":
            new.func.id = NAME_SWAPS[new.func.id]
        elif kind == "return":
            new.value = ast.Constant(value=None)
        break
    return clone


def _replace(tree: ast.Module, old: ast.AST, replacement: ast.AST) -> ast.Module:
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if value is old:
                setattr(parent, field, replacement)
                return tree
            if isinstance(value, list) and old in value:
                value[value.index(old)] = replacement
                return tree
    return tree


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """Kill the test command and everything it started (it leads its own process group)."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass  # the group is already gone (macOS reports EPERM for a group of only zombies)


def _drain(fd: int, sink: bytearray) -> None:
    """Read the pipe to EOF so the test command never blocks on a full pipe; keep only the tail."""
    while chunk := os.read(fd, 65536):
        sink += chunk
        if len(sink) > 2 * OUTPUT_TAIL_BYTES:
            del sink[:-OUTPUT_TAIL_BYTES]


def run(command: list[str], timeout: float) -> RunResult:
    """Run the test command; raises OSError if it cannot be started."""
    # Read the existing bytecode cache but never write one: a fresh, empty PYTHONPYCACHEPREFIX
    # per run made every run recompile pytest and the stdlib (about 3x slower per run).
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    start = time.monotonic()
    proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    if proc.stdout is None:
        raise RuntimeError("Popen(stdout=PIPE) returned no stdout")
    sink = bytearray()
    reader = threading.Thread(target=_drain, args=(proc.stdout.fileno(), sink), daemon=True)
    reader.start()
    try:
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status, code = "timeout", None
        else:
            status, code = ("passed" if returncode == 0 else "failed"), returncode
        seconds = time.monotonic() - start
    finally:
        _kill_group(proc)  # also reached on SIGINT/SIGTERM, which raise SystemExit here
        proc.wait()
        reader.join(timeout=5)  # EOF arrives once the whole group is dead
        proc.stdout.close()
    output = bytes(sink[-OUTPUT_TAIL_BYTES:]).decode("utf-8", errors="replace")
    return RunResult(status, code, seconds, output)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PYTEST_ID = re.compile(r"(?:FAILED|ERROR) (\S.*?)(?: - .*)?")
_UNITTEST_ID = re.compile(r"(?:FAIL|ERROR): (\w+ \([\w.]+\))")


def failing_test_ids(output: str) -> list[str]:
    """Failing test ids found in a test command's output; [] when none are recognised.

    pytest: the `FAILED id - msg` / `ERROR id - msg` lines after "short test summary info"
    (pytest 9.2.0 prints them by default, with -q, and with -x -q). unittest: `FAIL: test (mod.Class.test)`.
    """
    ids: list[str] = []
    in_summary = False
    for line in _ANSI.sub("", output).splitlines():
        if line.startswith("=") and "short test summary info" in line:
            in_summary = True
            continue
        if in_summary:
            found = _PYTEST_ID.fullmatch(line)
        else:
            found = _UNITTEST_ID.fullmatch(line)
        if found and found.group(1) not in ids:
            ids.append(found.group(1))
    return ids


def _kill_label(mutant: Mutant) -> str:
    if mutant.status != "killed" or not mutant.killed_by:
        return ""
    shown = ", ".join(mutant.killed_by[:MAX_IDS_SHOWN])
    rest = len(mutant.killed_by) - MAX_IDS_SHOWN
    return f"   by {shown}" + (f" (+{rest} more)" if rest > 0 else "")


def _mutant_json(mutant: Mutant, show_output: bool) -> dict[str, Any]:
    data = asdict(mutant)
    if not show_output:
        del data["output_tail"]
    return data


def classify(result: RunResult, kill_codes: frozenset[int]) -> str:
    if result.status == "passed":
        return "SURVIVED"
    if result.status == "timeout":
        return "killed (timeout)"
    if result.returncode in kill_codes:
        return "killed"
    return f"errored (exit {result.returncode})"


def _kill_codes(text: str) -> frozenset[int]:
    try:
        return frozenset(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from None


def _kind_list(text: str) -> frozenset[str]:
    names = [part.strip() for part in text.split(",")]
    unknown = [n for n in names if n not in KIND_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown kind {', '.join(map(repr, unknown))}; known kinds: {', '.join(KIND_NAMES)}")
    return frozenset(names)


def _fail(message: str) -> int:
    print(f"mutate_check: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else None
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--function", help="only mutate inside this function or class (bare name, or Class.method)")
    parser.add_argument("--max-survivors", type=int, default=0)
    parser.add_argument("--timeout-factor", type=float, default=5.0, help="mutant timeout as a multiple of the baseline")
    parser.add_argument("--timeout-floor", type=float, default=MUTANT_TIMEOUT_FLOOR_S,
                        help="the mutant timeout is at least this many seconds (default 2)")
    parser.add_argument("--kill-codes", type=_kill_codes, default=frozenset({1}),
                        help="comma-separated test-command exit codes that count as a kill (default 1, pytest's 'tests failed')")
    kinds = parser.add_mutually_exclusive_group()
    kinds.add_argument("--skip-kinds", type=_kind_list, default=frozenset(), metavar="KIND[,KIND...]",
                       help="do not mutate these kinds (known: " + ", ".join(KIND_NAMES) + ")")
    kinds.add_argument("--only-kinds", type=_kind_list, default=None, metavar="KIND[,KIND...]",
                       help="mutate only these kinds")
    parser.add_argument("--show-output", action="store_true",
                        help="text format: print the last lines of the test command's output for each errored mutant; "
                             "json format: add an output_tail field to every mutant")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    if split is None and not {"-h", "--help"} & set(argv):
        print("usage: mutate_check.py SOURCE [--function NAME] -- TEST COMMAND...", file=sys.stderr)
        return 2
    args = parser.parse_args(argv if split is None else argv[:split])
    command = [] if split is None else argv[split + 1:]
    if not command or not args.source.is_file():
        return _fail("need an existing SOURCE file and a test command after --")

    source_path = args.source
    backup = source_path.with_name(source_path.name + ".mutate_check.bak")
    if backup.exists():
        return _fail(f"{backup} exists from an interrupted run; restore it first")

    try:
        original = source_path.read_text(encoding="utf-8")
        tree = ast.parse(original)
    except UnicodeDecodeError:
        return _fail(f"{source_path} is not valid UTF-8")
    except SyntaxError as exc:
        return _fail(f"{source_path} does not parse: {exc.msg} (line {exc.lineno})")
    except (OSError, ValueError) as exc:  # ValueError: null bytes in the source
        return _fail(f"cannot read {source_path}: {exc}")
    finder = SiteFinder(args.function)
    finder.visit(tree)
    sites = [
        site for site in finder.sites
        if site[2].split(":")[0] not in args.skip_kinds and (args.only_kinds is None or site[2].split(":")[0] in args.only_kinds)
    ]
    if not sites:
        return _fail("no mutation sites found (check --function, --skip-kinds, --only-kinds)")

    try:
        baselines = [run(command, timeout=BASELINE_TIMEOUT_S) for _ in range(2)]
    except OSError as exc:
        return _fail(f"cannot start the test command {command[0]!r}: {exc.strerror or exc}")
    if baselines[0].status != "passed":
        return _fail(f"baseline test command did not pass ({baselines[0].status}); fix the suite first")
    if baselines[1].status != "passed":
        return _fail("baseline is flaky: it passed once and then did not pass on a second run; fix the suite first")
    baseline = max(b.seconds for b in baselines)
    mutant_timeout = max(args.timeout_floor, baseline * args.timeout_factor)

    shutil.copy2(source_path, backup)

    def restore(*_: object) -> None:
        if backup.exists():
            shutil.copy2(backup, source_path)
            backup.unlink()

    previous = {sig: signal.signal(sig, lambda *_: (restore(), sys.exit(130))) for sig in (signal.SIGINT, signal.SIGTERM)}
    mutants: list[Mutant] = []
    try:
        for index, (node_id, line, kind, description) in enumerate(sites, start=1):
            mutant = Mutant(index, line, description)
            source_path.write_text(ast.unparse(mutate(tree, node_id, kind)) + "\n", encoding="utf-8")
            result = run(command, timeout=mutant_timeout)
            mutant.seconds, mutant.exit_code = result.seconds, result.returncode
            mutant.status = classify(result, args.kill_codes)
            if mutant.status == "killed":
                mutant.killed_by = failing_test_ids(result.output)
            if args.show_output:
                mutant.output_tail = "\n".join(result.output.splitlines()[-OUTPUT_LINES_SHOWN:])
            mutants.append(mutant)
            if args.format == "text":
                print(f"  #{index:<3} line {line:<4} {description:<32} {mutant.status}{_kill_label(mutant)}", flush=True)
                if args.show_output and mutant.status.startswith("errored"):
                    for text_line in mutant.output_tail.splitlines():
                        print(f"        | {text_line}", flush=True)
    except OSError as exc:
        restore()
        return _fail(f"cannot start the test command {command[0]!r}: {exc.strerror or exc}")
    finally:
        restore()
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    survivors = [m for m in mutants if m.status == "SURVIVED"]
    errored = [m for m in mutants if m.status.startswith("errored")]
    timeouts = [m for m in mutants if m.status == "killed (timeout)"]
    killed = len(mutants) - len(survivors) - len(errored)
    if killed + len(survivors) == 0:
        return _fail("no mutant could be evaluated: every run of the test command exited with a code "
                     f"outside --kill-codes {sorted(args.kill_codes)} (use --kill-codes if your command fails with another code)")
    score = 100.0 * killed / (killed + len(survivors))
    if args.format == "json":
        print(json.dumps({
            "source": str(source_path), "score": round(score, 1), "killed": killed, "survived": len(survivors),
            "errored": len(errored), "timeouts": len(timeouts), "baseline_seconds": baseline,
            "timeout_seconds": mutant_timeout, "mutants": [_mutant_json(m, args.show_output) for m in mutants],
        }, indent=2))
    else:
        print(f"\n{killed}/{killed + len(survivors)} mutants killed ({score:.0f}%), baseline {baseline:.1f}s, "
              f"mutant timeout {mutant_timeout:.1f}s")
        if timeouts:
            print(f"{len(timeouts)} of the kills were timeouts (a slow or loaded machine causes these too)")
        if errored:
            print(f"{len(errored)} mutant(s) errored (the test command could not run, e.g. the mutant broke the import); "
                  "excluded from the score")
        for m in survivors:
            print(f"SURVIVED line {m.line}: {m.description}  <- no test fails when this changes")
    return 0 if len(survivors) <= args.max_survivors else 1


if __name__ == "__main__":
    raise SystemExit(main())
