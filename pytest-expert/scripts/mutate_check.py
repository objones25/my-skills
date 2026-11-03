#!/usr/bin/env python3
"""Targeted mutation check: prove that a test command fails when the code is wrong.

Stdlib only. Applies one small AST mutation at a time to ONE source file (optionally
one function), runs your test command, and reports which mutants survived. A
survivor is a change in behavior that no test noticed.

Usage:
    python mutate_check.py src/wallet/money.py -- pytest -x -q -p no:randomly --no-cov tests/unit
    python mutate_check.py src/wallet/money.py --function transfer_fee_cents -- pytest -x -q tests/unit
    python mutate_check.py src/pkg/mod.py --format json -- pytest -x -q

The file is edited in place and always restored (a .mutate_check.bak copy exists
while it runs; if one is left behind after a crash, the script refuses to start
until you restore it). Each run gets a fresh PYTHONPYCACHEPREFIX so a stale .pyc
cannot make a mutant look like it survived.

Exit status: 0 if no mutant survived (or survivors <= --max-survivors), 1 otherwise,
2 for usage errors or a failing baseline.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

COMPARE_SWAPS = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}
BINOP_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
    ast.Div: ast.Mult, ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult,
}
BOOLOP_SWAPS = {ast.And: ast.Or, ast.Or: ast.And}
NAME_SWAPS = {"max": "min", "min": "max", "any": "all", "all": "any"}
SYMBOLS = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
    ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in", ast.Add: "+",
    ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//", ast.Div: "/", ast.Mod: "%", ast.Pow: "**",
    ast.And: "and", ast.Or: "or",
}


@dataclass
class Mutant:
    index: int
    line: int
    description: str
    status: str = "pending"
    seconds: float = 0.0


def is_docstring(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    grand = parents.get(parent) if parent else None
    return (
        isinstance(parent, ast.Expr)
        and isinstance(grand, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and grand.body and grand.body[0] is parent
    )


class SiteFinder(ast.NodeVisitor):
    """Collect mutation sites as (node path, kind, description)."""

    def __init__(self, function: str | None) -> None:
        self.function = function
        self.qualname: list[str] = []
        self.sites: list[tuple[int, int, str, str]] = []
        self.parents: dict[ast.AST, ast.AST] = {}

    def in_scope(self) -> bool:
        if self.function is None:
            return True
        joined = ".".join(self.qualname)
        return joined == self.function or joined.startswith(self.function + ".") or (self.qualname and self.qualname[-1] == self.function and "." not in self.function)

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
        super().generic_visit(node)

    def _scoped(self, node) -> None:
        self.qualname.append(node.name)
        self.generic_visit(node)
        self.qualname.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def add(self, node: ast.AST, kind: str, description: str) -> None:
        if self.in_scope() and self.qualname:
            self.sites.append((id(node), getattr(node, "lineno", 0), kind, description))

    def visit_Compare(self, node: ast.Compare) -> None:
        for i, op in enumerate(node.ops):
            if type(op) in COMPARE_SWAPS:
                self.add(node, f"cmp:{i}", f"{SYMBOLS[type(op)]} -> {SYMBOLS[COMPARE_SWAPS[type(op)]]}")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if type(node.op) in BINOP_SWAPS:
            self.add(node, "binop", f"{SYMBOLS[type(node.op)]} -> {SYMBOLS[BINOP_SWAPS[type(node.op)]]}")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if type(node.op) in BINOP_SWAPS:
            self.add(node, "augassign", f"{SYMBOLS[type(node.op)]}= -> {SYMBOLS[BINOP_SWAPS[type(node.op)]]}=")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.add(node, "boolop", f"{SYMBOLS[type(node.op)]} -> {SYMBOLS[BOOLOP_SWAPS[type(node.op)]]}")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not):
            self.add(node, "not", "drop `not`")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool):
            self.add(node, "bool", f"{node.value} -> {not node.value}")
        elif isinstance(node.value, int):
            self.add(node, "int", f"{node.value} -> {node.value + 1}")

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
    originals = {id(n): n for n in ast.walk(tree)}
    clone = copy.deepcopy(tree)
    for orig, new in zip(ast.walk(tree), ast.walk(clone), strict=True):
        if id(orig) != target_id:
            continue
        if kind.startswith("cmp:"):
            i = int(kind.split(":")[1])
            new.ops[i] = COMPARE_SWAPS[type(new.ops[i])]()
        elif kind in {"binop", "augassign"}:
            new.op = BINOP_SWAPS[type(new.op)]()
        elif kind == "boolop":
            new.op = BOOLOP_SWAPS[type(new.op)]()
        elif kind == "not":
            return _replace(clone, new, new.operand)
        elif kind == "bool":
            new.value = not new.value
        elif kind == "int":
            new.value = new.value + 1
        elif kind == "name":
            new.func.id = NAME_SWAPS[new.func.id]
        elif kind == "return":
            new.value = ast.Constant(value=None)
        break
    del originals
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


def run(command: list[str], timeout: float) -> tuple[str, float]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="mutate-pyc-") as cache:
        env["PYTHONPYCACHEPREFIX"] = cache
        start = time.monotonic()
        try:
            proc = subprocess.run(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return "timeout", time.monotonic() - start
    elapsed = time.monotonic() - start
    return ("passed" if proc.returncode == 0 else "failed"), elapsed


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: mutate_check.py SOURCE [--function NAME] -- TEST COMMAND...", file=sys.stderr)
        return 2
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--function", help="only mutate inside this function (name or Class.method)")
    parser.add_argument("--max-survivors", type=int, default=0)
    parser.add_argument("--timeout-factor", type=float, default=5.0, help="mutant timeout as a multiple of the baseline")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command or not args.source.is_file():
        print("mutate_check: need an existing SOURCE file and a test command after --", file=sys.stderr)
        return 2

    source_path = args.source
    backup = source_path.with_name(source_path.name + ".mutate_check.bak")
    if backup.exists():
        print(f"mutate_check: {backup} exists from an interrupted run; restore it first", file=sys.stderr)
        return 2

    original = source_path.read_text(encoding="utf-8")
    tree = ast.parse(original)
    finder = SiteFinder(args.function)
    finder.visit(tree)
    sites = [s for s in finder.sites if not (s[2] in {"int", "bool"} and _in_docstring_or_annotation(tree, s[0], finder.parents))]
    if not sites:
        print("mutate_check: no mutation sites found (check --function)", file=sys.stderr)
        return 2

    status, baseline = run(command, timeout=600)
    if status != "passed":
        print(f"mutate_check: baseline test command did not pass ({status}); fix the suite first", file=sys.stderr)
        return 2

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
            result, mutant.seconds = run(command, timeout=max(10.0, baseline * args.timeout_factor))
            mutant.status = {"failed": "killed", "timeout": "killed (timeout)", "passed": "SURVIVED"}[result]
            mutants.append(mutant)
            if args.format == "text":
                print(f"  #{index:<3} line {line:<4} {description:<32} {mutant.status}", flush=True)
    finally:
        restore()
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    survivors = [m for m in mutants if m.status == "SURVIVED"]
    killed = len(mutants) - len(survivors)
    score = 100.0 * killed / len(mutants)
    if args.format == "json":
        print(json.dumps({"source": str(source_path), "score": round(score, 1), "mutants": [asdict(m) for m in mutants]}, indent=2))
    else:
        print(f"\n{killed}/{len(mutants)} mutants killed ({score:.0f}%), baseline {baseline:.1f}s")
        for m in survivors:
            print(f"SURVIVED line {m.line}: {m.description}  <- no test fails when this changes")
    return 0 if len(survivors) <= args.max_survivors else 1


def _in_docstring_or_annotation(tree: ast.Module, node_id: int, parents: dict) -> bool:
    node = next((n for n in ast.walk(tree) if id(n) == node_id), None)
    while node is not None:
        parent = parents.get(node)
        if isinstance(parent, (ast.arg, ast.AnnAssign)) and getattr(parent, "annotation", None) is node:
            return True
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and parent.returns is node:
            return True
        node = parent
    return False


if __name__ == "__main__":
    raise SystemExit(main())
