"""Checks the runtime accepts but the type checker rejects. Pyright strict fails
on this file on purpose; `scripts/test_types.py` asserts each rule ID below fires,
and that `typing_after.py` (the fixed version) is clean.

The shape mirrors a LangGraph-style node: a `State` TypedDict whose keys are
filled in over several steps, so most of them are optional (`total=False`).

Rule IDs reproduced (Pylance shows the same IDs as pyright):
    reportTypedDictNotRequiredAccess  state["item"] on a total=False key
    reportOptionalSubscript           item["id"] after require(item is not None)
    reportOptionalMemberAccess        item.get(...) on Item | None
    reportGeneralTypeIssues           {**item} where item may be None
    reportArgumentType                TypedDict passed where `dict` is declared
    reportPossiblyUnbound             loop variable used after a loop that may not run
                                      (JSON output names it reportPossiblyUnboundVariable)
    reportUnnecessaryIsInstance       isinstance() that restates the annotation
    reportImplicitOverride            subclass method without @override
"""

from __future__ import annotations

import ast
from typing import TypedDict

from negative_space import require


class Item(TypedDict, total=False):
    id: str
    sha256: str
    kind: str


class Config(TypedDict, total=False):
    """Stand-in for langchain_core's RunnableConfig, also a total=False TypedDict."""

    tags: list[str]
    run_name: str


class State(TypedDict, total=False):
    item: Item | None
    summary: str


def run(item: dict, run_config: dict) -> str:  # noqa: ARG001
    return str(item)


def node_tag(state: State, config: Config) -> State:
    item = state["item"]
    require(item is not None, "router guarantees an item here")
    item_id = item["id"]
    kind = item.get("kind", "unknown")
    merged = {**item, "seen": True}
    out = run(item, config)
    return {"summary": f"{item_id} {kind} {merged} {out}"}


def last_attempt(n: int) -> int:
    for attempt in range(n):
        if attempt > 3:
            break
    return attempt


def describe(node: ast.stmt) -> str:
    require(isinstance(node, ast.stmt), "expected a statement")
    return type(node).__name__


class Visitor(ast.NodeVisitor):
    def visit_Assert(self, node: ast.Assert) -> None:
        self.generic_visit(node)
