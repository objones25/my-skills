"""`typing_before.py` fixed. Clean under pyright strict and the audit script.

Each fix keeps the runtime check AND gives the checker a type it can prove:

    state["item"] + require(... is not None)  ->  item = expect(state.get("item"), msg)
    item["id"] on a total=False key           ->  expect(item.get("id"), msg)
    parameter annotated `dict`                 ->  Mapping[str, object], or the TypedDict
    variable bound only inside a loop          ->  bind it before the loop
    isinstance() restating the annotation      ->  delete it; check something the type can't say
    overriding a base-class method             ->  @override, so a renamed base fails loudly
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, TypedDict

from negative_space import expect, require

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import override
else:
    try:
        from typing import override
    except ImportError:  # Python < 3.12: the decorator only matters to checkers

        def override(func):  # noqa: ANN001, ANN201
            return func


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


def run(item: Mapping[str, object], run_config: Mapping[str, object]) -> str:
    require(item, "run() on an empty item")
    require("tags" not in run_config or run_config["tags"], "empty tag list")
    return f"{dict(item)}"


def node_tag(state: State, config: Config) -> State:
    item = expect(state.get("item"), "router guarantees an item here")
    item_id = expect(item.get("id"), "item reached tagging without an id")
    kind = item.get("kind", "unknown")
    merged: dict[str, object] = {**item, "seen": True}
    out = run(item, config)
    return {"summary": f"{item_id} {kind} {merged} {out}"}


def last_attempt(n: int) -> int:
    require(n >= 1, f"need at least one attempt, got {n}")
    attempt = -1
    for attempt in range(n):
        if attempt > 3:
            break
    require(0 <= attempt < n, f"loop variable escaped its range: {attempt}")
    return attempt


def describe(node: ast.stmt) -> str:
    require(node.lineno >= 1, "statement without a source line")
    name = type(node).__name__
    require(hasattr(ast, name), f"{name} is not an ast node type")
    return name


class Visitor(ast.NodeVisitor):
    @override
    def visit_Assert(self, node: ast.Assert) -> None:
        self.generic_visit(node)
