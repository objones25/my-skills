from __future__ import annotations

from pathlib import Path

import check_layers as cl


def make_pkg(write, orders_src: str) -> None:
    write("shop/__init__.py")
    write("shop/domain/__init__.py")
    write("shop/adapters/__init__.py")
    write("shop/adapters/memory.py", "class InMemoryInventory: ...\n")
    write("shop/domain/orders.py", orders_src)


def test_clean_domain_passes(write, tmp_path: Path) -> None:
    make_pkg(write, "from dataclasses import dataclass\n")
    assert cl.main([str(tmp_path), "--inner", "shop.domain", "--forbid", "shop.adapters", "--forbid-infra"]) == 0


def test_relative_absolute_and_function_level_imports_are_caught(write, tmp_path: Path) -> None:
    make_pkg(write, """
from ..adapters.memory import InMemoryInventory
import shop.adapters
def later():
    import sqlalchemy
""")
    violations, scanned = cl.check(tmp_path, "shop.domain", ["shop.adapters", "sqlalchemy"])
    assert scanned == 2
    assert sorted((v.line, v.rule) for v in violations) == [
        (1, "shop.adapters"), (2, "shop.adapters"), (4, "sqlalchemy")]


def test_from_package_import_submodule_is_caught(write, tmp_path: Path) -> None:
    make_pkg(write, "from shop import adapters\n")
    violations, _ = cl.check(tmp_path, "shop.domain", ["shop.adapters"])
    assert [v.imported for v in violations] == ["shop.adapters"]


def test_type_checking_imports_are_marked(write, tmp_path: Path) -> None:
    make_pkg(write, """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from shop.adapters.memory import InMemoryInventory
""")
    (v,) = cl.check(tmp_path, "shop.domain", ["shop.adapters"])[0]
    assert v.type_checking_only


def test_prefix_does_not_match_a_longer_name(write, tmp_path: Path) -> None:
    make_pkg(write, "import requests_toolbelt\n")
    assert cl.check(tmp_path, "shop.domain", ["requests"])[0] == []


def test_missing_inner_package_is_an_argument_error(write, tmp_path: Path) -> None:
    make_pkg(write, "")
    assert cl.main([str(tmp_path), "--inner", "shop.nope", "--forbid", "x"]) == 2


def test_shipped_example_is_clean() -> None:
    root = Path(__file__).resolve().parents[1] / "examples" / "ports_adapters"
    assert cl.main([str(root), "--inner", "shop.domain", "--forbid", "shop.adapters,shop.main",
                    "--forbid-infra"]) == 0
