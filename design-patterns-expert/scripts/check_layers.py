#!/usr/bin/env python3
"""Fail when an inner layer imports something it must not. Stdlib only.

The skill says "a layer without an enforced import direction buys nothing". This is the
enforcement: point it at the directory that holds your top-level package, name the inner
package, and list what it may not import (sibling packages or third-party modules).

Usage:
    python scripts/check_layers.py src --inner shop.domain --forbid shop.adapters,shop.main,sqlalchemy
    python scripts/check_layers.py src --inner shop.domain --forbid shop.adapters --forbid-infra

Checks DIRECT imports anywhere in the inner modules, including imports inside functions
and under `if TYPE_CHECKING:` (reported, marked). Transitive chains (domain -> utils ->
sqlalchemy) need import-linter's `forbidden` contract; see references/structural.md.
Exit 0 clean, 1 violations, 2 bad arguments.
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design_smells import INFRA_MODULES


@dataclass(frozen=True)
class Violation:
    module: str
    line: int
    imported: str
    rule: str
    type_checking_only: bool


def module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _type_checking_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
            for inner in ast.walk(node):
                line = getattr(inner, "lineno", None)
                if line is not None:
                    lines.add(line)
    return lines


def imports_of(path: Path, mod: str) -> list[tuple[int, str, bool]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    guarded = _type_checking_lines(tree)
    package = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
    out: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name, node.lineno in guarded) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".")
                base = ".".join(base_parts[: len(base_parts) - node.level + 1])
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            out.append((node.lineno, target, node.lineno in guarded))
            # `from shop import adapters` imports a submodule by name: record that too
            out += [(node.lineno, f"{target}.{a.name}", node.lineno in guarded) for a in node.names]
    return out


def check(root: Path, inner: str, forbidden: list[str]) -> tuple[list[Violation], int]:
    violations: list[Violation] = []
    scanned = 0
    for path in sorted(root.rglob("*.py")):
        mod = module_name(path, root)
        if mod != inner and not mod.startswith(inner + "."):
            continue
        scanned += 1
        seen: set[tuple[int, str]] = set()
        for line, target, tc_only in imports_of(path, mod):
            for rule in forbidden:
                if (target == rule or target.startswith(rule + ".")) and (line, rule) not in seen:
                    seen.add((line, rule))
                    violations.append(Violation(mod, line, target, rule, tc_only))
    return violations, scanned


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Enforce that an inner layer does not import outer ones.")
    ap.add_argument("root", help="directory that contains the top-level package (a sys.path entry)")
    ap.add_argument("--inner", required=True, help="dotted name of the inner package, e.g. shop.domain")
    ap.add_argument("--forbid", default="", help="comma-separated dotted prefixes the inner layer may not import")
    ap.add_argument("--forbid-infra", action="store_true", help="also forbid the scanner's INFRA_MODULES list")
    args = ap.parse_args(argv)

    root = Path(args.root)
    forbidden = [f.strip() for f in args.forbid.split(",") if f.strip()]
    if args.forbid_infra:
        forbidden += sorted(INFRA_MODULES)
    if not root.is_dir() or not forbidden:
        print("error: root must be a directory and at least one forbidden module is required", file=sys.stderr)
        return 2
    violations, scanned = check(root, args.inner, forbidden)
    if scanned == 0:
        print(f"error: no modules named {args.inner!r} under {root}", file=sys.stderr)
        return 2
    for v in violations:
        note = "  (under TYPE_CHECKING only)" if v.type_checking_only else ""
        print(f"{v.module}:{v.line}: imports {v.imported} (forbidden: {v.rule}){note}")
    print(f"{scanned} module(s) in {args.inner} checked; {len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
