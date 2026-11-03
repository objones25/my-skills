#!/usr/bin/env python3
"""Create a new skill folder from the templates, with the files its tiers require.

Usage: new_skill.py NAME [--dest DIR] [--tier tooling] [--tier knowledge]
Exit codes: 0 created, 2 invalid name or the folder already exists.
Every tier gets the core files. Fill in each TODO before release.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "assets" / "templates"
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
CORE = {"SKILL.md": "SKILL.md", "CHANGELOG.md": "CHANGELOG.md", "evals.json": "evals/evals.json",
        "trigger_evals.json": "evals/trigger_evals.json"}
TIERS = {"tooling": {"test_skill_structure.py": "tests/test_skill_structure.py"},
         "knowledge": {"sources.md": "references/sources.md", "AUDIT.md": "AUDIT.md"}}
TEST_COMMAND = '  test-command: "uv run --no-project --with pytest python -m pytest tests"\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--dest", type=Path, default=Path.cwd())
    parser.add_argument("--tier", action="append", choices=sorted(TIERS), default=[])
    args = parser.parse_args()
    if not NAME_RE.match(args.name) or len(args.name) > 64:
        print(f"invalid name {args.name!r}: use lowercase letters, digits and single hyphens, 64 at most", file=sys.stderr)
        return 2
    skill = args.dest / args.name
    if skill.exists():
        print(f"{skill} already exists; not overwriting it", file=sys.stderr)
        return 2
    files = dict(CORE)
    for tier in args.tier:
        files.update(TIERS[tier])
    for template, target in files.items():
        text = (TEMPLATES / template).read_text(encoding="utf-8").replace("{{name}}", args.name)
        if target == "SKILL.md":
            text = text.replace("{{metadata}}", TEST_COMMAND if "tooling" in args.tier else "")
        path = skill / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"created {skill}: {len(files)} files. Next: replace each TODO, then run check_skill.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
