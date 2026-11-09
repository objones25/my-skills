#!/usr/bin/env python3
"""Start an audit: create the run directory, copy the skill into a workspace, record meta.json.

Usage:
    python start_audit.py SKILL_DIR [--scope risk|full|targeted] [--paths P ...] [--root DIR] [--date YYYY-MM-DD]

The report goes to ROOT/audits/<skill>/<date>.md and working files to ROOT/audits/<skill>/<date>/.
ROOT defaults to the skill's git top-level, else the current directory; if that is the
skill itself (a standalone skill repo), audits/ goes next to the skill instead, so
nothing is ever written inside it. Targeted --paths are normalized and must stay
inside the skill. A second run on the
same day gets a -2, -3 ... suffix. The skill folder itself is only read; meta.json stores its
tree hash so validate_audit.py can prove that.

Prints the run directory as the last line of stdout. Exit 0 on success, 2 on bad input.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

from audit_common import (
    IGNORED_DIRS,
    IGNORED_FILES,
    IGNORED_SUFFIXES,
    NOT_COPIED_DIRS,
    tree_hash,
)

SCOPE_WORDS = {"risk": "risk-based", "full": "full", "targeted": "targeted"}


def git_toplevel(path: Path) -> Path | None:
    proc = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=False)
    return Path(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip() else None


def audits_is_ignored(root: Path) -> bool:
    proc = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "audits/x"],
                          capture_output=True, text=True, check=False)
    return proc.returncode == 0


def next_run_dir(base: Path, date: str) -> Path:
    candidate, n = base / date, 1
    while candidate.exists() or candidate.with_suffix(".md").exists():
        n += 1
        candidate = base / f"{date}-{n}"
    return candidate


def ignore_junk(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in IGNORED_DIRS or n in NOT_COPIED_DIRS or n in IGNORED_FILES
            or Path(n).suffix in IGNORED_SUFFIXES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("skill_dir", type=Path)
    parser.add_argument("--scope", choices=sorted(SCOPE_WORDS), default=None)
    parser.add_argument("--paths", nargs="*", default=[])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args(argv)

    skill = args.skill_dir.resolve()
    if not (skill / "SKILL.md").is_file():
        print(f"start_audit: {skill} has no SKILL.md", file=sys.stderr)
        return 2
    scope = args.scope
    if scope is None:
        scope = "risk"
        print("start_audit: no scope given, using risk-based; the report will say so", file=sys.stderr)
    paths: list[str] = []
    if scope == "targeted":
        if not args.paths:
            print("start_audit: --scope targeted needs --paths", file=sys.stderr)
            return 2
        for raw in args.paths:
            target = (skill / raw).resolve()
            if not target.is_relative_to(skill) or target == skill or not target.exists():
                print(f"start_audit: --paths {raw!r} must name an existing file or folder inside the skill "
                      f"(for the whole skill use --scope full)", file=sys.stderr)
                return 2
            paths.append(target.relative_to(skill).as_posix())
    elif args.paths:
        print("start_audit: --paths only applies to --scope targeted", file=sys.stderr)
        return 2

    root = (args.root or git_toplevel(skill) or Path.cwd()).resolve()
    if root == skill or root.is_relative_to(skill):
        print(f"start_audit: {root} is the skill or inside it; writing audits/ next to the skill instead, "
              f"in {skill.parent}", file=sys.stderr)
        root = skill.parent
    base = root / "audits" / skill.name
    while True:
        run = next_run_dir(base, args.date)
        try:
            run.mkdir(parents=True)
            break
        except FileExistsError:
            continue
    try:
        shutil.copytree(skill, run / "workspace", ignore=ignore_junk)
    except (shutil.Error, OSError) as exc:
        shutil.rmtree(run, ignore_errors=True)
        print(f"start_audit: could not copy the skill into the workspace: {exc}", file=sys.stderr)
        return 2
    for sub in ("sources", "review", "evidence"):
        (run / sub).mkdir()
    meta = {
        "skill": skill.name,
        "skill_path": str(skill),
        "repo_root": str(root),
        "date": args.date,
        "scope": {"mode": scope, "paths": paths},
        "tree_hash": tree_hash(skill),
        "report": str(run.with_suffix(".md")),
        "tools": [],
    }
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if git_toplevel(root) is None:
        print(f"start_audit: {root} is not a git repository, so nothing to ignore", file=sys.stderr)
    elif not audits_is_ignored(root):
        print(f"start_audit: warning: audits/ is not ignored by git in {root}; add '/audits/' to .gitignore",
              file=sys.stderr)
    print(f"Scope: {SCOPE_WORDS[scope]}. Next: python find_candidates.py {run}")
    print(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
