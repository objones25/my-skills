#!/usr/bin/env python3
"""SessionStart hook (matcher "startup|resume|compact"): give Claude current repo state.

SessionStart adds plain-text stdout to Claude's context, so this prints
facts, not instructions. Static conventions belong in CLAUDE.md instead.
"""
import json
import os
import subprocess
import sys


def git(args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main():
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        event = {}
    cwd = os.environ.get("CLAUDE_PROJECT_DIR", event.get("cwd", "."))
    branch = git(["branch", "--show-current"], cwd)
    if not branch:
        return 0  # not a git repo, nothing useful to say
    dirty = git(["status", "--porcelain"], cwd).splitlines()
    log = git(["log", "--oneline", "-5"], cwd)
    lines = [f"Current git branch: {branch}.",
             f"Uncommitted changes: {len(dirty)} file(s)."]
    if log:
        lines.append("Recent commits:\n" + log)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
