#!/usr/bin/env python3
"""Claude Code SessionStart hook: remember the commit the session started from.

The Stop gate runs diff_guard.py against this commit, so test edits stay visible
even if Claude commits them during the session. On resume, /clear, or compaction the
first recorded commit is kept. Prints nothing; it only writes the state file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import git_head, project_dir, read_state, state_path


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    project = project_dir(event)
    path = state_path(event, project)
    state = read_state(path)
    if state.get("base_sha"):
        return 0
    head = git_head(project)
    if head:
        state["base_sha"] = head
        path.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
