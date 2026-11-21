#!/usr/bin/env python3
"""Claude Code SessionStart hook: remember the commit the session started from.

The Stop gate runs diff_guard.py against this commit, so test edits stay visible
even if Claude commits them during the session. On resume, /clear, or compaction the
first recorded commit is kept. Prints nothing when it works (outside a git repository
there is nothing to record). If it cannot read its event or write its state file it
prints a systemMessage saying so, and the Stop gate then compares against
PYTEST_EXPERT_DIFF_BASE or HEAD instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit, git_head, project_dir, read_event, read_state, state_path


def record_base(event: dict) -> None:
    project = project_dir(event)
    path = state_path(event, project)
    state = read_state(path)
    if state.get("base_sha"):
        return
    head = git_head(project)
    if head:
        state["base_sha"] = head
        path.write_text(json.dumps(state))


def main() -> int:
    event = read_event()
    if event is None:
        emit(systemMessage="pytest-expert session-start hook: the event on stdin was empty, malformed or not a JSON object, "
                           "so the session's base commit was not recorded; the Stop gate will compare against PYTEST_EXPERT_DIFF_BASE or HEAD.")
        return 0
    try:
        record_base(event)
    except OSError as exc:  # state file unreadable or unwritable: show it rather than a traceback
        emit(systemMessage=f"pytest-expert session-start hook: could not record the session base commit ({type(exc).__name__}: {exc}); "
                           "the Stop gate will compare against PYTEST_EXPERT_DIFF_BASE or HEAD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
