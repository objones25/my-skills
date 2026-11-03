#!/usr/bin/env python3
"""Claude Code PostToolUse hook: audit a test file the moment Claude writes or edits it.

Audit findings come back to Claude as a blocking decision with the finding list.
Then the edited test file is run; failures are reported as context, not as a
block, because a red test is the expected state partway through test-first work.

Set PYTEST_EXPERT_RUN_EDITED_TESTS=0 to skip running the file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AUDIT, project_dir, project_python, pytest_command, tail


def is_test_file(path: Path) -> bool:
    return path.suffix == ".py" and (path.name.startswith("test_") or path.name.endswith("_test.py") or path.name == "conftest.py")


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    path = Path((event.get("tool_input") or {}).get("file_path") or "")
    if not is_test_file(path) or not path.exists():
        return 0
    project = project_dir(event)
    python = project_python(project)

    audit = subprocess.run([python, str(AUDIT), str(path)], capture_output=True, text=True, timeout=60, cwd=project)
    if audit.returncode == 1:
        reason = (
            f"pytest-expert audit flagged {path.name}. Fix each finding (or add "
            "`# audit: ignore[CODE] <reason>` when it is a deliberate exception):\n" + audit.stdout
        )
        print(json.dumps({"decision": "block", "reason": tail(reason)}))
        return 0

    if os.environ.get("PYTEST_EXPERT_RUN_EDITED_TESTS", "1") == "1" and path.name != "conftest.py":
        try:
            run = subprocess.run(pytest_command(python, str(path)), capture_output=True, text=True, timeout=90, cwd=project)
        except subprocess.TimeoutExpired:
            context = f"pytest-expert: {path.name} took over 90s to run; look for sleeps, network, or a slow fixture."
        else:
            if run.returncode in (0, 5):
                return 0
            context = f"pytest-expert: {path.name} fails right now (fine if you are writing tests first):\n" + run.stdout
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": tail(context)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
