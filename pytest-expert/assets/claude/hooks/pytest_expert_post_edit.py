#!/usr/bin/env python3
"""Claude Code PostToolUse hook: audit a test file the moment Claude writes or edits it.

Audit findings come back to Claude as a blocking decision with the finding list.
Then the edited test file is run; failures are reported as context, not as a
block, because a red test is the expected state partway through test-first work.

audit_tests.py is found next to the hook, else in the skill's scripts/ folder. Its
exit status is the contract: 0 clean, 1 findings, anything else (exit 2, a crash, a
timeout, a missing script) means the audit could not run. That is reported to Claude
as additionalContext and to the user as a systemMessage, never read as "clean".

Set PYTEST_EXPERT_RUN_EDITED_TESTS=0 to skip running the file.
PYTEST_EXPERT_CHECK_TIMEOUT (seconds, default 60) bounds the audit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    emit,
    find_script,
    int_env,
    missing_script_message,
    project_dir,
    project_python,
    pytest_command,
    read_event,
    run_detector,
    tail,
)

PYTEST_RUN_TIMEOUT_S = 90


def is_test_file(path: Path) -> bool:
    return path.suffix == ".py" and (path.name.startswith("test_") or path.name.endswith("_test.py") or path.name == "conftest.py")


def warn(message: str) -> None:
    """Show Claude and the user that a check did not happen."""
    emit(
        systemMessage=f"pytest-expert: {message}",
        hookSpecificOutput={"hookEventName": "PostToolUse", "additionalContext": tail(f"pytest-expert could not check this edit: {message}")},
    )


def check_edit(event: dict) -> None:
    tool_input = event.get("tool_input")
    if tool_input is None:
        return
    if not isinstance(tool_input, dict):
        emit(systemMessage="pytest-expert post-edit hook: tool_input is not an object, so the edit was not checked.")
        return
    raw_path = tool_input.get("file_path")
    if not isinstance(raw_path, str) or not raw_path:
        return
    project = project_dir(event)
    path = Path(raw_path)
    if not path.is_absolute():
        path = project / path
    if not is_test_file(path) or not path.exists():
        return
    python = project_python(project)

    timeout_s, bad_setting = int_env("PYTEST_EXPERT_CHECK_TIMEOUT", 60)
    audit_script = find_script("audit_tests.py")
    if audit_script is None:
        warn(missing_script_message("audit_tests.py"))
        return
    findings, unchecked = run_detector("audit_tests.py", [python, str(audit_script), str(path)], timeout_s=timeout_s, cwd=project)
    if unchecked:
        warn(unchecked + (f" {bad_setting}" if bad_setting else ""))
        return
    if findings:
        reason = (
            f"pytest-expert audit flagged {path.name}. Fix each finding (or add "
            "`# audit: ignore[CODE] <reason>` when it is a deliberate exception):\n" + findings
        )
        emit(decision="block", reason=tail(reason))
        return

    if os.environ.get("PYTEST_EXPERT_RUN_EDITED_TESTS", "1") == "1" and path.name != "conftest.py":
        try:
            run = subprocess.run(pytest_command(python, str(path)), capture_output=True, text=True, timeout=PYTEST_RUN_TIMEOUT_S, cwd=project, check=False)
        except subprocess.TimeoutExpired:
            context = f"pytest-expert: {path.name} took over {PYTEST_RUN_TIMEOUT_S}s to run; look for sleeps, network, or a slow fixture."
        except OSError as exc:
            context = f"pytest-expert: pytest could not run {path.name} ({exc}); set PYTEST_EXPERT_PYTHON to an interpreter that has pytest."
        else:
            if run.returncode in (0, 5):
                return
            if run.returncode == 1:
                context = f"pytest-expert: {path.name} fails right now (fine if you are writing tests first):\n" + run.stdout
            else:
                context = f"pytest-expert: pytest could not run {path.name} (exit {run.returncode}):\n" + (run.stdout or run.stderr)
        emit(hookSpecificOutput={"hookEventName": "PostToolUse", "additionalContext": tail(context)})


def main() -> int:
    event = read_event()
    if event is None:
        emit(systemMessage="pytest-expert post-edit hook: the event on stdin was empty, malformed or not a JSON object, so the edit was not checked.")
        return 0
    try:
        check_edit(event)
    except Exception:  # noqa: BLE001 - last resort: a crash must be shown, not left as exit 1 (non-blocking, invisible)
        warn("the post-edit hook crashed, so the edit was not checked:\n" + traceback.format_exc(limit=-3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
