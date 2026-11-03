#!/usr/bin/env python3
"""Claude Code Stop / SubagentStop hook: no stopping while the audit, the diff guard, or
the fast suite is red.

  * Diff guard: inside a git repository, diff_guard.py compares the tests with the commit
    recorded by pytest_expert_session_start.py (else PYTEST_EXPERT_DIFF_BASE, else HEAD)
    and blocks on deleted tests, dropped assertions, new skips, deselection, or a lowered
    coverage floor. PYTEST_EXPERT_DIFF_ALLOW takes comma-separated codes to accept.
  * Escalation exit: if a failing test contradicts the specification, Claude writes
    .claude/pytest-expert-escalation.md (fields below) instead of editing the test or
    special-casing the code. The gate archives the note, lets Claude stop, and shows the
    note and the remaining problems to the user. An incomplete note is sent back.
  * Skips entirely when no .py file changed since the last green run (fingerprint).
  * Honors stop_hook_active, and also keeps its own counter, because a reported bug
    (anthropics/claude-code#54360) can deliver stop_hook_active=false on repeat fires.
    After PYTEST_EXPERT_MAX_BLOCKS consecutive blocks (default 3) it lets Claude stop
    and leaves a visible message instead of looping.

Environment: PYTEST_EXPERT_PYTHON, PYTEST_EXPERT_MAX_BLOCKS, PYTEST_EXPERT_STOP_TIMEOUT (seconds),
PYTEST_EXPERT_DIFF_BASE, PYTEST_EXPERT_DIFF_ALLOW.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AUDIT, GUARD, SKIP_DIRS, project_dir, project_python, pytest_command, read_state, state_path, tail

ESCALATION = Path(".claude") / "pytest-expert-escalation.md"
ESCALATION_FIELDS = ("issue type:", "test:", "problem:", "tried:", "recommendation:")
BLOCK_HEADER = (
    "pytest-expert: do not stop yet. Fix the code so these pass, then finish.\n"
    "Do not edit, skip, or delete a failing test to make it pass, and do not special-case the code for a "
    "test's inputs. If a failing test contradicts the specification, stop and escalate instead: write "
    f"{ESCALATION} with the lines 'Issue type:', 'Test:', 'Problem:', 'Tried:', 'Recommendation:'.\n\n"
)


def read_escalation(project: Path) -> tuple[str | None, list[str]]:
    """(note text or None, missing field names)."""
    path = project / ESCALATION
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, []
    lowered = [line.strip().lower() for line in text.splitlines()]
    missing = [f for f in ESCALATION_FIELDS if not any(line.startswith(f) and line[len(f):].strip() for line in lowered)]
    return text, missing


def archive_escalation(project: Path) -> Path:
    archive = project / ".claude" / "pytest-expert-escalations"
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{os.getpid()}.md"
    (project / ESCALATION).replace(target)
    return target


def fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(project.rglob("*.py")):
        if SKIP_DIRS & set(path.relative_to(project).parts):
            continue
        stat = path.stat()
        digest.update(f"{path.relative_to(project)}:{stat.st_mtime_ns}:{stat.st_size}\n".encode())
    return digest.hexdigest()


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if event.get("stop_hook_active"):
        return 0
    project = project_dir(event)
    tests = project / "tests"
    if not tests.is_dir():
        return 0

    path = state_path(event, project)
    state = read_state(path)

    current = fingerprint(project)
    if state.get("last_green") == current:
        return 0
    max_blocks = int(os.environ.get("PYTEST_EXPERT_MAX_BLOCKS", "3"))
    if state.get("blocks", 0) >= max_blocks:
        path.write_text(json.dumps({**state, "blocks": 0}))
        print(json.dumps({"systemMessage": f"pytest-expert stop gate gave up after {max_blocks} attempts; the suite is still red."}))
        return 0

    python = project_python(project)
    timeout = int(os.environ.get("PYTEST_EXPERT_STOP_TIMEOUT", "480"))
    problems = []
    audit = subprocess.run([python, str(AUDIT), str(tests)], capture_output=True, text=True, timeout=120, cwd=project)
    if audit.returncode == 1:
        problems.append("Audit findings:\n" + audit.stdout)
    if GUARD.exists():
        base = state.get("base_sha") or os.environ.get("PYTEST_EXPERT_DIFF_BASE", "HEAD")
        allow = os.environ.get("PYTEST_EXPERT_DIFF_ALLOW", "")
        guard = subprocess.run([python, str(GUARD), str(project), "--base", base, "--allow", allow],
                               capture_output=True, text=True, timeout=120, cwd=project)
        if guard.returncode == 1:
            problems.append(
                f"Test-weakening edits since {base[:12]} (if the user asked for one, say so and they can set "
                "PYTEST_EXPERT_DIFF_ALLOW):\n" + guard.stdout
            )
    try:
        run = subprocess.run(pytest_command(python), capture_output=True, text=True, timeout=timeout, cwd=project)
        if run.returncode not in (0, 5):
            problems.append("Fast suite failing:\n" + run.stdout)
    except subprocess.TimeoutExpired:
        problems.append(f"Fast suite exceeded {timeout}s; something is hanging or far too slow.")

    note, missing = read_escalation(project)
    if note is not None and not missing:
        target = archive_escalation(project)
        path.write_text(json.dumps({**state, "blocks": 0}))
        status = "The suite is still red:\n" + "\n\n".join(problems) if problems else "The suite is green."
        message = f"pytest-expert: Claude escalated instead of finishing (saved to {target.relative_to(project)}).\n\n{note}\n\n{status}"
        print(json.dumps({"systemMessage": tail(message)}))
        return 0
    if note is not None:
        problems.insert(0, f"Escalation note {ESCALATION} is missing: {', '.join(missing)}")

    if not problems:
        path.write_text(json.dumps({**state, "last_green": current, "blocks": 0}))
        return 0
    path.write_text(json.dumps({**state, "blocks": state.get("blocks", 0) + 1}))
    reason = BLOCK_HEADER + "\n\n".join(problems)
    print(json.dumps({"decision": "block", "reason": tail(reason)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
