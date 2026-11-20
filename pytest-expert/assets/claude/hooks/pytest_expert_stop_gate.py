#!/usr/bin/env python3
"""Claude Code Stop / SubagentStop hook: no stopping while the audit, the diff guard, or
the fast suite is red, and no silent pass when a check could not run.

  * Audit: audit_tests.py over the project (every test file it finds).
  * Diff guard: inside a git repository, diff_guard.py compares the tests with the commit
    recorded by pytest_expert_session_start.py (else PYTEST_EXPERT_DIFF_BASE, else HEAD)
    and blocks on deleted tests, dropped assertions, new skips, deselection, or a lowered
    coverage floor. PYTEST_EXPERT_DIFF_ALLOW takes comma-separated codes to accept.
  * Fast suite: pytest on the whole project, so pytest itself decides where the tests are
    (tests/, test/, the root, testpaths). Exit 5 (nothing collected) is not green: it is a
    visible warning that nothing was checked, and the tree is not remembered as green.
  * The scripts are found next to the hook, else in the skill's scripts/ folder. Their exit
    status is the contract: 0 clean, 1 findings, anything else (exit 2, a crash, a timeout,
    a missing script, an unknown diff base) means the check could not run. That BLOCKS, with
    a reason naming what could not run and how to fix it, and shows a systemMessage. The
    give-up counter below keeps such a block from becoming a trap. The exceptions are
    warnings (shown, not blocking): not a git repository, so no diff guard; pytest exit 5.
  * Any unexpected exception in the gate blocks once with the traceback; an unreadable event
    prints a systemMessage and lets the stop through, since nothing can be checked without it.
  * Escalation exit: if a failing test contradicts the specification, Claude writes
    .claude/pytest-expert-escalation.md (fields below) instead of editing the test or
    special-casing the code. The gate archives the note, lets Claude stop, and shows the
    note and the remaining problems to the user. An incomplete note is sent back.
  * Skips entirely when no .py file and no pytest/coverage config file changed since the
    last green run (fingerprint).
  * Its own counter is the loop guard. `stop_hook_active` (Claude Code is already continuing
    because of a Stop hook) is not a reason to let a red tree stop; when it is true and the gate
    blocks again, the reason says "block N of MAX". After PYTEST_EXPERT_MAX_BLOCKS consecutive
    blocks (default 3) it lets Claude stop and leaves a visible message naming the last problem
    instead of looping. The counter lives in the per-session state file and is reset only by a
    green run, a give-up or an escalation. Claude Code itself overrides a Stop block after
    CLAUDE_CODE_STOP_HOOK_BLOCK_CAP consecutive ones (default 8, 0 = no cap), so the limit is
    lowered to that cap and the gate always gives up first.

Environment: PYTEST_EXPERT_PYTHON, PYTEST_EXPERT_MAX_BLOCKS, PYTEST_EXPERT_STOP_TIMEOUT (seconds, fast
suite, default 480), PYTEST_EXPERT_CHECK_TIMEOUT (seconds, audit and diff guard each, default 120),
PYTEST_EXPERT_DIFF_BASE, PYTEST_EXPERT_DIFF_ALLOW. The three numbers must be positive integers; a bad
value is replaced by its default and reported in a systemMessage.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    CONFIG_FILES,
    SKIP_DIRS,
    emit,
    find_script,
    git_head,
    int_env,
    missing_script_message,
    project_dir,
    project_python,
    pytest_command,
    read_event,
    read_state,
    run_detector,
    state_path,
    tail,
)

PLATFORM_BLOCK_CAP_DEFAULT = 8  # Claude Code's default consecutive Stop-block cap (hooks reference, Stop input)
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
    for name in CONFIG_FILES:
        try:
            stat = (project / name).stat()
        except OSError:
            continue
        digest.update(f"{name}:{stat.st_mtime_ns}:{stat.st_size}\n".encode())
    return digest.hexdigest()


class Checks:
    """What the three checks found. `problems` are failures, `unchecked` are checks that could not run
    (both block), `warnings` are shown only when the stop is allowed."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.unchecked: list[str] = []
        self.warnings: list[str] = []

    def blocking(self) -> list[str]:
        return self.unchecked + self.problems


def check_audit(checks: Checks, python: str, project: Path, timeout_s: int) -> str | None:
    """Run the audit; return its could-not-check reason (also recorded in checks.unchecked)."""
    script = find_script("audit_tests.py")
    if script is None:
        reason = missing_script_message("audit_tests.py")
    else:
        findings, reason = run_detector("audit_tests.py", [python, str(script), str(project)], timeout_s=timeout_s, cwd=project)
        if findings:
            checks.problems.append("Audit findings:\n" + findings)
    if reason:
        checks.unchecked.append("COULD NOT CHECK: " + reason)
    return reason


def check_guard(checks: Checks, state: dict, python: str, project: Path, timeout_s: int) -> None:
    explicit_base = os.environ.get("PYTEST_EXPERT_DIFF_BASE")
    base = state.get("base_sha") or explicit_base or "HEAD"
    if not state.get("base_sha") and not explicit_base and git_head(project) is None:
        checks.warnings.append("the project is not a git repository (or has no commit), so the diff guard did not run and "
                               "weakened tests would not be caught.")
        return
    script = find_script("diff_guard.py")
    if script is None:
        checks.unchecked.append("COULD NOT CHECK: " + missing_script_message("diff_guard.py"))
        return
    allow = os.environ.get("PYTEST_EXPERT_DIFF_ALLOW", "")
    findings, reason = run_detector("diff_guard.py", [python, str(script), str(project), "--base", base, "--allow", allow],
                                    timeout_s=timeout_s, cwd=project)
    if findings:
        checks.problems.append(
            f"Test-weakening edits since {base[:12]} (if the user asked for one, say so and they can set "
            "PYTEST_EXPERT_DIFF_ALLOW):\n" + findings
        )
    if reason:
        checks.unchecked.append(f"COULD NOT CHECK: {reason} The base was {base!r}: it is the commit the session started from, "
                                "else PYTEST_EXPERT_DIFF_BASE, else HEAD; fix PYTEST_EXPERT_DIFF_BASE if it is wrong.")


def check_suite(checks: Checks, python: str, project: Path, timeout_s: int) -> int | None:
    """Run the fast suite; return pytest's exit code (None if it did not finish)."""
    try:
        run = subprocess.run(pytest_command(python), capture_output=True, text=True, timeout=timeout_s, cwd=project, check=False)
    except subprocess.TimeoutExpired:
        checks.problems.append(f"Fast suite exceeded {timeout_s}s; something is hanging or far too slow.")
        return None
    except OSError as exc:
        checks.unchecked.append(f"COULD NOT CHECK: pytest could not be started with {python} ({exc}); set PYTEST_EXPERT_PYTHON.")
        return None
    if run.returncode == 0:
        pass
    elif run.returncode == 5:
        checks.warnings.append("pytest: no tests were collected, so the fast suite checked nothing (is the test path or a -k/-m "
                               "option in the pytest config wrong?).")
    elif run.returncode == 1 and "No module named pytest" in run.stderr:
        checks.unchecked.append(f"COULD NOT CHECK: pytest is not installed in {python}; install it there or set PYTEST_EXPERT_PYTHON.")
    elif run.returncode == 1:
        checks.problems.append("Fast suite failing:\n" + run.stdout)
    else:
        checks.unchecked.append(f"COULD NOT CHECK: Fast suite could not run (pytest exit {run.returncode}):\n" + (run.stdout or run.stderr))
    return run.returncode


def run_gate(event: dict) -> int:
    project = project_dir(event)
    path = state_path(event, project)
    state = read_state(path)

    current = fingerprint(project)
    if state.get("last_green") == current:
        return 0
    notes: list[str] = []
    max_blocks, bad = int_env("PYTEST_EXPERT_MAX_BLOCKS", 3)
    notes += [bad] if bad else []
    platform_cap, bad = int_env("CLAUDE_CODE_STOP_HOOK_BLOCK_CAP", PLATFORM_BLOCK_CAP_DEFAULT, allow_zero=True)
    notes += [bad] if bad else []
    if platform_cap and max_blocks > platform_cap:
        notes.append(f"PYTEST_EXPERT_MAX_BLOCKS={max_blocks} is above CLAUDE_CODE_STOP_HOOK_BLOCK_CAP={platform_cap}, after which "
                     f"Claude Code overrides a block; using {platform_cap}.")
        max_blocks = platform_cap
    suite_timeout_s, bad = int_env("PYTEST_EXPERT_STOP_TIMEOUT", 480)
    notes += [bad] if bad else []
    check_timeout_s, bad = int_env("PYTEST_EXPERT_CHECK_TIMEOUT", 120)
    notes += [bad] if bad else []
    if state.get("blocks", 0) >= max_blocks:
        path.write_text(json.dumps({**state, "blocks": 0}))
        last = state.get("last_problems") or "(not recorded)"
        emit(systemMessage=tail(f"pytest-expert stop gate gave up after {max_blocks} attempts and let Claude stop. "
                                f"The last problems it saw:\n{last}\n" + " ".join(notes)))
        return 0

    python = project_python(project)
    checks = Checks()
    audit_reason = check_audit(checks, python, project, check_timeout_s)
    check_guard(checks, state, python, project, check_timeout_s)
    suite_exit = check_suite(checks, python, project, suite_timeout_s)
    if suite_exit == 5 and audit_reason:
        # No test file at all: an audit that finds nothing to audit is not a second failure.
        checks.unchecked = [u for u in checks.unchecked if not u.startswith("COULD NOT CHECK: audit_tests.py")]
        checks.warnings.append(audit_reason)

    note, missing = read_escalation(project)
    if note is not None and not missing:
        target = archive_escalation(project)
        path.write_text(json.dumps({**state, "blocks": 0}))
        remaining = checks.blocking()
        status = "The suite is still red:\n" + "\n\n".join(remaining) if remaining else "The suite is green."
        message = f"pytest-expert: Claude escalated instead of finishing (saved to {target.relative_to(project)}).\n\n{note}\n\n{status}"
        emit(systemMessage=tail(message))
        return 0
    if note is not None:
        checks.problems.insert(0, f"Escalation note {ESCALATION} is missing: {', '.join(missing)}")

    blocking = checks.blocking()
    if not blocking:
        # Exit 5 checked nothing, so it must not be remembered as a green tree.
        green = {"last_green": current} if suite_exit != 5 else {}
        path.write_text(json.dumps({**state, **green, "blocks": 0}))
        shown = notes + checks.warnings
        if shown:
            emit(systemMessage=tail("pytest-expert stop gate: " + " ".join(shown)))
        return 0
    path.write_text(json.dumps({**state, "blocks": state.get("blocks", 0) + 1, "last_problems": tail("\n\n".join(blocking), 1500)}))
    reason = BLOCK_HEADER + "\n\n".join(blocking)
    if event.get("stop_hook_active"):
        reason = (f"pytest-expert: block {state.get('blocks', 0) + 1} of {max_blocks}; the gate stops blocking after block "
                  f"{max_blocks} (Claude Code reports that a Stop hook already continued this turn).\n") + reason
    fields: dict[str, object] = {"decision": "block", "reason": tail(reason)}
    if checks.unchecked or notes:
        fields["systemMessage"] = tail("pytest-expert stop gate: " + (
            f"{len(checks.unchecked)} check(s) could not check anything; Claude was told what to fix. " if checks.unchecked else "") + " ".join(notes))
    emit(**fields)
    return 0


def main() -> int:
    event = read_event()
    if event is None:
        emit(systemMessage="pytest-expert stop gate: the event on stdin was empty, malformed or not a JSON object, "
                           "so the gate did not run and nothing was checked before this stop.")
        return 0
    try:
        return run_gate(event)
    except Exception:  # noqa: BLE001 - last resort: exit 1 would be non-blocking and invisible, so block with the traceback
        detail = traceback.format_exc(limit=-3)
        emit(decision="block", reason=tail(f"pytest-expert stop gate crashed, so it could not check anything:\n{detail}"),
             systemMessage="pytest-expert stop gate: the gate crashed and could not check anything; Claude was shown the traceback.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
