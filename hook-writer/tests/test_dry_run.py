"""dry_run.py reports what a hook would do and can gate on the expected outcome."""
from __future__ import annotations

import json
import sys

PRE_TOOL = ("--event", "PreToolUse", "--tool", "Bash", "--command")


def test_blocking_hook_is_reported(project, run_script):
    result = run_script("dry_run.py", *PRE_TOOL, "rm -rf build", cwd=project)
    assert result.returncode == 0
    assert "combined PreToolUse outcome: deny" in result.stdout


def test_expect_passes_when_the_outcome_matches(project, run_script):
    result = run_script("dry_run.py", *PRE_TOOL, "rm -rf build", "--expect", "deny", cwd=project)
    assert result.returncode == 0


def test_expect_fails_when_the_outcome_differs(project, run_script):
    result = run_script("dry_run.py", *PRE_TOOL, "ls", "--expect", "deny", cwd=project)
    assert result.returncode == 1
    assert "expected deny" in result.stdout


def test_expect_none_matches_a_hook_that_makes_no_decision(project, run_script):
    result = run_script("dry_run.py", *PRE_TOOL, "ls", "--expect", "none", cwd=project)
    assert result.returncode == 0


def test_expect_fails_when_no_handler_matches(project, run_script):
    result = run_script("dry_run.py", "--event", "PreToolUse", "--tool", "Write", "--file", "a.py",
                        "--expect", "deny", cwd=project)
    assert result.returncode == 1


def _permission_hook(project, behavior):
    decision = {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": {"behavior": behavior}}}
    script = project / ".claude" / "hooks" / "perm.py"
    script.write_text(f"print({json.dumps(decision)!r})\n")
    settings = {"hooks": {"PermissionRequest": [{"matcher": "Bash", "hooks": [
        {"type": "command", "command": f"{sys.executable} .claude/hooks/perm.py"}]}]}}
    (project / ".claude" / "settings.json").write_text(json.dumps(settings))


def test_expect_allow_on_a_permission_request(project, run_script):
    _permission_hook(project, "allow")
    args = ("--event", "PermissionRequest", "--tool", "Bash", "--command", "ls", "--expect", "allow")
    result = run_script("dry_run.py", *args, cwd=project)
    assert result.returncode == 0, result.stdout


def test_expect_deny_on_a_permission_request(project, run_script):
    _permission_hook(project, "deny")
    args = ("--event", "PermissionRequest", "--tool", "Bash", "--command", "ls", "--expect", "deny")
    result = run_script("dry_run.py", *args, cwd=project)
    assert result.returncode == 0, result.stdout


def test_handler_that_was_not_run_is_not_reported_as_no_decision(project, run_script):
    settings = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "http", "url": "http://localhost:1"}]}]}}
    (project / ".claude" / "settings.json").write_text(json.dumps(settings))
    assert run_script("dry_run.py", *PRE_TOOL, "ls", "--expect", "none", cwd=project).returncode == 1
    assert run_script("dry_run.py", *PRE_TOOL, "ls", "--expect", "not-run", cwd=project).returncode == 0


def test_expect_rejects_an_unknown_outcome(project, run_script):
    assert run_script("dry_run.py", *PRE_TOOL, "ls", "--expect", "denny", cwd=project).returncode == 2


def test_invalid_settings_could_not_be_checked(project, run_script):
    (project / ".claude" / "settings.json").write_text("{not json")
    result = run_script("dry_run.py", *PRE_TOOL, "ls", cwd=project)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def _script_hook(project, event, body, matcher="Bash", kind="command"):
    script = project / ".claude" / "hooks" / "h.py"
    script.write_text(body)
    hooks = [{"type": "command", "command": f"{sys.executable} .claude/hooks/h.py"}]
    (project / ".claude" / "settings.json").write_text(json.dumps({"hooks": {event: [{"matcher": matcher, "hooks": hooks}]}}))


def test_pretooluse_hook_that_stops_claude_counts_as_block(project, run_script):
    _script_hook(project, "PreToolUse", 'print(\'{"continue": false}\')\n')
    assert run_script("dry_run.py", *PRE_TOOL, "ls", "--expect", "block", cwd=project).returncode == 0


def test_unrun_handler_keeps_a_weaker_decision_from_passing(project, run_script):
    body = 'print(\'{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}\')\n'
    (project / ".claude" / "hooks" / "h.py").write_text(body)
    hooks = [{"type": "command", "command": f"{sys.executable} .claude/hooks/h.py"}, {"type": "http", "url": "http://localhost:1"}]
    (project / ".claude" / "settings.json").write_text(json.dumps({"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": hooks}]}}))
    assert run_script("dry_run.py", *PRE_TOOL, "ls", "--expect", "allow", cwd=project).returncode == 1
    assert run_script("dry_run.py", *PRE_TOOL, "ls", "--expect", "not-run", cwd=project).returncode == 0


def test_non_utf8_settings_could_not_be_checked(project, run_script):
    (project / ".claude" / "settings.json").write_bytes(b"\xff\xfe")
    result = run_script("dry_run.py", *PRE_TOOL, "ls", cwd=project)
    assert result.returncode == 2 and "Traceback" not in result.stderr


import pytest  # noqa: E402


@pytest.mark.parametrize("content", ["[]", '{"hooks": 3}'])
def test_settings_of_the_wrong_shape_could_not_be_checked(project, run_script, content):
    (project / ".claude" / "settings.json").write_text(content)
    assert run_script("dry_run.py", *PRE_TOOL, "ls", cwd=project).returncode == 2
