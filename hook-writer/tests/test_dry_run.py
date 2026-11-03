"""dry_run.py reports what a hook would do and can gate on the expected outcome."""
from __future__ import annotations

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
