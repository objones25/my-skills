"""Drive the Claude Code hooks the way Claude Code does: JSON on stdin, JSON on stdout."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest
from conftest import SKILL_ROOT

HOOKS = SKILL_ROOT / "assets" / "claude" / "hooks"
GOOD = "def test_adds():\n    assert 1 + 1 == 2\n"
BAD_AUDIT = "def test_nothing():\n    sum([1])\n"
RED = "def test_red():\n    assert 1 + 1 == 3\n"


@pytest.fixture
def project(tmp_path, monkeypatch):
    hooks = tmp_path / ".claude" / "hooks"
    shutil.copytree(HOOKS, hooks)
    shutil.copy(SKILL_ROOT / "scripts" / "audit_tests.py", hooks / "audit_tests.py")
    shutil.copy(SKILL_ROOT / "scripts" / "diff_guard.py", hooks / "diff_guard.py")
    (tmp_path / "tests").mkdir()
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("PYTEST_EXPERT_PYTHON", sys.executable)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "state"))
    (tmp_path / "state").mkdir()
    return tmp_path


def fire(project, hook: str, event: dict) -> dict | None:
    proc = subprocess.run(
        [sys.executable, str(project / ".claude" / "hooks" / hook)],
        input=json.dumps(event), capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def edit_event(path):
    return {"hook_event_name": "PostToolUse", "tool_name": "Write", "tool_input": {"file_path": str(path)}}


def test_post_edit_blocks_with_audit_findings(project):
    path = project / "tests" / "test_x.py"
    path.write_text(BAD_AUDIT)
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out["decision"] == "block"
    assert "NO_ASSERT" in out["reason"]


def test_post_edit_reports_red_tests_as_context_not_block(project):
    path = project / "tests" / "test_x.py"
    path.write_text(RED)
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert "decision" not in out
    assert "fails right now" in out["hookSpecificOutput"]["additionalContext"]


def test_post_edit_is_silent_for_clean_passing_tests(project):
    path = project / "tests" / "test_x.py"
    path.write_text(GOOD)
    assert fire(project, "pytest_expert_post_edit.py", edit_event(path)) is None


def test_post_edit_ignores_non_test_files(project):
    path = project / "module.py"
    path.write_text("x = 1\n")
    assert fire(project, "pytest_expert_post_edit.py", edit_event(path)) is None


def stop_event(**extra):
    return {"hook_event_name": "Stop", "session_id": "s1", "stop_hook_active": False, **extra}


def test_stop_gate_blocks_on_a_red_suite(project):
    (project / "tests" / "test_x.py").write_text(RED)
    out = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert out["decision"] == "block"
    assert "Fast suite failing" in out["reason"]


def test_stop_gate_passes_green_suite_then_skips_unchanged_tree(project):
    (project / "tests" / "test_x.py").write_text(GOOD)
    first = fire(project, "pytest_expert_stop_gate.py", stop_event())
    second = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert (first, second) == (None, None)


def test_stop_gate_honors_stop_hook_active(project):
    (project / "tests" / "test_x.py").write_text(RED)
    assert fire(project, "pytest_expert_stop_gate.py", stop_event(stop_hook_active=True)) is None


def test_stop_gate_gives_up_after_max_blocks(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "2")
    (project / "tests" / "test_x.py").write_text(RED)
    outcomes = [fire(project, "pytest_expert_stop_gate.py", stop_event()) for _ in range(3)]
    assert [("decision" in o, "systemMessage" in o) for o in outcomes] == [(True, False), (True, False), (False, True)]


def git(project, *args):
    subprocess.run(["git", *args], cwd=project, check=True, capture_output=True)


@pytest.fixture
def git_project(project, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "t")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "t@example.com")
    (project / "tests" / "test_x.py").write_text(GOOD + "\n\ndef test_red():\n    assert 1 + 1 == 3\n")
    (project / ".gitignore").write_text(".claude/\nstate/\n")
    git(project, "init", "-q", "-b", "main")
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "base")
    return project


def test_stop_gate_blocks_when_the_failing_test_was_deleted(git_project):
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    out = fire(git_project, "pytest_expert_stop_gate.py", stop_event())
    assert out["decision"] == "block"
    assert "TEST_DELETED" in out["reason"]


def test_stop_gate_blocks_when_the_failing_test_was_skipped(git_project):
    (git_project / "tests" / "test_x.py").write_text(
        "import pytest\n" + GOOD + "\n\n@pytest.mark.skip(reason='later')\ndef test_red():\n    assert 1 + 1 == 3\n"
    )
    out = fire(git_project, "pytest_expert_stop_gate.py", stop_event())
    assert "SKIP_ADDED" in out["reason"]


def test_session_start_base_survives_a_commit_of_the_weakened_tests(git_project):
    assert fire(git_project, "pytest_expert_session_start.py", {"hook_event_name": "SessionStart", "session_id": "s1", "source": "startup"}) is None
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    git(git_project, "commit", "-q", "-am", "make it green")
    out = fire(git_project, "pytest_expert_stop_gate.py", stop_event())
    assert "TEST_DELETED" in out["reason"]


def test_diff_allow_env_lets_an_approved_deletion_through(git_project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_DIFF_ALLOW", "TEST_DELETED")
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    assert fire(git_project, "pytest_expert_stop_gate.py", stop_event()) is None


def test_subagent_stop_event_is_gated_the_same_way(git_project):
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    out = fire(git_project, "pytest_expert_stop_gate.py", {**stop_event(), "hook_event_name": "SubagentStop"})
    assert out["decision"] == "block"


NOTE = (
    "Issue type: test contradicts spec\n"
    "Test: tests/test_x.py::test_red\n"
    "Problem: expects 1 + 1 == 3; the spec says addition\n"
    "Tried: read the spec and the callers\n"
    "Recommendation: fix the expected value to 2\n"
)


def test_block_message_names_the_escalation_path(project):
    (project / "tests" / "test_x.py").write_text(RED)
    out = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert "pytest-expert-escalation.md" in out["reason"]
    assert "Do not edit, skip, or delete a failing test" in out["reason"]


def test_complete_escalation_note_lets_claude_stop_and_reaches_the_user(project):
    (project / "tests" / "test_x.py").write_text(RED)
    (project / ".claude" / "pytest-expert-escalation.md").write_text(NOTE)
    out = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert "decision" not in out
    assert "Claude escalated" in out["systemMessage"]
    assert "Recommendation: fix the expected value to 2" in out["systemMessage"]
    assert "Fast suite failing" in out["systemMessage"]
    assert not (project / ".claude" / "pytest-expert-escalation.md").exists()
    assert len(list((project / ".claude" / "pytest-expert-escalations").glob("*.md"))) == 1


def test_escalation_is_one_shot(project):
    (project / "tests" / "test_x.py").write_text(RED)
    (project / ".claude" / "pytest-expert-escalation.md").write_text(NOTE)
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    second = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert second["decision"] == "block"


def test_incomplete_escalation_note_is_sent_back(project):
    (project / "tests" / "test_x.py").write_text(RED)
    (project / ".claude" / "pytest-expert-escalation.md").write_text("Test: tests/test_x.py::test_red\nProblem: wrong\n")
    out = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert out["decision"] == "block"
    assert "missing: issue type:, tried:, recommendation:" in out["reason"]
