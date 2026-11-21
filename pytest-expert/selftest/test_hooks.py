"""Drive the Claude Code hooks the way Claude Code does: JSON on stdin, JSON on stdout."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SKILL_ROOT

HOOKS = SKILL_ROOT / "assets" / "claude" / "hooks"
GOOD = "def test_adds():\n    assert 1 + 1 == 2\n"
BAD_AUDIT = "def test_nothing():\n    sum([1])\n"
RED = "def test_red():\n    assert 1 + 1 == 3\n"
COUNTING = (
    "from pathlib import Path\n\n\n"
    "def test_counts_its_runs():\n"
    "    with open(Path(__file__).with_name('runs.log'), 'a') as log:\n"
    "        log.write('x')\n"
    "    assert log.name.endswith('runs.log')\n"
)


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


def run_hook(script: Path, stdin_text: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script)], input=stdin_text, capture_output=True, text=True, timeout=120, check=False)


def parse_stdout(proc: subprocess.CompletedProcess) -> dict | None:
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def fire(project, hook: str, event: dict) -> dict | None:
    return parse_stdout(run_hook(project / ".claude" / "hooks" / hook, json.dumps(event)))


def state_file_for(project, event: dict) -> Path:
    """Where the hooks keep their state for this project (the `project` fixture points TMPDIR at project/state)."""
    common = load_common()
    return Path(project / "state") / common.state_path(event, Path(project).resolve()).name


def load_common():
    spec = importlib.util.spec_from_file_location("common_under_test", HOOKS / "_common.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def edit_event(path):
    return {"hook_event_name": "PostToolUse", "tool_name": "Write", "tool_input": {"file_path": str(path)}}


def test_post_edit_blocks_with_audit_findings(project):
    path = project / "tests" / "test_x.py"
    path.write_text(BAD_AUDIT)
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out is not None
    assert out["decision"] == "block"
    assert "NO_ASSERT" in out["reason"]


def test_post_edit_reports_red_tests_as_context_not_block(project):
    path = project / "tests" / "test_x.py"
    path.write_text(RED)
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out is not None
    assert set(out) == {"hookSpecificOutput"}  # no block, and no "hook crashed" system message
    context = out["hookSpecificOutput"]["additionalContext"]
    assert context.startswith("pytest-expert: test_x.py fails right now (fine if you are writing tests first):\n")
    assert "test_red" in context


def test_post_edit_is_silent_for_clean_passing_tests(project):
    path = project / "tests" / "test_x.py"
    path.write_text(GOOD)
    assert fire(project, "pytest_expert_post_edit.py", edit_event(path)) is None


@pytest.mark.parametrize(("name", "source"), [("helpers.py", BAD_AUDIT), ("module.py", RED)], ids=["helpers", "module"])
def test_post_edit_ignores_non_test_files(project, name, source):
    # Either file would be reported if it were audited or run: helpers.py has a test with no
    # assertion (audit: NO_ASSERT) and module.py has a red test (pytest collects an explicit path).
    path = project / name
    path.write_text(source)
    assert fire(project, "pytest_expert_post_edit.py", edit_event(path)) is None


def test_post_edit_audits_the_underscore_test_suffix(project):
    path = project / "tests" / "calc_test.py"
    path.write_text(BAD_AUDIT)
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out is not None
    assert out["decision"] == "block"
    assert "NO_ASSERT" in out["reason"]


def test_post_edit_audits_but_does_not_run_conftest(project):
    path = project / "tests" / "conftest.py"
    path.write_text("raise RuntimeError('conftest is imported by pytest, not run on its own')\n")
    assert fire(project, "pytest_expert_post_edit.py", edit_event(path)) is None


def test_post_edit_can_be_told_not_to_run_the_edited_file(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_RUN_EDITED_TESTS", "0")
    path = project / "tests" / "test_x.py"
    path.write_text(RED)
    assert fire(project, "pytest_expert_post_edit.py", edit_event(path)) is None


def test_post_edit_is_silent_for_a_test_file_with_no_tests(project):
    path = project / "tests" / "test_empty.py"
    path.write_text("VALUE = 1\n")
    assert fire(project, "pytest_expert_post_edit.py", edit_event(path)) is None


def test_post_edit_resolves_a_relative_path_against_the_project(project):
    (project / "tests" / "test_x.py").write_text(BAD_AUDIT)
    out = fire(project, "pytest_expert_post_edit.py", edit_event("tests/test_x.py"))
    assert out is not None
    assert out["decision"] == "block"
    assert "NO_ASSERT" in out["reason"]


def test_post_edit_says_so_when_pytest_cannot_run_the_file(project):
    path = project / "tests" / "test_x.py"
    path.write_text("import module_that_does_not_exist\n\n\ndef test_x():\n    assert module_that_does_not_exist\n")
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out is not None
    assert set(out) == {"hookSpecificOutput"}
    context = out["hookSpecificOutput"]["additionalContext"]
    assert context.startswith("pytest-expert: pytest could not run test_x.py (exit 2):\n")
    assert "module_that_does_not_exist" in context


@pytest.mark.parametrize("stdin_text", ["{not json", "[1, 2]", ""])
def test_post_edit_reports_an_unreadable_event(project, stdin_text):
    out = parse_stdout(run_hook(project / ".claude" / "hooks" / "pytest_expert_post_edit.py", stdin_text))
    assert out is not None
    assert "pytest-expert post-edit hook" in out["systemMessage"]
    assert "not a JSON object" in out["systemMessage"]


def test_post_edit_reports_a_tool_input_that_is_not_an_object(project):
    event = {"hook_event_name": "PostToolUse", "tool_name": "Write", "tool_input": "tests/test_x.py"}
    out = fire(project, "pytest_expert_post_edit.py", event)
    assert out is not None
    assert "tool_input" in out["systemMessage"]


def test_post_edit_reports_a_crashing_audit_instead_of_an_empty_finding_list(project):
    (project / ".claude" / "hooks" / "audit_tests.py").write_text("raise RuntimeError('audit bug')\n")
    path = project / "tests" / "test_x.py"
    path.write_text(GOOD)
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out is not None
    assert "decision" not in out
    context = out["hookSpecificOutput"]["additionalContext"]
    assert "audit_tests.py could not check (exit 1)" in context
    assert "audit bug" in context
    assert "could not check" in out["systemMessage"]


def test_post_edit_reports_an_audit_that_exits_2(project):
    (project / ".claude" / "hooks" / "audit_tests.py").write_text("import sys\nprint('bad arguments', file=sys.stderr)\nsys.exit(2)\n")
    path = project / "tests" / "test_x.py"
    path.write_text(GOOD)
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out is not None
    assert "audit_tests.py could not check (exit 2): bad arguments" in out["hookSpecificOutput"]["additionalContext"]


def test_post_edit_reports_an_audit_timeout(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_CHECK_TIMEOUT", "1")
    (project / ".claude" / "hooks" / "audit_tests.py").write_text("import time\ntime.sleep(60)\n")
    path = project / "tests" / "test_x.py"
    path.write_text(GOOD)
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out is not None
    assert "did not finish within 1s" in out["hookSpecificOutput"]["additionalContext"]


def test_post_edit_reports_an_interpreter_that_cannot_be_started(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_PYTHON", str(project / "no-such-python"))
    path = project / "tests" / "test_x.py"
    path.write_text(GOOD)
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out is not None
    assert "audit_tests.py could not be started" in out["hookSpecificOutput"]["additionalContext"]


def test_post_edit_reports_a_missing_audit_script(project):
    (project / ".claude" / "hooks" / "audit_tests.py").unlink()
    path = project / "tests" / "test_x.py"
    path.write_text(GOOD)
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out is not None
    assert "audit_tests.py was not found" in out["hookSpecificOutput"]["additionalContext"]


def stop_event(**extra):
    return {"hook_event_name": "Stop", "session_id": "s1", "stop_hook_active": False, **extra}


def test_stop_gate_blocks_on_a_red_suite(project):
    (project / "tests" / "test_x.py").write_text(RED)
    out = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert out is not None
    assert out["decision"] == "block"
    assert "Fast suite failing" in out["reason"]


def test_stop_gate_passes_green_suite_then_skips_unchanged_tree(project):
    # Outside a git repository the diff guard cannot run: that is a visible warning, not silence.
    (project / "tests" / "test_x.py").write_text(GOOD)
    first = fire(project, "pytest_expert_stop_gate.py", stop_event())
    second = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert first is not None
    assert "decision" not in first
    assert "not a git repository" in first["systemMessage"]
    assert second is None


def test_stop_gate_skips_an_unchanged_tree_without_rerunning_the_suite(project):
    (project / "tests" / "test_x.py").write_text(COUNTING)
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert (project / "tests" / "runs.log").read_text() == "x"


def test_stop_gate_reruns_after_a_py_file_changes(project):
    (project / "tests" / "test_x.py").write_text(COUNTING)
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    (project / "tests" / "test_x.py").write_text(COUNTING + "\n\ndef test_other():\n    assert 2 == 2\n")
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert (project / "tests" / "runs.log").read_text() == "xx"


CONFIG_TEMPLATES = {
    "pyproject.toml": "[tool.coverage.report]\nfail_under = {n}\n",
    "pytest.ini": "[pytest]\nmarkers =\n    m{n}: marker number {n}\n",
    "setup.cfg": "[coverage:report]\nfail_under = {n}\n",
    "tox.ini": "[coverage:report]\nfail_under = {n}\n",
    ".coveragerc": "[report]\nfail_under = {n}\n",
}


@pytest.mark.parametrize("config_name", sorted(CONFIG_TEMPLATES))
def test_stop_gate_reruns_after_a_pytest_or_coverage_config_changes(project, config_name):
    (project / "tests" / "test_x.py").write_text(COUNTING)
    (project / config_name).write_text(CONFIG_TEMPLATES[config_name].format(n=90))
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    (project / config_name).write_text(CONFIG_TEMPLATES[config_name].format(n=10))
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert (project / "tests" / "runs.log").read_text() == "xx"


NEW_CONFIG_FILES = {
    "pytest.toml": "[pytest]\naddopts = [\"-q\"]\n",
    ".pytest.toml": "[pytest]\naddopts = [\"-q\"]\n",
    ".pytest.ini": "[pytest]\naddopts = -q\n",
    "pytest.ini": "[pytest]\naddopts = -q\n",
    ".coveragerc.toml": "[report]\nfail_under = 0\n",
    ".coveragerc": "[report]\nfail_under = 0\n",
    "setup.cfg": "[coverage:report]\nfail_under = 0\n",
    "tox.ini": "[coverage:report]\nfail_under = 0\n",
    "pyproject.toml": "[tool.coverage.report]\nfail_under = 0\n",
}


@pytest.mark.parametrize("config_name", sorted(NEW_CONFIG_FILES))
# addopts is harmless here: the point is that the file is hashed, and a deselect would also hide the marker run.
def test_stop_gate_does_not_skip_after_a_config_file_is_created(project, config_name):
    (project / "tests" / "test_x.py").write_text(COUNTING)
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert (project / "tests" / "runs.log").read_text() == "x"
    (project / config_name).write_text(NEW_CONFIG_FILES[config_name])
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert (project / "tests" / "runs.log").read_text() != "x", f"{config_name} was not fingerprinted"


@pytest.mark.parametrize("name", ["notes.txt", "pytest.ini.bak", "coveragerc.toml", "data.json"])
def test_stop_gate_still_skips_when_an_unrelated_non_python_file_changes(project, name):
    (project / "tests" / "test_x.py").write_text(COUNTING)
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    (project / name).write_text("addopts = --deselect x\n")
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert (project / "tests" / "runs.log").read_text() == "x"


def _tuple_constant(path: Path, name: str) -> set[str]:
    import ast

    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not found in {path}")


def test_hook_config_list_matches_the_scripts_lists():
    scripts = SKILL_ROOT / "scripts"
    expected = (_tuple_constant(scripts / "diff_guard.py", "PYTEST_CONFIG_ORDER")
                | _tuple_constant(scripts / "diff_guard.py", "COVERAGE_CONFIG_ORDER")
                | _tuple_constant(scripts / "audit_tests.py", "CONFIG_NAMES"))
    assert set(load_common().CONFIG_FILES) == expected


def test_stop_gate_allows_a_green_suite_silently_when_stop_hook_active(project):
    # Outside git the first green run warns that the diff guard did not run (not a block); an unchanged green tree is then silent.
    (project / "tests" / "test_x.py").write_text(GOOD)
    first = fire(project, "pytest_expert_stop_gate.py", stop_event(stop_hook_active=True))
    assert first is not None
    assert set(first) == {"systemMessage"}
    assert "not a git repository" in first["systemMessage"]
    assert fire(project, "pytest_expert_stop_gate.py", stop_event(stop_hook_active=True)) is None


def active_block_prefix(n: int, limit: int) -> str:
    return (f"pytest-expert: block {n} of {limit}; the gate stops blocking after block {limit} "
            "(Claude Code reports that a Stop hook already continued this turn).\n")


def test_red_suite_blocks_through_the_whole_budget_when_stop_hook_active(project, monkeypatch):
    # The gate's own counter is the loop guard: stop_hook_active=true is not a reason to let a red suite stop.
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "3")
    (project / "tests" / "test_x.py").write_text(RED)
    outs = [fire(project, "pytest_expert_stop_gate.py", stop_event(stop_hook_active=True)) for _ in range(4)]
    for n, out in enumerate(outs[:3], start=1):
        assert out is not None
        assert set(out) == {"decision", "reason"}
        assert out["decision"] == "block"
        assert out["reason"].startswith(active_block_prefix(n, 3) + "pytest-expert: do not stop yet.")
        assert "Fast suite failing:\n" in out["reason"]
    last = outs[3]
    assert last is not None
    assert set(last) == {"systemMessage"}
    assert last["systemMessage"].startswith(
        "pytest-expert stop gate gave up after 3 attempts and let Claude stop. The last problems it saw:\nFast suite failing:\n")


def test_block_reason_has_no_progress_line_when_stop_hook_active_is_false(project):
    (project / "tests" / "test_x.py").write_text(RED)
    out = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert out is not None
    assert out["reason"].startswith("pytest-expert: do not stop yet.")
    assert "block 1 of" not in out["reason"]


def test_block_counter_resets_after_a_green_run_with_stop_hook_active(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "2")
    test_file = project / "tests" / "test_x.py"
    test_file.write_text(RED)
    first = fire(project, "pytest_expert_stop_gate.py", stop_event(stop_hook_active=True))
    assert first is not None and first["reason"].startswith(active_block_prefix(1, 2))
    test_file.write_text(GOOD + "\n# fixed\n")
    assert "decision" not in (fire(project, "pytest_expert_stop_gate.py", stop_event(stop_hook_active=True)) or {})
    assert json.loads(state_file_for(project, stop_event()).read_text())["blocks"] == 0
    test_file.write_text(RED + "\n# red again\n")
    again = fire(project, "pytest_expert_stop_gate.py", stop_event(stop_hook_active=True))
    assert again is not None and again["reason"].startswith(active_block_prefix(1, 2))


def test_max_blocks_one_gives_exactly_one_block_when_stop_hook_active(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "1")
    (project / "tests" / "test_x.py").write_text(RED)
    outs = [fire(project, "pytest_expert_stop_gate.py", stop_event(stop_hook_active=True)) for _ in range(2)]
    assert outs[0] is not None and outs[0]["decision"] == "block"
    assert outs[0]["reason"].startswith(active_block_prefix(1, 1))
    assert outs[1] is not None and set(outs[1]) == {"systemMessage"}
    assert "gave up after 1 attempts" in outs[1]["systemMessage"]


def test_max_blocks_above_the_platform_cap_is_lowered_to_it(project, monkeypatch):
    # Claude Code overrides a Stop block after CLAUDE_CODE_STOP_HOOK_BLOCK_CAP (default 8) in a row; the gate must give up first.
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "20")
    monkeypatch.setenv("CLAUDE_CODE_STOP_HOOK_BLOCK_CAP", "2")
    (project / "tests" / "test_x.py").write_text(RED)
    outs = [fire(project, "pytest_expert_stop_gate.py", stop_event(stop_hook_active=True)) for _ in range(3)]
    assert [o is not None and "decision" in o for o in outs] == [True, True, False]
    assert outs[0] is not None and outs[0]["reason"].startswith(active_block_prefix(1, 2))
    assert outs[2] is not None and "gave up after 2 attempts" in outs[2]["systemMessage"]
    assert "CLAUDE_CODE_STOP_HOOK_BLOCK_CAP=2" in outs[2]["systemMessage"]


def test_platform_cap_zero_does_not_lower_max_blocks(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "10")
    monkeypatch.setenv("CLAUDE_CODE_STOP_HOOK_BLOCK_CAP", "0")
    (project / "tests" / "test_x.py").write_text(RED)
    out = fire(project, "pytest_expert_stop_gate.py", stop_event(stop_hook_active=True))
    assert out is not None and out["reason"].startswith(active_block_prefix(1, 10))


def test_stop_gate_gives_up_after_max_blocks(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "2")
    (project / "tests" / "test_x.py").write_text(RED)
    outcomes = [fire(project, "pytest_expert_stop_gate.py", stop_event()) for _ in range(3)]
    assert all(o is not None for o in outcomes)
    assert [("decision" in o, "systemMessage" in o) for o in outcomes if o is not None] == [(True, False), (True, False), (False, True)]


def git(project, *args):
    subprocess.run(["git", *args], cwd=project, check=True, capture_output=True)


@pytest.fixture
def git_project(project, monkeypatch):
    # A missing git must fail loudly: skipping would hide that the diff-guard tests never ran.
    assert shutil.which("git") is not None, "git is required for the diff-guard hook tests"
    # Ignore the machine's git config (signing, hooks, templates); configure the repo locally instead.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    (project / "tests" / "test_x.py").write_text(GOOD + "\n\ndef test_red():\n    assert 1 + 1 == 3\n")
    (project / ".gitignore").write_text(".claude/\nstate/\n")
    git(project, "init", "-q", "-b", "main")
    for key, value in (("user.name", "t"), ("user.email", "t@example.com"), ("commit.gpgsign", "false"), ("tag.gpgsign", "false")):
        git(project, "config", key, value)
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "base")
    return project


def test_stop_gate_blocks_when_the_failing_test_was_deleted(git_project):
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    out = fire(git_project, "pytest_expert_stop_gate.py", stop_event())
    assert out is not None
    assert out["decision"] == "block"
    assert "TEST_DELETED" in out["reason"]


def test_stop_gate_blocks_when_the_failing_test_was_skipped(git_project):
    (git_project / "tests" / "test_x.py").write_text(
        "import pytest\n" + GOOD + "\n\n@pytest.mark.skip(reason='later')\ndef test_red():\n    assert 1 + 1 == 3\n"
    )
    out = fire(git_project, "pytest_expert_stop_gate.py", stop_event())
    assert out is not None
    assert "SKIP_ADDED" in out["reason"]


def test_session_start_base_survives_a_commit_of_the_weakened_tests(git_project):
    assert fire(git_project, "pytest_expert_session_start.py", {"hook_event_name": "SessionStart", "session_id": "s1", "source": "startup"}) is None
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    git(git_project, "commit", "-q", "-am", "make it green")
    out = fire(git_project, "pytest_expert_stop_gate.py", stop_event())
    assert out is not None
    assert "TEST_DELETED" in out["reason"]
    base = json.loads(state_file_for(git_project, stop_event()).read_text())["base_sha"]
    assert f"Test-weakening edits since {base[:12]} (" in out["reason"]


def test_diff_allow_env_lets_an_approved_deletion_through(git_project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_DIFF_ALLOW", "TEST_DELETED")
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    assert fire(git_project, "pytest_expert_stop_gate.py", stop_event()) is None


def test_subagent_stop_event_is_gated_the_same_way(git_project):
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    out = fire(git_project, "pytest_expert_stop_gate.py", {**stop_event(), "hook_event_name": "SubagentStop"})
    assert out is not None
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
    assert out is not None
    assert "pytest-expert-escalation.md" in out["reason"]
    assert "Do not edit, skip, or delete a failing test" in out["reason"]


def test_complete_escalation_note_lets_claude_stop_and_reaches_the_user(project):
    (project / "tests" / "test_x.py").write_text(RED)
    (project / ".claude" / "pytest-expert-escalation.md").write_text(NOTE)
    out = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert out is not None
    assert "decision" not in out
    assert "Claude escalated" in out["systemMessage"]
    assert "Recommendation: fix the expected value to 2" in out["systemMessage"]
    assert "Fast suite failing" in out["systemMessage"]
    assert not (project / ".claude" / "pytest-expert-escalation.md").exists()
    assert len(list((project / ".claude" / "pytest-expert-escalations").glob("*.md"))) == 1


def test_an_escalation_resets_the_block_counter(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "2")
    (project / "tests" / "test_x.py").write_text(RED)
    assert gate(project)["decision"] == "block"  # type: ignore[index]  # one block counted
    (project / ".claude" / "pytest-expert-escalation.md").write_text(NOTE)
    escalated = gate(project)
    assert escalated is not None and "Claude escalated" in escalated["systemMessage"]
    # With the counter back at 0 two more blocks fit before the gate gives up.
    assert [gate(project)["decision"] for _ in range(2)] == ["block", "block"]  # type: ignore[index]


def test_escalation_is_one_shot(project):
    (project / "tests" / "test_x.py").write_text(RED)
    (project / ".claude" / "pytest-expert-escalation.md").write_text(NOTE)
    fire(project, "pytest_expert_stop_gate.py", stop_event())
    second = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert second is not None
    assert second["decision"] == "block"


def test_a_second_escalation_is_archived_next_to_the_first(project):
    (project / "tests" / "test_x.py").write_text(RED)
    for _ in range(2):
        (project / ".claude" / "pytest-expert-escalation.md").write_text(NOTE)
        out = fire(project, "pytest_expert_stop_gate.py", stop_event())
        assert out is not None
        assert "Claude escalated" in out["systemMessage"]
    assert len(list((project / ".claude" / "pytest-expert-escalations").glob("*.md"))) == 2


def test_incomplete_escalation_note_is_sent_back(project):
    (project / "tests" / "test_x.py").write_text(RED)
    (project / ".claude" / "pytest-expert-escalation.md").write_text("Test: tests/test_x.py::test_red\nProblem: wrong\n")
    out = fire(project, "pytest_expert_stop_gate.py", stop_event())
    assert out is not None
    assert out["decision"] == "block"
    assert "missing: issue type:, tried:, recommendation:" in out["reason"]
    assert out["reason"].index("Escalation note") < out["reason"].index("Fast suite failing")  # the note's gaps come first


# --- stop gate: every "could not check" outcome is visible, none is silently green ----------------


def gate(project, **extra):
    return fire(project, "pytest_expert_stop_gate.py", stop_event(**extra))


def test_stop_gate_runs_tests_found_in_a_test_directory_not_named_tests(project):
    (project / "tests").rmdir()
    (project / "test").mkdir()
    (project / "test" / "test_x.py").write_text(RED)
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"
    assert "Fast suite failing" in out["reason"]


def test_stop_gate_runs_tests_found_in_the_project_root(project):
    (project / "tests").rmdir()
    (project / "test_root.py").write_text(RED)
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"
    assert "Fast suite failing" in out["reason"]


def test_stop_gate_audits_tests_found_outside_tests(project):
    (project / "tests").rmdir()
    (project / "test_root.py").write_text(BAD_AUDIT)
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"
    assert "Audit findings" in out["reason"]
    assert "NO_ASSERT" in out["reason"]


def test_stop_gate_blocks_on_audit_findings_even_when_the_suite_is_green(project):
    (project / "tests" / "test_x.py").write_text(BAD_AUDIT)
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"
    assert out["reason"].count("Audit findings:") == 1
    assert "Fast suite failing" not in out["reason"]


def test_stop_gate_says_when_no_tests_were_collected(project):
    (project / "tests" / "test_x.py").write_text("VALUE = 1\n")
    first = gate(project)
    second = gate(project)
    assert first is not None and second is not None
    assert "decision" not in first
    assert "decision" not in second
    assert "no tests were collected" in first["systemMessage"]
    assert "no tests were collected" in second["systemMessage"]
    # Nothing was checked, so the tree must not be remembered as green: the second run warns again.


def test_an_audit_with_nothing_to_audit_is_a_warning_when_pytest_collected_no_tests_either(project):
    # audit_tests.py exits 2 when it finds no test files; with no tests anywhere that is not a second failure.
    (project / ".claude" / "hooks" / "audit_tests.py").write_text("import sys\nprint('no test files found', file=sys.stderr)\nsys.exit(2)\n")
    (project / "tests" / "test_x.py").write_text("VALUE = 1\n")
    out = gate(project)
    assert out is not None
    assert "decision" not in out
    assert "audit_tests.py could not check (exit 2): no test files found" in out["systemMessage"]
    assert "no tests were collected" in out["systemMessage"]


def test_stop_gate_blocks_when_every_test_was_deleted(git_project):
    (git_project / "tests" / "test_x.py").unlink()
    out = gate(git_project)
    assert out is not None
    assert out["decision"] == "block"
    assert "TEST_FILE_DELETED" in out["reason"]
    assert "no tests were collected" not in out.get("systemMessage", "")  # a block carries only could-not-check notes


def test_stop_gate_blocks_when_the_diff_base_is_unknown(git_project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_DIFF_BASE", "no-such-ref")
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    out = gate(git_project)
    assert out is not None
    assert out["decision"] == "block"
    assert "diff_guard.py could not check (exit 2)" in out["reason"]
    assert "PYTEST_EXPERT_DIFF_BASE" in out["reason"]
    assert "no-such-ref" in out["reason"]
    assert "could not check" in out["systemMessage"]


def test_stop_gate_is_clean_in_a_git_repo_when_nothing_is_wrong(git_project):
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    git(git_project, "commit", "-q", "-am", "make the suite green")
    assert gate(git_project) is None


@pytest.mark.parametrize("name", ["audit_tests.py", "diff_guard.py"])
def test_stop_gate_blocks_loudly_when_a_checker_script_is_missing(git_project, name):
    (git_project / ".claude" / "hooks" / name).unlink()
    (git_project / "tests" / "test_x.py").write_text(GOOD + "\n\ndef test_red():\n    assert 1 + 1 == 2\n")
    out = gate(git_project)
    assert out is not None
    assert out["decision"] == "block"
    assert f"{name} was not found" in out["reason"]
    assert f"copy pytest-expert/scripts/{name}" in out["reason"]
    assert "could not check" in out["systemMessage"]


def test_hook_in_the_shipped_layout_finds_the_checkers_in_the_skill_scripts_folder(project):
    (project / "tests" / "test_x.py").write_text(BAD_AUDIT)
    proc = run_hook(HOOKS / "pytest_expert_stop_gate.py", json.dumps(stop_event()))
    out = parse_stdout(proc)
    assert out is not None
    assert "Audit findings" in out["reason"]
    assert "was not found" not in out["reason"]


@pytest.mark.parametrize("script, expected", [
    ("import time\ntime.sleep(60)\n", "audit_tests.py did not finish within 1s"),
    ("import sys\nprint('bad arguments', file=sys.stderr)\nsys.exit(2)\n", "audit_tests.py could not check (exit 2): bad arguments"),
    ("raise RuntimeError('audit bug')\n", "audit_tests.py could not check (exit 1)"),
], ids=["timeout", "exit-2", "crash"])
def test_stop_gate_blocks_when_the_audit_cannot_run(project, monkeypatch, script, expected):
    monkeypatch.setenv("PYTEST_EXPERT_CHECK_TIMEOUT", "1")
    (project / ".claude" / "hooks" / "audit_tests.py").write_text(script)
    (project / "tests" / "test_x.py").write_text(GOOD)
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"
    assert expected in out["reason"]
    assert "could not check" in out["systemMessage"]


def test_stop_gate_blocks_when_the_diff_guard_times_out(git_project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_CHECK_TIMEOUT", "1")
    (git_project / ".claude" / "hooks" / "diff_guard.py").write_text("import time\ntime.sleep(60)\n")
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    out = gate(git_project)
    assert out is not None
    assert out["decision"] == "block"
    assert "diff_guard.py did not finish within 1s" in out["reason"]


def test_stop_gate_blocks_when_the_suite_is_too_slow(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_STOP_TIMEOUT", "2")
    (project / "tests" / "test_x.py").write_text("import time\n\n\ndef test_hangs():\n    time.sleep(60)\n    assert True\n")
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"
    assert "Fast suite exceeded 2s; something is hanging or far too slow." in out["reason"]


def test_stop_gate_blocks_when_pytest_itself_cannot_run(project):
    (project / "tests" / "conftest.py").write_text("raise RuntimeError('broken conftest')\n")
    (project / "tests" / "test_x.py").write_text(GOOD)
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"
    assert "Fast suite could not run (pytest exit 4)" in out["reason"]
    assert "broken conftest" in out["reason"]


def test_stop_gate_blocks_when_the_interpreter_cannot_be_started(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_PYTHON", str(project / "no-such-python"))
    (project / "tests" / "test_x.py").write_text(GOOD)
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"
    assert "audit_tests.py could not be started" in out["reason"]
    assert "pytest could not be started with" in out["reason"]
    assert "set PYTEST_EXPERT_PYTHON" in out["reason"]


def test_stop_gate_blocks_when_pytest_is_not_installed(project, monkeypatch):
    fake = project / "fakepython"
    fake.write_text(f"#!/bin/sh\nif [ \"$2\" = pytest ]; then echo 'No module named pytest' >&2; exit 1; fi\nexec {sys.executable} \"$@\"\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PYTEST_EXPERT_PYTHON", str(fake))
    (project / "tests" / "test_x.py").write_text(GOOD)
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"
    assert "pytest is not installed" in out["reason"]
    assert "PYTEST_EXPERT_PYTHON" in out["reason"]


@pytest.mark.parametrize("name, value", [
    ("PYTEST_EXPERT_MAX_BLOCKS", "three"),
    ("PYTEST_EXPERT_MAX_BLOCKS", "0"),
    ("PYTEST_EXPERT_MAX_BLOCKS", "-2"),
    ("PYTEST_EXPERT_STOP_TIMEOUT", "soon"),
    ("PYTEST_EXPERT_STOP_TIMEOUT", "0"),
    ("PYTEST_EXPERT_CHECK_TIMEOUT", "1.5"),
])
def test_stop_gate_survives_a_bad_numeric_setting_and_still_runs_the_suite(project, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    (project / "tests" / "test_x.py").write_text(RED)
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"
    assert "Fast suite failing" in out["reason"]
    assert f"{name}={value!r} is not a positive integer" in out["systemMessage"]


def test_stop_gate_uses_the_default_of_three_blocks_when_max_blocks_is_zero(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "0")
    (project / "tests" / "test_x.py").write_text(RED)
    outcomes = [gate(project) for _ in range(4)]
    assert all(o is not None for o in outcomes)
    assert [o.get("decision") for o in outcomes if o is not None] == ["block", "block", "block", None]


def test_give_up_message_names_the_last_problem(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "1")
    (project / "tests" / "test_x.py").write_text(RED)
    gate(project)
    out = gate(project)
    assert out is not None
    assert "gave up after 1 attempts" in out["systemMessage"]
    assert "Fast suite failing" in out["systemMessage"]


def test_giving_up_resets_the_counter_so_the_gate_blocks_again_later(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "1")
    (project / "tests" / "test_x.py").write_text(RED)
    shapes = []
    for _ in range(4):
        out = gate(project)
        assert out is not None
        shapes.append("block" if out.get("decision") == "block" else "gave-up")
    assert shapes == ["block", "gave-up", "block", "gave-up"]


def test_green_run_resets_the_block_counter(project, monkeypatch):
    monkeypatch.setenv("PYTEST_EXPERT_MAX_BLOCKS", "2")
    test_file = project / "tests" / "test_x.py"
    test_file.write_text(RED)
    assert gate(project)["decision"] == "block"  # type: ignore[index]
    test_file.write_text(GOOD)
    # Not git: the diff guard warning is the only output, the suite is green.
    green = gate(project)
    assert green is not None and "decision" not in green
    test_file.write_text(RED + "\n# edit\n")
    assert gate(project)["decision"] == "block"  # type: ignore[index]
    test_file.write_text(RED + "\n# edit again\n")
    out = gate(project)
    assert out is not None
    assert out["decision"] == "block"  # a reset counter has only reached 1 of 2


@pytest.mark.parametrize("stdin_text", ["{not json", "[1, 2]", ""])
def test_stop_gate_reports_an_unreadable_event_instead_of_exiting_silently(project, stdin_text):
    out = parse_stdout(run_hook(project / ".claude" / "hooks" / "pytest_expert_stop_gate.py", stdin_text))
    assert out is not None
    assert "decision" not in out
    assert "pytest-expert stop gate" in out["systemMessage"]
    assert "not a JSON object" in out["systemMessage"]
    assert "gate did not run" in out["systemMessage"]


def test_stop_gate_turns_an_unexpected_exception_into_a_visible_block(project):
    # A directory where the state file should be: reading yields no state, writing raises.
    event = stop_event()
    state_file = state_file_for(project, event)
    state_file.mkdir()
    try:
        (project / "tests" / "test_x.py").write_text(GOOD)
        proc = run_hook(project / ".claude" / "hooks" / "pytest_expert_stop_gate.py", json.dumps(event))
    finally:
        state_file.rmdir()
    out = parse_stdout(proc)
    assert out is not None
    assert out["decision"] == "block"
    assert "stop gate crashed" in out["reason"]
    assert "IsADirectoryError" in out["reason"]
    assert "could not check" in out["systemMessage"]


def test_stop_gate_truncates_a_huge_failure_report_keeping_the_end(project):
    (project / "tests" / "test_x.py").write_text("def test_big():\n    assert False, 'a' * 30000 + 'THE-END'\n")
    out = gate(project)
    assert out is not None
    common = load_common()
    assert out["reason"].startswith("...(truncated)...\n")
    assert len(out["reason"]) == common.MAX_CHARS + len("...(truncated)...\n")
    assert "THE-END" in out["reason"]


# --- session start --------------------------------------------------------------------------------

START = {"hook_event_name": "SessionStart", "session_id": "s1", "source": "startup"}


def test_session_start_keeps_the_first_recorded_base(git_project):
    state_file = state_file_for(git_project, START)
    assert fire(git_project, "pytest_expert_session_start.py", START) is None
    first = json.loads(state_file.read_text())["base_sha"]
    (git_project / "tests" / "test_x.py").write_text(GOOD)
    git(git_project, "commit", "-q", "-am", "second")
    assert fire(git_project, "pytest_expert_session_start.py", {**START, "source": "resume"}) is None
    assert json.loads(state_file.read_text())["base_sha"] == first
    head = subprocess.run(["git", "-C", str(git_project), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert first != head


def test_session_start_writes_nothing_outside_a_git_repository(project):
    assert fire(project, "pytest_expert_session_start.py", START) is None
    assert not state_file_for(project, START).exists()


@pytest.mark.parametrize("stdin_text", ["{not json", "[1, 2]", ""])
def test_session_start_reports_an_unreadable_event(project, stdin_text):
    out = parse_stdout(run_hook(project / ".claude" / "hooks" / "pytest_expert_session_start.py", stdin_text))
    assert out is not None
    assert "pytest-expert session-start hook" in out["systemMessage"]
    assert "not a JSON object" in out["systemMessage"]


def test_session_start_reports_a_state_file_it_cannot_write(git_project):
    state_file = state_file_for(git_project, START)
    state_file.mkdir()
    try:
        out = fire(git_project, "pytest_expert_session_start.py", START)
    finally:
        state_file.rmdir()
    assert out is not None
    assert "could not record the session base commit" in out["systemMessage"]


# --- _common.py -------------------------------------------------------------------------------------


def test_project_python_prefers_the_override_then_the_project_venv(tmp_path, monkeypatch):
    common = load_common()
    monkeypatch.delenv("PYTEST_EXPERT_PYTHON", raising=False)
    assert common.project_python(tmp_path) == sys.executable
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    assert common.project_python(tmp_path) == str(venv_python)
    dot_venv_python = tmp_path / ".venv" / "bin" / "python"
    dot_venv_python.parent.mkdir(parents=True)
    dot_venv_python.write_text("")
    assert common.project_python(tmp_path) == str(dot_venv_python)
    monkeypatch.setenv("PYTEST_EXPERT_PYTHON", "/opt/special/python")
    assert common.project_python(tmp_path) == "/opt/special/python"


def test_pytest_command_is_unrandomized_unfailfast_free_of_coverage_and_cache(monkeypatch):
    common = load_common()
    cmd = common.pytest_command(sys.executable, "tests/unit")
    assert cmd[:3] == [sys.executable, "-m", "pytest"]
    assert "-x" in cmd
    assert cmd[cmd.index("-p") : cmd.index("-p") + 2] == ["-p", "no:randomly"]
    assert ["-p", "no:cacheprovider"] == cmd[cmd.index("no:cacheprovider") - 1 : cmd.index("no:cacheprovider") + 1]
    assert cmd[-1] == "tests/unit"
    has_cov = importlib.util.find_spec("pytest_cov") is not None
    assert ("--no-cov" in cmd) is has_cov


def test_pytest_command_omits_no_cov_when_pytest_cov_is_missing(tmp_path):
    common = load_common()
    fake = tmp_path / "py"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    assert "--no-cov" not in common.pytest_command(str(fake))


def test_tail_keeps_the_end_and_marks_the_cut():
    common = load_common()
    assert common.tail("short", 10) == "short"
    assert common.tail("0123456789", 10) == "0123456789"
    assert common.tail("0123456789A", 10) == "...(truncated)...\n123456789A"
    assert common.MAX_CHARS < 10_000


@pytest.mark.parametrize(
    ("event", "expected"),
    [("Stop", "block"), ("SubagentStop", "block"), ("PostToolUse", None), ("SessionStart", None)],
)
def test_settings_block_the_stop_when_the_gate_itself_cannot_run(event, expected):
    """A gate whose script is missing or cannot start must not let the stop through.

    Claude Code carries out the action when a hook fails unless the hook sets onFailure to
    "block"; only the two gate events need that, the advisory hooks must not block on failure.
    """
    settings = json.loads((SKILL_ROOT / "assets" / "claude" / "settings.json").read_text())
    hooks = [hook for group in settings["hooks"][event] for hook in group["hooks"]]
    assert [hook.get("onFailure") for hook in hooks] == [expected]


# Follow-up to the sequential mutation run of 2026-11-21.

@pytest.mark.parametrize("file_path", [5, ["tests/test_x.py"], ""])
def test_post_edit_is_silent_when_the_file_path_is_not_a_usable_string(project, file_path):
    event = {"hook_event_name": "PostToolUse", "tool_name": "Write", "tool_input": {"file_path": file_path}}
    assert fire(project, "pytest_expert_post_edit.py", event) is None


def test_post_edit_reports_its_own_crash_with_the_traceback(project):
    path = project / "tests" / "test_x.py"
    path.write_text(GOOD)
    with open(project / ".claude" / "hooks" / "_common.py", "a") as common:
        common.write("\n\ndef find_script(name):\n    raise RuntimeError('planted crash')\n")
    out = fire(project, "pytest_expert_post_edit.py", edit_event(path))
    assert out is not None
    assert out["systemMessage"].startswith("pytest-expert: the post-edit hook crashed, so the edit was not checked:\nTraceback")
    assert "RuntimeError: planted crash" in out["systemMessage"]
    assert "RuntimeError: planted crash" in out["hookSpecificOutput"]["additionalContext"]
