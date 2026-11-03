"""new_skill.py scaffolds a skill that already passes the checker for what it contains."""
from __future__ import annotations

import json


def test_creates_core_files(tmp_path, run_script):
    result = run_script("new_skill.py", "widget-expert", "--dest", tmp_path)
    assert result.returncode == 0, result.stderr
    made = tmp_path / "widget-expert"
    for name in ("SKILL.md", "CHANGELOG.md", "evals/evals.json", "evals/trigger_evals.json"):
        assert (made / name).is_file(), name


def test_fills_in_the_skill_name(tmp_path, run_script):
    run_script("new_skill.py", "widget-expert", "--dest", tmp_path)
    made = tmp_path / "widget-expert"
    assert "name: widget-expert" in (made / "SKILL.md").read_text()
    assert json.loads((made / "evals" / "evals.json").read_text())["skill_name"] == "widget-expert"


def test_leaves_no_placeholder_markers_outside_todos(tmp_path, run_script):
    run_script("new_skill.py", "widget-expert", "--dest", tmp_path)
    assert "{{" not in (tmp_path / "widget-expert" / "SKILL.md").read_text()


def test_scaffold_has_no_checker_errors(tmp_path, run_script):
    run_script("new_skill.py", "widget-expert", "--dest", tmp_path)
    assert run_script("check_skill.py", tmp_path / "widget-expert").returncode == 0


def test_tooling_tier_adds_a_tested_script_layout(tmp_path, run_script):
    run_script("new_skill.py", "widget-expert", "--dest", tmp_path, "--tier", "tooling")
    made = tmp_path / "widget-expert"
    assert (made / "tests" / "test_skill_structure.py").is_file()
    assert "test-command" in (made / "SKILL.md").read_text()


def test_knowledge_tier_adds_a_source_record_and_audit(tmp_path, run_script):
    run_script("new_skill.py", "widget-expert", "--dest", tmp_path, "--tier", "knowledge")
    made = tmp_path / "widget-expert"
    assert (made / "references" / "sources.md").is_file()
    assert (made / "AUDIT.md").is_file()


def test_refuses_to_overwrite(tmp_path, run_script):
    (tmp_path / "widget-expert").mkdir()
    result = run_script("new_skill.py", "widget-expert", "--dest", tmp_path)
    assert result.returncode == 2
    assert list((tmp_path / "widget-expert").iterdir()) == []


def test_rejects_an_invalid_name(tmp_path, run_script):
    assert run_script("new_skill.py", "Widget_Expert", "--dest", tmp_path).returncode == 2


def test_generated_structure_test_passes(tmp_path, run_script):
    import subprocess
    import sys
    run_script("new_skill.py", "widget-expert", "--dest", tmp_path, "--tier", "tooling")
    made = tmp_path / "widget-expert"
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                            cwd=made, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout


def test_every_scaffold_states_a_version(tmp_path, run_script):
    run_script("new_skill.py", "widget-expert", "--dest", tmp_path)
    assert "version: 0.1.0" in (tmp_path / "widget-expert" / "SKILL.md").read_text()


def test_tooling_scaffold_has_version_and_test_command_in_one_metadata_block(tmp_path, run_script):
    run_script("new_skill.py", "widget-expert", "--dest", tmp_path, "--tier", "tooling")
    text = (tmp_path / "widget-expert" / "SKILL.md").read_text()
    assert text.count("metadata:") == 1
    assert "version: 0.1.0" in text and "test-command:" in text
