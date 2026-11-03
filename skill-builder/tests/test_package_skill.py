"""package_skill.py builds a zip without junk and proves the zip works in a clean directory."""
from __future__ import annotations

import json
import zipfile

import pytest
from conftest import write_skill


def _with_test_command(tmp_path, command):
    made = write_skill(tmp_path)
    text = (made / "SKILL.md").read_text()
    (made / "SKILL.md").write_text(text.replace("---\n\n#", f'metadata:\n  test-command: "{command}"\n---\n\n#', 1), encoding="utf-8")
    return made


def test_zip_contains_the_folder_and_excludes_junk(skill, tmp_path, run_script):
    (skill / ".DS_Store").write_text("x", encoding="utf-8")
    (skill / "__pycache__").mkdir()
    (skill / "__pycache__" / "a.pyc").write_text("x", encoding="utf-8")
    out = tmp_path / "out"
    result = run_script("package_skill.py", skill, "--out", out)
    assert result.returncode == 0, result.stdout + result.stderr
    names = zipfile.ZipFile(out / "demo-skill.zip").namelist()
    assert "demo-skill/SKILL.md" in names
    assert not [n for n in names if ".DS_Store" in n or "__pycache__" in n]


def test_keeps_dotfiles_that_are_part_of_the_skill(skill, tmp_path, run_script):
    (skill / "assets").mkdir()
    (skill / "assets" / ".pre-commit-config.yaml").write_text("x", encoding="utf-8")
    out = tmp_path / "out"
    run_script("package_skill.py", skill, "--out", out)
    assert "demo-skill/assets/.pre-commit-config.yaml" in zipfile.ZipFile(out / "demo-skill.zip").namelist()


def test_tests_run_in_the_unpacked_copy_not_the_source(tmp_path, run_script):
    marker = tmp_path / "ran_in.txt"
    made = _with_test_command(tmp_path, f"pwd > {marker}")
    result = run_script("package_skill.py", made, "--out", tmp_path / "out", "--json")
    assert json.loads(result.stdout)["tests"] == "passed"
    assert marker.read_text().strip() != str(made.resolve())
    assert marker.read_text().strip().endswith("demo-skill")


def test_a_test_needing_an_excluded_file_fails(tmp_path, run_script):
    made = _with_test_command(tmp_path, "test -f .DS_Store")
    (made / ".DS_Store").write_text("x", encoding="utf-8")
    assert run_script("package_skill.py", made, "--out", tmp_path / "out").returncode == 1


def test_no_zip_is_left_when_tests_fail(tmp_path, run_script):
    made = _with_test_command(tmp_path, "false")
    out = tmp_path / "out"
    run_script("package_skill.py", made, "--out", out)
    assert not (out / "demo-skill.zip").exists()


def test_stale_zip_is_removed_when_a_later_run_fails(skill, tmp_path, run_script):
    out = tmp_path / "out"
    run_script("package_skill.py", skill, "--out", out)
    assert (out / "demo-skill.zip").exists()
    (skill / "evals" / "evals.json").write_text("not json", encoding="utf-8")
    assert run_script("package_skill.py", skill, "--out", out).returncode == 1
    assert not (out / "demo-skill.zip").exists()


def test_todo_blocks_packaging(skill, tmp_path, run_script):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "\nTODO write this.\n", encoding="utf-8")
    assert run_script("package_skill.py", skill, "--out", tmp_path / "out").returncode == 1


@pytest.mark.parametrize("command", ["python3 -c 'import sys; sys.stdout.buffer.write(bytes([255]))'; exit 1"])
def test_undecodable_test_output_still_reports_failure(tmp_path, run_script, command):
    made = write_skill(tmp_path)
    text = (made / "SKILL.md").read_text()
    (made / "SKILL.md").write_text(text.replace("---\n\n#", f'metadata:\n  test-command: "{command}"\n---\n\n#', 1), encoding="utf-8")
    result = run_script("package_skill.py", made, "--out", tmp_path / "out")
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "out" / "demo-skill.zip").exists()


def test_non_string_test_command_could_not_be_run(tmp_path, run_script):
    pytest.importorskip("yaml")
    made = write_skill(tmp_path)
    text = (made / "SKILL.md").read_text()
    (made / "SKILL.md").write_text(text.replace("---\n\n#", "metadata:\n  test-command: 123\n---\n\n#", 1), encoding="utf-8")
    result = run_script("package_skill.py", made, "--out", tmp_path / "out")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_symlink_out_of_the_skill_is_refused(skill, tmp_path, run_script):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")
    (skill / "leak.txt").symlink_to(secret)
    out = tmp_path / "out"
    result = run_script("package_skill.py", skill, "--out", out)
    assert result.returncode == 1
    assert "leak.txt" in result.stdout
    assert not (out / "demo-skill.zip").exists()


@pytest.mark.parametrize("excluded", [".env", ".env.local", ".git/config", ".venv/x", "node_modules/y", "a.pyc"])
def test_secrets_and_environments_stay_out_of_the_zip(skill, tmp_path, run_script, excluded):
    path = skill / excluded
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    out = tmp_path / "out"
    run_script("package_skill.py", skill, "--out", out)
    assert not [n for n in zipfile.ZipFile(out / "demo-skill.zip").namelist() if excluded in n]


def test_output_folder_inside_the_skill_is_refused(skill, run_script):
    assert run_script("package_skill.py", skill, "--out", skill).returncode == 2


def test_exec_bits_survive_the_unpacked_copy(tmp_path, run_script):
    made = _with_test_command(tmp_path, "./run.sh")
    script = made / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    assert run_script("package_skill.py", made, "--out", tmp_path / "out").returncode == 0


def test_failing_test_command_fails_the_package(tmp_path, run_script):
    made = _with_test_command(tmp_path, "false")
    result = run_script("package_skill.py", made, "--out", tmp_path / "out")
    assert result.returncode == 1
    assert "tests failed" in result.stdout


def test_test_command_that_needs_a_missing_file_fails(tmp_path, run_script):
    made = _with_test_command(tmp_path, "test -f needed.txt")
    (made / "needed.txt").write_text("x", encoding="utf-8")
    assert run_script("package_skill.py", made, "--out", tmp_path / "out").returncode == 0
    (made / "needed.txt").unlink()
    assert run_script("package_skill.py", made, "--out", tmp_path / "out").returncode == 1


def test_no_test_command_is_reported_not_passed_silently(skill, tmp_path, run_script):
    result = run_script("package_skill.py", skill, "--out", tmp_path / "out")
    assert result.returncode == 0
    assert "tests not run" in result.stdout


def test_checker_errors_fail_the_package(tmp_path, run_script):
    bad = write_skill(tmp_path, description="<bad>")
    result = run_script("package_skill.py", bad, "--out", tmp_path / "out")
    assert result.returncode == 1
    assert not (tmp_path / "out" / "demo-skill.zip").exists()


def test_not_a_skill_could_not_be_packaged(tmp_path, run_script):
    assert run_script("package_skill.py", tmp_path, "--out", tmp_path / "out").returncode == 2
