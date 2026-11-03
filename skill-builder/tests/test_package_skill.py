"""package_skill.py builds a zip without junk and proves the zip works in a clean directory."""
from __future__ import annotations

import json
import zipfile

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


def test_runs_the_declared_test_command_from_the_unpacked_zip(tmp_path, run_script):
    made = _with_test_command(tmp_path, "python3 -c 'import os; open(os.environ[\\\"MARK\\\"], \\\"w\\\")'")
    (made / "CHANGELOG.md").write_text("x\n", encoding="utf-8")
    result = run_script("package_skill.py", made, "--out", tmp_path / "out", "--json")
    assert json.loads(result.stdout)["test_command"] is not None


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
