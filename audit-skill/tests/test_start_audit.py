"""start_audit.py: creates the run directory, the workspace copy and meta.json, never touching the skill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import run_script


def start(skill: Path, root: Path, *extra: str):
    return run_script("start_audit.py", skill, "--root", root, "--date", "2026-11-07", *extra)


def test_creates_run_dir_with_meta_and_workspace(sample_skill, tmp_path):
    proc = start(sample_skill, tmp_path / "repo", "--scope", "full")
    assert proc.returncode == 0, proc.stderr
    run = Path(proc.stdout.strip().splitlines()[-1])
    assert run == tmp_path / "repo" / "audits" / "sample" / "2026-11-07"
    meta = json.loads((run / "meta.json").read_text())
    assert meta["skill"] == "sample"
    assert meta["scope"] == {"mode": "full", "paths": []}
    assert len(meta["tree_hash"]) == 64
    assert (run / "workspace" / "SKILL.md").read_text() == (sample_skill / "SKILL.md").read_text()
    assert meta["report"] == str(tmp_path / "repo" / "audits" / "sample" / "2026-11-07.md")


def test_second_run_on_same_day_gets_a_suffix(sample_skill, tmp_path):
    start(sample_skill, tmp_path / "repo", "--scope", "full")
    proc = start(sample_skill, tmp_path / "repo", "--scope", "full")
    assert proc.stdout.strip().endswith("2026-11-07-2")


def test_default_scope_is_risk_based(sample_skill, tmp_path):
    proc = start(sample_skill, tmp_path / "repo")
    run = Path(proc.stdout.strip().splitlines()[-1])
    assert json.loads((run / "meta.json").read_text())["scope"]["mode"] == "risk"
    assert "risk-based" in proc.stderr


def test_targeted_scope_requires_existing_paths(sample_skill, tmp_path):
    assert start(sample_skill, tmp_path / "repo", "--scope", "targeted").returncode == 2
    assert start(sample_skill, tmp_path / "repo", "--scope", "targeted", "--paths", "nope.md").returncode == 2
    proc = start(sample_skill, tmp_path / "repo", "--scope", "targeted", "--paths", "scripts")
    assert proc.returncode == 0, proc.stderr


def test_missing_skill_md_exits_2(tmp_path):
    (tmp_path / "empty").mkdir()
    assert start(tmp_path / "empty", tmp_path).returncode == 2


def test_skill_folder_is_unchanged(sample_skill, tmp_path):
    before = {p: p.read_bytes() for p in sample_skill.rglob("*") if p.is_file()}
    start(sample_skill, tmp_path / "repo", "--scope", "full")
    after = {p: p.read_bytes() for p in sample_skill.rglob("*") if p.is_file()}
    assert before == after




def test_no_gitignore_warning_outside_a_git_repo_but_a_note(sample_skill, tmp_path):
    proc = start(sample_skill, tmp_path / "repo", "--scope", "full")
    assert "warning:" not in proc.stderr
    assert "not a git repository" in proc.stderr


def test_warns_when_audits_is_not_gitignored_in_a_repo(sample_skill, tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path / "repo")], check=True)
    proc = start(sample_skill, tmp_path / "repo", "--scope", "full")
    assert "warning:" in proc.stderr and "not ignored" in proc.stderr
    (tmp_path / "repo" / ".gitignore").write_text("/audits/\n")
    proc = start(sample_skill, tmp_path / "repo", "--scope", "full")
    assert "not ignored" not in proc.stderr


def test_skill_that_is_its_own_repo_is_never_written_into(sample_skill, tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(sample_skill)], check=True)
    proc = run_script("start_audit.py", sample_skill, "--scope", "full", "--date", "2026-11-07")
    assert proc.returncode == 0, proc.stderr
    assert not (sample_skill / "audits").exists()
    assert Path(proc.stdout.strip().splitlines()[-1]) == sample_skill.parent / "audits" / "sample" / "2026-11-07"


@pytest.mark.parametrize("raw,stored", [("./scripts", "scripts"), ("scripts/", "scripts"), ("scripts/tool.py", "scripts/tool.py")])
def test_targeted_paths_are_normalized(sample_skill, tmp_path, raw, stored):
    proc = start(sample_skill, tmp_path / "repo", "--scope", "targeted", "--paths", raw)
    run = Path(proc.stdout.strip().splitlines()[-1])
    assert json.loads((run / "meta.json").read_text())["scope"]["paths"] == [stored]


@pytest.mark.parametrize("raw", ["..", "/etc", ".", "../sample/scripts/../../x"])
def test_targeted_paths_outside_the_skill_are_refused(sample_skill, tmp_path, raw):
    assert start(sample_skill, tmp_path / "repo", "--scope", "targeted", "--paths", raw).returncode == 2
