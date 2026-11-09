"""plant_errors.py: plants known errors in a copy, never the original, and scores an audit against them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import SKILL_ROOT, run_script, write_jsonl

PLANTS = {"base": "sample", "plants": [
    {"id": "P1", "type": "changed-number", "file": "SKILL.md", "old": "Python 3.13 added", "new": "Python 3.11 added",
     "expect": ["wrong"]},
    {"id": "P2", "type": "script-edge-case", "file": "scripts/tool.py", "old": "print('hi')",
     "new": "print('hi'[5])", "expect": ["wrong"]},
]}


@pytest.fixture
def plants_file(tmp_path: Path) -> Path:
    path = tmp_path / "plants.json"
    path.write_text(json.dumps(PLANTS))
    return path


def test_plants_into_a_new_copy_and_leaves_source_alone(sample_skill, tmp_path, plants_file):
    before = (sample_skill / "SKILL.md").read_text()
    dest = tmp_path / "planted" / "sample"
    proc = run_script("plant_errors.py", "plant", sample_skill, dest, plants_file)
    assert proc.returncode == 0, proc.stderr
    assert "Python 3.11 added" in (dest / "SKILL.md").read_text()
    assert (sample_skill / "SKILL.md").read_text() == before
    manifest = json.loads((dest.parent / "sample.plants.json").read_text())
    assert manifest["P1"] == {"file": "SKILL.md", "line": 8}


def test_refuses_existing_destination(sample_skill, tmp_path, plants_file):
    dest = tmp_path / "exists"
    dest.mkdir()
    assert run_script("plant_errors.py", "plant", sample_skill, dest, plants_file).returncode == 2


def test_refuses_when_a_plant_string_is_not_unique_or_missing(sample_skill, tmp_path):
    bad = {"base": "sample", "plants": [{"id": "P1", "type": "x", "file": "SKILL.md", "old": "not there",
                                         "new": "y", "expect": ["wrong"]}]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    proc = run_script("plant_errors.py", "plant", sample_skill, tmp_path / "out" / "sample", path)
    assert proc.returncode == 2
    assert not (tmp_path / "out" / "sample").exists()


def test_score_counts_detections(sample_skill, tmp_path, plants_file):
    dest = tmp_path / "planted" / "sample"
    run_script("plant_errors.py", "plant", sample_skill, dest, plants_file)
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"skill_path": str(dest)}))
    write_jsonl(run / "items.jsonl", [
        {"id": "I1", "file": "SKILL.md", "line": 8, "verdict": "wrong"},
        {"id": "I2", "file": "scripts/tool.py", "line": 1, "verdict": "correct"},
        {"id": "I3", "file": "SKILL.md", "line": 9, "verdict": "stale"},
    ])
    proc = run_script("plant_errors.py", "score", run, plants_file, "--json")
    assert proc.returncode == 1
    score = json.loads(proc.stdout)
    assert score["detected"] == ["P1"] and score["missed"] == ["P2"]
    assert score["other_flags"] == ["I3"]


def test_pytest_expert_plants_still_apply_to_the_current_skill(tmp_path):
    base = SKILL_ROOT.parent / "pytest-expert"
    if not base.is_dir():
        pytest.skip("pytest-expert is not next to this skill (only true inside the my-skills repo)")
    plants = SKILL_ROOT / "evals" / "files" / "pytest-expert-plants.json"
    proc = run_script("plant_errors.py", "plant", base, tmp_path / "copy" / "pytest-expert", plants)
    assert proc.returncode == 0, proc.stderr


def scored(sample_skill, tmp_path, plants_file, items):
    dest = tmp_path / "planted" / "sample"
    run_script("plant_errors.py", "plant", sample_skill, dest, plants_file)
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"skill_path": str(dest)}))
    write_jsonl(run / "items.jsonl", items)
    return json.loads(run_script("plant_errors.py", "score", run, plants_file, "--json").stdout)


def test_one_huge_item_does_not_detect_every_plant(sample_skill, tmp_path, plants_file):
    score = scored(sample_skill, tmp_path, plants_file,
                   [{"id": "I1", "file": "SKILL.md", "line": 1, "end_line": 10**6, "verdict": "wrong"}])
    assert score["detected"] == []


def test_disputed_counts_only_when_one_side_flagged_it(sample_skill, tmp_path, plants_file):
    score = scored(sample_skill, tmp_path, plants_file, [
        {"id": "I1", "file": "SKILL.md", "line": 8, "verdict": "disputed",
         "dispute": {"auditor": {"verdict": "correct"}, "reviewer": {"verdict": "correct"}}}])
    assert score["detected"] == []
