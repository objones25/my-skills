"""audit_common.py: tree hashing and JSONL helpers shared by the scripts."""

from __future__ import annotations

import audit_common
import pytest


def test_tree_hash_changes_when_a_file_changes(sample_skill):
    before = audit_common.tree_hash(sample_skill)
    (sample_skill / "SKILL.md").write_text("changed", encoding="utf-8")
    assert audit_common.tree_hash(sample_skill) != before


def test_tree_hash_ignores_caches(sample_skill):
    before = audit_common.tree_hash(sample_skill)
    (sample_skill / "__pycache__").mkdir()
    (sample_skill / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (sample_skill / ".DS_Store").write_bytes(b"\0")
    assert audit_common.tree_hash(sample_skill) == before


def test_read_jsonl_reports_the_bad_line(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="x.jsonl:2"):
        audit_common.read_jsonl(path)


def test_read_jsonl_of_missing_file_is_empty(tmp_path):
    assert audit_common.read_jsonl(tmp_path / "none.jsonl") == []


def test_item_hash_is_stable_and_ignores_dispute():
    item = {"id": "I1", "text": "x", "verdict": "correct"}
    assert audit_common.item_hash(item) == audit_common.item_hash(dict(reversed(list(item.items()))))
    assert audit_common.item_hash(item) == audit_common.item_hash({**item, "dispute": {"a": 1}})
    assert audit_common.item_hash(item) != audit_common.item_hash({**item, "verdict": "wrong"})


@pytest.mark.parametrize("change", ["chmod", "node_modules", "nested_venv", "empty_dir"])
def test_tree_hash_sees_changes_reviewers_found_it_missed(sample_skill, change):
    before = audit_common.tree_hash(sample_skill)
    if change == "chmod":
        (sample_skill / "scripts" / "tool.py").chmod(0o755)
    elif change == "node_modules":
        (sample_skill / "node_modules").mkdir()
        (sample_skill / "node_modules" / "x.js").write_text("x")
    elif change == "nested_venv":
        (sample_skill / "scripts" / ".venv").mkdir()
        (sample_skill / "scripts" / ".venv" / "evil.py").write_text("x")
    else:
        (sample_skill / "newdir").mkdir()
    assert audit_common.tree_hash(sample_skill) != before


def test_tree_hash_sees_source_planted_in_pycache(sample_skill):
    before = audit_common.tree_hash(sample_skill)
    (sample_skill / "__pycache__").mkdir()
    (sample_skill / "__pycache__" / "evil.py").write_text("x")
    assert audit_common.tree_hash(sample_skill) != before
