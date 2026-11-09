"""find_candidates.py: lists every line or file an inventory must account for."""

from __future__ import annotations

from conftest import read_jsonl


def by_line(run):
    return {(c["file"], c["line"]): c for c in read_jsonl(run / "candidates.jsonl")}


def test_flags_url_number_and_absolute_lines(run_dir):
    cands = by_line(run_dir)
    assert {"url", "version", "number"} <= set(cands[("SKILL.md", 8)]["reasons"])
    assert "absolute" in cands[("SKILL.md", 9)]["reasons"]


def test_description_is_always_a_candidate(run_dir):
    assert "frontmatter" in by_line(run_dir)[("SKILL.md", 3)]["reasons"]


def test_code_block_is_one_candidate_spanning_its_lines(run_dir):
    block = by_line(run_dir)[("SKILL.md", 11)]
    assert block["end_line"] == 13 and block["reasons"] == ["code-block"]
    assert ("SKILL.md", 12) not in by_line(run_dir)


def test_plain_prose_is_a_candidate_and_headings_are_skipped(run_dir):
    cands = by_line(run_dir)
    assert cands[("SKILL.md", 15)]["reasons"] == ["prose"]
    assert ("SKILL.md", 6) not in cands


def test_each_script_is_a_file_level_candidate(run_dir):
    assert by_line(run_dir)[("scripts/tool.py", 1)]["reasons"] == ["script"]


def test_ids_are_unique(run_dir):
    ids = [c["id"] for c in read_jsonl(run_dir / "candidates.jsonl")]
    assert len(ids) == len(set(ids))


def write_skill(tmp_path, text, name="SKILL.md"):
    import json as _json

    skill = tmp_path / "s"
    skill.mkdir(exist_ok=True)
    (skill / name).write_text(text, encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "meta.json").write_text(_json.dumps({"skill_path": str(skill)}))
    from conftest import run_script

    assert run_script("find_candidates.py", run).returncode == 0
    return by_line(run)


def test_four_backtick_fence_is_not_closed_by_three(tmp_path):
    cands = write_skill(tmp_path, "intro line\n````md\n```bash\nx\n```\n````\nafter line\n")
    assert cands[("SKILL.md", 2)]["end_line"] == 6
    assert ("SKILL.md", 7) in cands


def test_folded_description_spans_all_its_lines(tmp_path):
    cands = write_skill(tmp_path, "---\nname: s\ndescription: >\n  first part\n  second part\nother: 1\n---\n")
    assert cands[("SKILL.md", 3)]["end_line"] == 5


def test_rst_and_txt_are_read_line_by_line(tmp_path):
    cands = write_skill(tmp_path, "A fact about teardown order.\n\nAnother fact.\n", name="notes.rst")
    assert ("notes.rst", 1) in cands and ("notes.rst", 3) in cands
