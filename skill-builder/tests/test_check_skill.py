"""check_skill.py: every rule fires on a skill that breaks it, and stays quiet otherwise."""
from __future__ import annotations

import json

import pytest
from conftest import write_skill


def test_conforming_skill_has_no_findings(skill, codes):
    assert codes(skill) == set()


def test_clean_skill_exits_zero(skill, run_script):
    assert run_script("check_skill.py", skill).returncode == 0


def test_error_exits_one(tmp_path, run_script):
    bad = write_skill(tmp_path, description="<bad>")
    assert run_script("check_skill.py", bad).returncode == 1


def test_warning_alone_exits_zero_unless_strict(skill, run_script):
    (skill / "CHANGELOG.md").unlink()
    assert run_script("check_skill.py", skill).returncode == 0
    assert run_script("check_skill.py", skill, "--strict").returncode == 1


def test_folder_without_skill_md_could_not_be_checked(tmp_path, run_script):
    result = run_script("check_skill.py", tmp_path)
    assert result.returncode == 2
    assert "SKILL.md" in result.stderr


def test_list_codes_names_every_rule(run_script):
    listed = run_script("check_skill.py", "--list-codes").stdout
    assert "SB002" in listed and "SB050" in listed


def test_ignore_drops_a_code(skill, codes):
    (skill / "CHANGELOG.md").unlink()
    assert codes(skill, "--ignore", "SB008") == set()


# ------------------------------------------------------------------ frontmatter


def test_sb002_no_frontmatter(tmp_path, codes):
    bad = tmp_path / "demo-skill"
    bad.mkdir()
    (bad / "SKILL.md").write_text("# No frontmatter\n", encoding="utf-8")
    assert "SB002" in codes(bad)


def test_sb002_unquoted_colon_in_a_value(tmp_path, codes):
    bad = write_skill(tmp_path)
    (bad / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Do this: then that\n---\n\nBody\n", encoding="utf-8")
    assert "SB002" in codes(bad)


def test_folded_description_parses(tmp_path, codes):
    good = write_skill(tmp_path)
    (good / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: >\n  Check widgets. Use when asked: is it broken.\n  Not for gadgets.\n---\n\nBody\n",
        encoding="utf-8")
    assert "SB002" not in codes(good)
    assert "SB004" not in codes(good)


def test_sb003_name_differs_from_folder(tmp_path, codes):
    bad = write_skill(tmp_path, name="demo-skill")
    moved = bad.rename(tmp_path / "other-name")
    assert "SB003" in codes(moved)


@pytest.mark.parametrize("name", ["Demo_Skill", "-demo", "demo--skill", "a" * 65])
def test_sb003_invalid_name(tmp_path, codes, name):
    assert "SB003" in codes(write_skill(tmp_path, name=name))


@pytest.mark.parametrize("description", ["", "x <tag> y", "x" * 1025])
def test_sb004_bad_description(tmp_path, codes, description):
    assert "SB004" in codes(write_skill(tmp_path, description=description))


def test_sb005_description_without_a_when_clause(tmp_path, codes):
    assert "SB005" in codes(write_skill(tmp_path, description="A widget tool. Not for gadgets."))


def test_sb006_description_without_negative_scope(tmp_path, codes):
    assert "SB006" in codes(write_skill(tmp_path, description="Use when inspecting widgets."))


@pytest.mark.parametrize("scope", ["Not for gadgets.", "Do not use for gadgets.", "For gadgets use gadget-expert."])
def test_sb006_recognises_negative_scope(tmp_path, codes, scope):
    assert "SB006" not in codes(write_skill(tmp_path, description=f"Use when inspecting widgets. {scope}"))


# ------------------------------------------------------------------ paths


def test_sb007_missing_bundled_path(skill, codes):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "\nRead `references/gone.md`.\n", encoding="utf-8")
    assert "SB007" in codes(skill)


@pytest.mark.parametrize("mention", ["`references/<topic>.md`", "`assets/dockerignore.{python,node}`", "`tests/*.py`"])
def test_sb007_ignores_patterns_and_placeholders(skill, codes, mention):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + f"\nSee {mention}.\n", encoding="utf-8")
    assert "SB007" not in codes(skill)


def test_sb051_bare_filename_missing_in_a_reference(skill, codes):
    refs = skill / "references"
    refs.mkdir()
    (refs / "topic.md").write_text("Also see `gone.md`.\n", encoding="utf-8")
    assert "SB051" in codes(skill)


def test_sb051_bare_filename_that_exists(skill, codes):
    refs = skill / "references"
    refs.mkdir()
    (refs / "topic.md").write_text("Also see `other.md`.\n", encoding="utf-8")
    (refs / "other.md").write_text("Hi\n", encoding="utf-8")
    assert "SB051" not in codes(skill)


# ------------------------------------------------------------------ core files


def test_sb008_no_changelog(skill, codes):
    (skill / "CHANGELOG.md").unlink()
    assert "SB008" in codes(skill)


def test_sb009_no_evals(skill, codes):
    (skill / "evals" / "evals.json").unlink()
    assert "SB009" in codes(skill)


def test_sb010_no_trigger_evals(skill, codes):
    (skill / "evals" / "trigger_evals.json").unlink()
    assert "SB010" in codes(skill)


def test_sb010_trigger_evals_need_both_kinds(skill, codes):
    (skill / "evals" / "trigger_evals.json").write_text(json.dumps([{"query": "a", "should_trigger": True}]), encoding="utf-8")
    assert "SB010" in codes(skill)


@pytest.mark.parametrize("content", [
    "not json",
    json.dumps({"evals": []}),
    json.dumps({"skill_name": "demo-skill", "evals": [{"id": 1, "prompt": "p"}]}),
    json.dumps({"skill_name": "demo-skill", "evals": [{"id": 1, "prompt": "p", "expectations": []}]}),
    json.dumps({"skill_name": "wrong", "evals": [{"id": 1, "prompt": "p", "expectations": ["e"]}]}),
    json.dumps({"skill_name": "demo-skill", "evals": [{"id": 1, "prompt": "p", "expectations": ["e"]},
                                                       {"id": 1, "prompt": "q", "expectations": ["e"]}]}),
])
def test_sb011_invalid_evals(skill, codes, content):
    (skill / "evals" / "evals.json").write_text(content, encoding="utf-8")
    assert "SB011" in codes(skill)


@pytest.mark.parametrize("junk", [".DS_Store", "__pycache__/a.pyc", ".ruff_cache/x", "scripts/.pytest_cache/y"])
def test_sb012_cache_or_junk(skill, codes, junk):
    path = skill / junk
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    assert "SB012" in codes(skill)


def test_sb013_em_dash(skill, codes):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "\nOne — two.\n", encoding="utf-8")
    assert "SB013" in codes(skill)


# ------------------------------------------------------------------ tooling tier


def _add_script(skill):
    (skill / "scripts").mkdir()
    (skill / "scripts" / "tool.py").write_text("print('hi')\n", encoding="utf-8")


def test_sb020_scripts_without_tests(skill, codes):
    _add_script(skill)
    assert "SB020" in codes(skill)


@pytest.mark.parametrize("test_path", ["tests/test_tool.py", "scripts/test_tool.py", "scripts/tests/test_tool.py", "selftest/test_tool.py"])
def test_sb020_accepts_existing_test_locations(skill, codes, test_path):
    _add_script(skill)
    path = skill / test_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def test_x(): pass\n", encoding="utf-8")
    assert "SB020" not in codes(skill)


def test_sb022_tests_without_a_declared_command(skill, codes):
    (skill / "tests").mkdir()
    (skill / "tests" / "test_x.py").write_text("def test_x(): pass\n", encoding="utf-8")
    assert "SB022" in codes(skill)


def test_sb022_satisfied_by_metadata(tmp_path, codes):
    made = write_skill(tmp_path)
    text = (made / "SKILL.md").read_text()
    (made / "SKILL.md").write_text(text.replace("---\n\n#", 'metadata:\n  test-command: "python -m pytest tests"\n---\n\n#', 1), encoding="utf-8")
    (made / "tests").mkdir()
    (made / "tests" / "test_x.py").write_text("def test_x(): pass\n", encoding="utf-8")
    assert "SB022" not in codes(made)


# ------------------------------------------------------------------ knowledge tier


def test_sb030_cites_urls_without_a_source_record(skill, codes):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "\nSee https://example.org/spec.\n", encoding="utf-8")
    assert "SB030" in codes(skill)


def test_sb030_satisfied_by_sources_file(skill, codes):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "\nSee https://example.org/spec.\n", encoding="utf-8")
    (skill / "references").mkdir()
    (skill / "references" / "sources.md").write_text("# Sources\n", encoding="utf-8")
    assert "SB030" not in codes(skill)


def test_sb031_source_record_without_audit(skill, codes):
    (skill / "references").mkdir()
    (skill / "references" / "sources.md").write_text("# Sources\n", encoding="utf-8")
    assert "SB031" in codes(skill)


# ------------------------------------------------------------------ large tier


def test_sb040_reference_not_routed(skill, codes):
    (skill / "references").mkdir()
    (skill / "references" / "orphan.md").write_text("Hi\n", encoding="utf-8")
    assert "SB040" in codes(skill)


def test_sb040_reference_routed(skill, codes):
    (skill / "references").mkdir()
    (skill / "references" / "topic.md").write_text("Hi\n", encoding="utf-8")
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "\nRead `references/topic.md` when X.\n", encoding="utf-8")
    assert "SB040" not in codes(skill)


def test_sb040_digests_may_route_through_their_folder_index(skill, codes):
    papers = skill / "references" / "papers"
    papers.mkdir(parents=True)
    (papers / "README.md").write_text("`one.md`\n", encoding="utf-8")
    (papers / "one.md").write_text("Hi\n", encoding="utf-8")
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "\nSee `references/papers/README.md`.\n", encoding="utf-8")
    assert "SB040" not in codes(skill)


def test_sb041_large_skill_without_a_structure_test(skill, codes):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "line\n" * 160, encoding="utf-8")
    assert "SB041" in codes(skill)


def test_sb042_skill_md_over_the_size_limit(skill, codes):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "line\n" * 500, encoding="utf-8")
    assert "SB042" in codes(skill)


# ------------------------------------------------------------------ machine-specific results


@pytest.mark.parametrize("line", ["All 40 passed.", "Result: 12 passed, 1 skipped", "Tested on Python 3.12."])
def test_sb050_result_without_date_and_machine(skill, codes, line):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + f"\n{line}\n", encoding="utf-8")
    assert "SB050" in codes(skill)


@pytest.mark.parametrize("line", [
    "On 2026-11-03 on macOS arm64: 40 passed.",
    "On 2026-11-03, machine not recorded: 40 passed.",
    "On 2026-11-03 on Linux x86_64, tested on Python 3.12.",
])
def test_sb050_result_with_date_and_machine(skill, codes, line):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + f"\n{line}\n", encoding="utf-8")
    assert "SB050" not in codes(skill)


def test_sb050_result_labelled_by_its_table_header(skill, codes):
    table = "\n| Date | Machine | Result |\n|---|---|---|\n| 2026-11-03 | macOS arm64 | 40 passed |\n"
    (skill / "CHANGELOG.md").write_text((skill / "CHANGELOG.md").read_text() + table, encoding="utf-8")
    assert "SB050" not in codes(skill)


# ------------------------------------------------------------------ calibration against real skills


def test_sb050_date_may_come_from_the_section_heading(skill, codes):
    text = "# Changelog\n\n## 1.0.0, 2026-11-03\n\n| Check | Result |\n|---|---|\n| one | ok |\n| two | ok |\n| three | ok |\n| tests, machine not recorded | 40 passed |\n"
    (skill / "CHANGELOG.md").write_text(text, encoding="utf-8")
    assert "SB050" not in codes(skill)


def test_sb050_machine_may_come_from_the_table_header(skill, codes):
    rows = "".join(f"| 2026-11-0{i} | macOS arm64 | ok |\n" for i in range(1, 6))
    text = f"# Changelog\n\n## 1.0.0\n\n| Date | Machine | Result |\n|---|---|---|\n{rows}| 2026-11-06 | Linux | 40 passed |\n"
    (skill / "CHANGELOG.md").write_text(text, encoding="utf-8")
    assert "SB050" not in codes(skill)


@pytest.mark.parametrize("name", ["CONTRIBUTING.md", "README.md", "NOTES.md", "CLAUDE.md", "AGENTS.md"])
def test_sb051_ignores_files_in_the_users_project(skill, codes, name):
    refs = skill / "references"
    refs.mkdir()
    (refs / "topic.md").write_text(f"Write `{name}` in your repo.\n", encoding="utf-8")
    assert "SB051" not in codes(skill)


def test_sb051_finds_a_file_elsewhere_in_the_skill(skill, codes):
    (skill / "references" / "theory").mkdir(parents=True)
    (skill / "references" / "training").mkdir(parents=True)
    (skill / "references" / "theory" / "mdp.md").write_text("Hi\n", encoding="utf-8")
    (skill / "references" / "training" / "t.md").write_text("See `mdp.md`.\n", encoding="utf-8")
    assert "SB051" not in codes(skill)


def test_sb051_ignores_history_in_a_changelog(skill, codes):
    refs = skill / "references"
    refs.mkdir()
    (refs / "CHANGELOG.md").write_text("Removed `course.md`.\n", encoding="utf-8")
    assert "SB051" not in codes(skill)


def test_sb007_is_a_warning_because_a_path_can_mean_the_users_project(skill, run_script):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "\nCopy to `tests/conftest.py`.\n", encoding="utf-8")
    assert run_script("check_skill.py", skill).returncode == 0


@pytest.mark.parametrize("line", ['A bare "375 passed" tells you little.', "Search for `tested on` in the docs."])
def test_sb050_ignores_quoted_examples(skill, codes, line):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + f"\n{line}\n", encoding="utf-8")
    assert "SB050" not in codes(skill)


def test_sb020_accepts_any_python_file_in_a_tests_folder(skill, codes):
    _add_script(skill)
    (skill / "tests").mkdir()
    (skill / "tests" / "run_tests.py").write_text("print('ok')\n", encoding="utf-8")
    assert "SB020" not in codes(skill)


# ------------------------------------------------------------------ second review


def test_sb014_todo_left_in_skill_or_evals(skill, codes):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "\nTODO write this.\n", encoding="utf-8")
    assert "SB014" in codes(skill)


def test_sb014_ignores_todo_in_code_and_quotes(skill, codes):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + '\nFlag `TODO` comments and "TODO" notes.\n', encoding="utf-8")
    assert "SB014" not in codes(skill)


def test_sb014_finds_todo_in_trigger_evals(skill, codes):
    (skill / "evals" / "trigger_evals.json").write_text(json.dumps([
        {"query": "TODO a request", "should_trigger": True}, {"query": "x", "should_trigger": False}]), encoding="utf-8")
    assert "SB014" in codes(skill)


def test_sb015_unreadable_markdown_is_reported_not_a_crash(skill, run_script):
    (skill / "references").mkdir()
    (skill / "references" / "bad.md").write_bytes(b"\xff\xfe not utf-8")
    result = run_script("check_skill.py", skill, "--json")
    assert "Traceback" not in result.stderr
    assert result.returncode == 1
    assert "SB015" in {f["code"] for f in json.loads(result.stdout)["findings"]}


@pytest.mark.parametrize("name", ["evals.json", "trigger_evals.json"])
def test_sb015_unreadable_evals_are_reported_not_a_crash(skill, run_script, name):
    (skill / "evals" / name).write_bytes(b"\xff\xfe")
    result = run_script("check_skill.py", skill, "--json")
    assert "Traceback" not in result.stderr
    assert "SB015" in {f["code"] for f in json.loads(result.stdout)["findings"]}


def test_binary_files_are_fine(skill, codes):
    (skill / "assets").mkdir()
    (skill / "assets" / "image.png").write_bytes(b"\x89PNG\xff\xfe")
    assert codes(skill) == set()


def test_unreadable_skill_md_could_not_be_checked(tmp_path, run_script):
    made = tmp_path / "demo-skill"
    made.mkdir()
    (made / "SKILL.md").write_bytes(b"\xff\xfe")
    result = run_script("check_skill.py", made)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_unknown_ignore_code_is_a_usage_error(skill, run_script):
    assert run_script("check_skill.py", skill, "--ignore", "BOGUS").returncode == 2


def test_name_with_a_trailing_newline_is_invalid(tmp_path, codes):
    assert "SB003" in codes(write_skill(tmp_path, name="foo\n"))


@pytest.mark.parametrize("name", ["synced", "anthropic-skills"])
def test_sb003_reserved_names(tmp_path, codes, name):
    assert "SB003" in codes(write_skill(tmp_path, name=name))


@pytest.mark.parametrize("cases", [
    [{"prompt": "p", "expectations": ["e"]}],
    [{"id": [1], "prompt": "p", "expectations": ["e"]}],
    [{"id": True, "prompt": "p", "expectations": ["e"]}],
    [{"id": "", "prompt": "p", "expectations": ["e"]}],
])
def test_sb011_id_must_be_an_integer_or_a_name(skill, codes, cases):
    (skill / "evals" / "evals.json").write_text(json.dumps({"skill_name": "demo-skill", "evals": cases}), encoding="utf-8")
    assert "SB011" in codes(skill)


def test_sb011_accepts_named_ids(skill, codes):
    cases = [{"id": "retry-on-429", "prompt": "p", "expectations": ["e"]}]
    (skill / "evals" / "evals.json").write_text(json.dumps({"skill_name": "demo-skill", "evals": cases}), encoding="utf-8")
    assert "SB011" not in codes(skill)


def test_sb041_applies_to_several_references_even_if_short(skill, codes):
    refs = skill / "references"
    refs.mkdir()
    for name in ("one.md", "two.md"):
        (refs / name).write_text("Hi\n", encoding="utf-8")
    assert "SB041" in codes(skill)


def test_sb043_large_skill_without_an_extend_recipe(skill, codes):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "line\n" * 160, encoding="utf-8")
    assert "SB043" in codes(skill)


def test_sb043_satisfied_by_a_recipe(skill, codes):
    (skill / "SKILL.md").write_text((skill / "SKILL.md").read_text() + "line\n" * 160 + "\n## Extend this skill\n", encoding="utf-8")
    assert "SB043" not in codes(skill)


def test_sb050_date_may_come_from_a_parent_heading(skill, codes):
    text = "# Changelog\n\n## 1.0.0, 2026-11-03\n\n### Verification\n\n| Check | Result |\n|---|---|\n| one | ok |\n| two | ok |\n| three | ok |\n| tests, machine not recorded | 40 passed |\n"
    (skill / "CHANGELOG.md").write_text(text, encoding="utf-8")
    assert "SB050" not in codes(skill)


def test_sb050_not_recorded_is_enough_without_a_date(skill, codes):
    (skill / "CHANGELOG.md").write_text("# Changelog\n\n40 passed, date and machine not recorded.\n", encoding="utf-8")
    assert "SB050" not in codes(skill)


def test_sb050_still_needs_the_machine(skill, codes):
    (skill / "CHANGELOG.md").write_text("# Changelog\n\n## 1.0.0, 2026-11-03\n\n40 passed.\n", encoding="utf-8")
    assert "SB050" in codes(skill)


def test_sb050_skips_paper_digests(skill, codes):
    papers = skill / "references" / "papers"
    papers.mkdir(parents=True)
    (papers / "x.md").write_text("The method was tested on a robot arm.\n", encoding="utf-8")
    assert "SB050" not in codes(skill)


def test_json_reports_the_counts(skill, run_script):
    (skill / "CHANGELOG.md").unlink()
    data = json.loads(run_script("check_skill.py", skill, "--json").stdout)
    assert data["errors"] == 0 and data["warnings"] == 1


# ------------------------------------------------------------------ the parser without PyYAML


@pytest.fixture
def fallback(monkeypatch):
    import check_skill
    monkeypatch.setattr(check_skill, "yaml", None)
    return check_skill.parse_frontmatter


def _block(*lines):
    return "---\n" + "\n".join(lines) + "\n---\n\nBody\n"


@pytest.mark.parametrize("lines, key, expected", [
    (["name: demo # note"], "name", "demo"),
    (['description: "use: this" # note'], "description", "use: this"),
    (["description: foo # note: bar"], "description", "foo"),
    (["description: first", "  second"], "description", "first second"),
    (["description: >-", "  one", "  two"], "description", "one two"),
])
def test_fallback_reads_top_level_values(fallback, lines, key, expected):
    fields, problem = fallback(_block(*lines))
    assert problem is None
    assert fields[key] == expected


@pytest.mark.parametrize("lines, expected", [
    (['metadata:', '  test-command: "true" # note'], "true"),
    (["metadata:", "  test-command: >-", "    false"], "false"),
    (["metadata:", "  test-command: a b", "  version: 1"], "a b"),
])
def test_fallback_reads_nested_metadata(fallback, lines, expected):
    fields, problem = fallback(_block(*lines))
    assert problem is None
    assert fields["metadata"]["test-command"] == expected


@pytest.mark.parametrize("lines", [
    ['metadata: {test-command: "true"}'],
    ['description: "starts', '  and continues"'],
    ["metadata:", "  deep:", "    test-command: x"],
])
def test_fallback_refuses_what_it_cannot_read(fallback, lines):
    import check_skill
    with pytest.raises(check_skill.NeedsPyYAML):
        fallback(_block(*lines))


def test_fallback_flags_an_unterminated_quote(fallback):
    _, problem = fallback(_block('name: "demo'))
    assert problem is not None


def test_cli_exits_two_when_frontmatter_needs_pyyaml(tmp_path, run_script):
    import importlib.util
    if importlib.util.find_spec("yaml"):
        pytest.skip("PyYAML is installed, so the fallback parser is not used")
    made = write_skill(tmp_path)
    (made / "SKILL.md").write_text('---\nname: demo-skill\nmetadata: {a: b}\n---\n\nBody\n', encoding="utf-8")
    assert run_script("check_skill.py", made).returncode == 2
