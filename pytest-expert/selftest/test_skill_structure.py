"""Structural checks on the skill itself: paths resolve, frontmatter valid, style holds, evals well formed."""

from __future__ import annotations

import json
import re

import pytest
from conftest import SKILL_ROOT

ROOT = SKILL_ROOT
SKILL = (ROOT / "SKILL.md").read_text()
MD_FILES = [
    ROOT / "SKILL.md", ROOT / "INSTALL.md", ROOT / "AUDIT.md",
    *sorted((ROOT / "references").rglob("*.md")), *sorted((ROOT / "examples").rglob("*.md")),
]
PATH_RE = re.compile(r"`((?:references|scripts|assets|examples|selftest|evals|papers)/[A-Za-z0-9_./-]+)`")
REF_NAME_RE = re.compile(r"`([a-z0-9-]+\.md)`")


def _exists(md, rel: str) -> bool:
    rel = rel.split("#")[0].rstrip(".,:")
    candidates = [ROOT / rel, ROOT / "references" / rel, md.parent / rel]
    return any(c.exists() for c in candidates)


@pytest.mark.parametrize("md", MD_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_backticked_paths_exist(md):
    """CHANGELOG.md is left out: history names files that were later renamed or removed."""
    missing = sorted({m for m in PATH_RE.findall(md.read_text()) if not _exists(md, m)})
    assert missing == []


def test_changelog_has_no_em_dashes():
    assert "\u2014" not in (ROOT / "CHANGELOG.md").read_text()


@pytest.mark.parametrize("md", [ROOT / "SKILL.md", *sorted((ROOT / "references").glob("*.md"))], ids=lambda p: p.name)
def test_bare_reference_names_exist(md):
    known = {p.name for p in (ROOT / "references").rglob("*.md")} | {"SKILL.md", "README.md", "INSTALL.md", "CHANGELOG.md", "AUDIT.md", "CLAUDE.md"}
    unknown = sorted({n for n in REF_NAME_RE.findall(md.read_text()) if n not in known and not n[0].isdigit()})
    assert unknown == []


def test_frontmatter():
    m = re.match(r"^---\nname: ([a-z0-9-]+)\ndescription: (.+?)\n(?:metadata:\n(?:  .+\n)+)?---\n", SKILL, re.DOTALL)
    assert m, "SKILL.md must open with name and description frontmatter"
    assert m.group(1) == "pytest-expert"
    assert len(m.group(2)) <= 1024
    assert "<" not in m.group(2)
    assert ">" not in m.group(2)


def test_test_command_runs_the_selftests():
    """skill-builder's packager runs metadata.test-command in the unpacked copy; without it the zip is never tested."""
    m = re.search(r"^metadata:\n(?:  .+\n)*?  test-command: \"(.+)\"\n", SKILL, re.M)
    assert m, "SKILL.md frontmatter needs metadata.test-command"
    assert "pytest selftest" in m.group(1)
    assert "--with pytest-randomly" in m.group(1), "test_flake_hunt.py skips without pytest-randomly"


def test_trigger_evals_have_both_kinds():
    cases = json.loads((ROOT / "evals" / "trigger_evals.json").read_text())
    assert all(set(c) == {"query", "should_trigger"} for c in cases)
    assert sum(c["should_trigger"] for c in cases) >= 8
    assert sum(not c["should_trigger"] for c in cases) >= 8
    assert len({c["query"] for c in cases}) == len(cases)


def test_skill_md_is_short():
    assert len(SKILL.splitlines()) <= 260


@pytest.mark.parametrize("md", MD_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_em_dashes(md):
    assert "—" not in md.read_text()


def test_install_states_the_same_gate_count_as_the_rules():
    rules = (ROOT / "references" / "rules.md").read_text()
    actual = len(re.findall(r"^### Rule ", rules, re.M))
    stated = re.search(r"(\d+) hard gates", (ROOT / "INSTALL.md").read_text())
    assert stated is not None
    assert int(stated.group(1)) == actual


def test_every_script_has_a_selftest():
    scripts = {p.stem for p in (ROOT / "scripts").glob("*.py")}
    tests = {p.stem.removeprefix("test_") for p in (ROOT / "selftest").glob("test_*.py")}
    assert scripts <= tests, f"scripts without a selftest: {sorted(scripts - tests)}"


def test_every_reference_file_is_routed_from_skill_md():
    unrouted = [p.name for p in (ROOT / "references").glob("*.md") if f"references/{p.name}" not in SKILL]
    assert unrouted == []
    assert "references/papers/README.md" in SKILL


def test_every_paper_digest_is_indexed():
    index = (ROOT / "references" / "papers" / "README.md").read_text()
    unindexed = [p.name for p in (ROOT / "references" / "papers").glob("*.md") if p.name != "README.md" and f"`{p.name}`" not in index]
    assert unindexed == []


EVALS = json.loads((ROOT / "evals" / "evals.json").read_text())


def test_evals_header_and_unique_ids():
    assert EVALS["skill_name"] == "pytest-expert"
    ids = [e["id"] for e in EVALS["evals"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", EVALS["evals"], ids=lambda e: e["id"])
def test_eval_case_well_formed(case):
    assert case["prompt"].strip()
    assert 3 <= len(case["expectations"]) <= 5


@pytest.mark.parametrize("example", sorted((ROOT / "examples" / "diagnoses").glob("0*.md")), ids=lambda p: p.name)
def test_examples_are_marked_illustrative(example):
    assert "*Illustrative" in example.read_text()


STEP_HEADING_RE = re.compile(r"^\*\*(\d+)\. ([^*]+)\*\*", re.MULTILINE)
STEP_REF_RE = re.compile(r"workflow step (\d+): ([a-z ]+)")


def test_workflow_step_references_match_skill_md():
    """A reference that names a workflow step must name the step SKILL.md gives that number."""
    steps = {int(n): title.lower() for n, title in STEP_HEADING_RE.findall(SKILL)}
    wrong = []
    for md in MD_FILES:
        for number, phrase in STEP_REF_RE.findall(md.read_text()):
            key = next((w for w in phrase.split() if len(w) > 4), phrase)
            if key not in steps.get(int(number), ""):
                wrong.append(f"{md.name}: step {number} is {steps.get(int(number))!r}, not {phrase!r}")
    assert not wrong


def test_mutmut_result_is_stated_the_same_everywhere():
    """The final mutmut run on the wallet example is quoted in two files; they must agree."""
    reference = (ROOT / "references" / "mutation-testing.md").read_text()
    readme = (ROOT / "examples" / "wallet" / "README.md").read_text()
    final = re.findall(r"194/194\s+killed (\d+)\s+survived (\d+)", reference)[-1]
    assert f"{final[0]} of 194 killed" in readme
    assert f"mutmut reports {final[1]} survivors" in readme
    assert f"What the remaining {final[1]} teach" in reference
