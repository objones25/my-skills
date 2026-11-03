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
    m = re.match(r"^---\nname: ([a-z0-9-]+)\ndescription: (.+?)\n---\n", SKILL, re.DOTALL)
    assert m, "SKILL.md must open with name and description frontmatter"
    assert m.group(1) == "pytest-expert"
    assert len(m.group(2)) <= 1024
    assert "<" not in m.group(2)
    assert ">" not in m.group(2)


def test_skill_md_is_short():
    assert len(SKILL.splitlines()) <= 260


@pytest.mark.parametrize("md", MD_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_em_dashes(md):
    assert "—" not in md.read_text()


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
