"""Structural checks on the skill itself: links resolve, citations are registered, evals are well formed.

These catch the drift that creeps in when references are edited separately:
a file renamed but still linked, an arXiv id quoted in a reference but missing
from sources.md, a duplicate eval id.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / "SKILL.md").read_text()
REFS = {p.name: p.read_text() for p in (ROOT / "references").glob("*.md")}
SOURCES = REFS["sources.md"]


def test_frontmatter():
    m = re.match(r"^---\nname: \"?([a-z0-9-]+)\"?\ndescription: (.+?)\n---\n", SKILL, re.S)
    assert m, "frontmatter must open with name and description"
    assert m.group(1) == "building-effective-agents"
    assert len(m.group(2)) <= 1024
    assert "<" not in m.group(2) and ">" not in m.group(2)


def test_every_linked_path_exists():
    linked = set(re.findall(r"`((?:references|scripts|examples|tests|evals)/[\w./-]+)`", SKILL))
    for text in list(REFS.values()) + [p.read_text() for p in (ROOT / "examples").glob("*.md")]:
        linked |= set(re.findall(r"`((?:references|scripts|examples|tests|evals)/[\w./-]+)`", text))
    missing = [p for p in sorted(linked) if not (ROOT / p.split()[0]).exists()]
    assert not missing, missing


def test_bare_reference_names_exist():
    names = set(re.findall(r"`(\w[\w-]*\.md)`", SKILL + "".join(REFS.values())))
    known = set(REFS) | {"SKILL.md", "README.md", "NOTES.md", "CLAUDE.md"}
    unknown = sorted(n for n in names if n not in known and not n[0].isdigit())
    assert not unknown, unknown


def test_every_reference_file_is_listed_in_skill_md():
    for name in REFS:
        if name != "sources.md":
            assert f"references/{name}" in SKILL, name


def test_every_arxiv_id_cited_is_in_sources():
    cited = set()
    for name, text in REFS.items():
        if name != "sources.md":
            cited |= set(re.findall(r"arXiv (\d{4}\.\d{4,5})", text))
    registered = set(re.findall(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", SOURCES))
    missing = sorted(cited - registered)
    assert not missing, f"cited but not in sources.md: {missing}"


def test_evals_well_formed():
    d = json.loads((ROOT / "evals" / "evals.json").read_text())
    ids = [e["id"] for e in d["evals"]]
    assert len(ids) == len(set(ids))
    for e in d["evals"]:
        assert e["prompt"].strip() and len(e["assertions"]) >= 3, e["name"]


def test_examples_are_marked_illustrative():
    for p in (ROOT / "examples").glob("0*.md"):
        assert "*Illustrative" in p.read_text(), p.name
