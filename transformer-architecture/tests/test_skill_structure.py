"""Structural checks on the skill itself: paths resolve, citations are registered, evals and
examples are well formed, and house style holds. These catch drift when files are edited
separately: a renamed file still linked, an arXiv id cited but missing from sources.md."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MD_FILES = [ROOT / "SKILL.md", *sorted((ROOT / "references").rglob("*.md")),
            *sorted((ROOT / "examples").glob("*.md")), *sorted((ROOT / "assets").rglob("*.md"))]
PATH_RE = re.compile(r"`((?:references|scripts|examples|assets|tests|evals)/[A-Za-z0-9_./-]+)`")


def _ids(p: Path) -> str:
    return str(p.relative_to(ROOT))


@pytest.mark.parametrize("md", MD_FILES, ids=_ids)
def test_backticked_paths_exist(md):
    missing = sorted({m.group(1).rstrip(".,") for m in PATH_RE.finditer(md.read_text())
                      if not (ROOT / m.group(1).rstrip(".,")).exists()})
    assert not missing, f"{md.name} mentions missing files: {missing}"


def test_frontmatter():
    m = re.match(r"^---\nname: ([a-z0-9-]+)\ndescription: (.+?)\n", (ROOT / "SKILL.md").read_text(), re.S)
    assert m and m.group(1) == "transformer-architecture"
    assert len(m.group(2)) <= 1024 and "<" not in m.group(2) and ">" not in m.group(2)


def test_skill_md_stays_short():
    assert len((ROOT / "SKILL.md").read_text().splitlines()) <= 400


@pytest.mark.parametrize("md", MD_FILES, ids=_ids)
def test_no_em_dashes(md):
    assert "—" not in md.read_text(), f"em dash in {md.name}"


def test_every_reference_file_is_routed_from_skill_md():
    skill = (ROOT / "SKILL.md").read_text()
    for p in (ROOT / "references").glob("*.md"):
        assert f"references/{p.name}" in skill, p.name


def test_every_script_has_a_test_file():
    scripts = {p.stem for p in (ROOT / "scripts").glob("*.py")}
    tests = {p.stem.removeprefix("test_") for p in (ROOT / "tests").glob("test_*.py")}
    assert scripts <= tests, f"untested scripts: {sorted(scripts - tests)}"


def test_every_arxiv_id_cited_in_references_is_registered_in_sources():
    sources = (ROOT / "references" / "sources.md").read_text()
    registered = set(re.findall(r"\b(\d{4}\.\d{4,5})\b", sources))
    papers = ROOT / "references" / "papers"
    cited: dict[str, str] = {}
    for p in [*(ROOT / "references").glob("*.md"), ROOT / "SKILL.md"]:
        if p.name == "sources.md":
            continue
        for aid in re.findall(r"\b(2\d{3}\.\d{4,5})\b", p.read_text()):
            cited.setdefault(aid, p.name)
    if papers.is_dir():
        for p in papers.glob("*.md"):
            registered |= set(re.findall(r"\b(\d{4}\.\d{4,5})\b", p.read_text()[:600]))
    missing = {a: f for a, f in cited.items() if a not in registered}
    assert not missing, f"cited but not in sources.md or a paper digest header: {missing}"


def test_evals_well_formed():
    d = json.loads((ROOT / "evals" / "evals.json").read_text())
    assert d["skill_name"] == "transformer-architecture"
    ids = [e["id"] for e in d["evals"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 12
    for e in d["evals"]:
        assert e["prompt"].strip() and len(e["expectations"]) >= 3, e["id"]


def test_examples_are_marked_illustrative_and_indexed():
    index = (ROOT / "examples" / "README.md").read_text()
    for p in (ROOT / "examples").glob("0*.md"):
        assert "*Illustrative" in p.read_text(), p.name
        assert p.name in index, p.name


def test_config_assets_are_valid_json():
    for p in (ROOT / "assets" / "hf-configs").glob("*.json"):
        json.loads(p.read_text())
