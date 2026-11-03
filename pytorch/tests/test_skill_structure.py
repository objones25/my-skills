"""Structural checks on the skill itself: paths resolve, every reference is routed, every script
is tested, evals and examples are well formed, and house style holds. These catch drift when
files are edited separately: a renamed file still linked, a new script with no test."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MD_FILES = [ROOT / "SKILL.md", *sorted((ROOT / "references").glob("*.md")),
            *sorted((ROOT / "examples").glob("*.md"))]
TEXT_FILES = MD_FILES + sorted(ROOT.glob("*.md")) + sorted((ROOT / "scripts").glob("*.py")) \
    + sorted((ROOT / "assets").glob("*.py"))
PATH_RE = re.compile(r"`((?:references|scripts|examples|assets|tests|evals)/[A-Za-z0-9_./-]+)`")


def _ids(p: Path) -> str:
    return str(p.relative_to(ROOT))


@pytest.mark.parametrize("md", MD_FILES, ids=_ids)
def test_backticked_paths_exist(md):
    missing = sorted({m.group(1).rstrip(".,") for m in PATH_RE.finditer(md.read_text())
                      if not (ROOT / m.group(1).rstrip(".,")).exists()})
    assert not missing, f"{md.name} mentions missing files: {missing}"


def test_frontmatter_name_and_description():
    m = re.match(r"^---\nname: ([a-z0-9-]+)\ndescription: (.+?)\n", (ROOT / "SKILL.md").read_text(), re.S)
    assert m and m.group(1) == "pytorch"
    assert len(m.group(2)) <= 1024 and "<" not in m.group(2) and ">" not in m.group(2)


def test_skill_md_stays_short():
    assert len((ROOT / "SKILL.md").read_text().splitlines()) <= 300


@pytest.mark.parametrize("path", TEXT_FILES, ids=_ids)
def test_no_em_dashes(path):
    assert "\u2014" not in path.read_text(), f"em dash in {path.name}"


@pytest.mark.parametrize("ref", sorted((ROOT / "references").glob("*.md")), ids=_ids)
def test_every_reference_file_is_routed_from_skill_md(ref):
    assert f"references/{ref.name}" in (ROOT / "SKILL.md").read_text()


@pytest.mark.parametrize("ref", sorted((ROOT / "references").glob("[0-9][0-9]-*.md")), ids=_ids)
def test_numbered_reference_follows_the_section_layout(ref):
    headers = re.findall(r"^## (\w[\w ]*)$", ref.read_text(), re.M)
    assert [h for h in headers if h in {"Walkthrough", "Gotchas", "Sources"}] == \
        ["Walkthrough", "Gotchas", "Sources"] or ref.name == "00-module-map.md"


def test_every_script_has_a_test_file():
    scripts = {p.stem for p in (ROOT / "scripts").glob("*.py")}
    tests = {p.stem.removeprefix("test_") for p in (ROOT / "tests").glob("test_*.py")}
    assert scripts <= tests, f"untested scripts: {sorted(scripts - tests)}"


def test_every_script_is_listed_in_skill_md():
    skill = (ROOT / "SKILL.md").read_text()
    assert [p.name for p in (ROOT / "scripts").glob("*.py") if f"scripts/{p.name}" not in skill] == []


def test_every_walkthrough_file_is_executed_by_the_walkthrough_test():
    runner = (ROOT / "tests" / "test_walkthroughs.py").read_text()
    with_walkthrough = [p.name for p in (ROOT / "references").glob("*.md")
                        if re.search(r"^## Walkthrough$", p.read_text(), re.M)]
    assert [n for n in with_walkthrough if f'"{n}"' not in runner] == []


def test_evals_well_formed():
    d = json.loads((ROOT / "evals" / "evals.json").read_text())
    ids = [e["id"] for e in d["evals"]]
    assert d["skill_name"] == "pytorch" and len(ids) == len(set(ids)) and len(ids) >= 15


@pytest.mark.parametrize("case", json.loads((ROOT / "evals" / "evals.json").read_text())["evals"],
                         ids=lambda c: c["id"])
def test_each_eval_has_a_prompt_and_three_expectations(case):
    assert case["prompt"].strip() and len(case["expectations"]) >= 3


@pytest.mark.parametrize("ex", sorted((ROOT / "examples").glob("0*.md")), ids=_ids)
def test_example_is_marked_illustrative_and_indexed(ex):
    assert "*Illustrative" in ex.read_text() and ex.name in (ROOT / "examples" / "README.md").read_text()


def test_version_metadata_agrees_with_verify_api():
    skill = (ROOT / "SKILL.md").read_text()
    script = (ROOT / "scripts" / "verify_api.py").read_text()
    assert 'pytorch-version: "2.15.0"' in skill and 'REFERENCE_VERSION = "torch 2.15.0' in script
