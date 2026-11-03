"""Keep this skill's files consistent with each other. Runs with pytest and the standard library."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / "SKILL.md").read_text(encoding="utf-8")
MARKDOWN = [p for p in ROOT.rglob("*.md") if "evals" not in p.relative_to(ROOT).parts]
PATH_RE = re.compile(r"`((?:references|scripts|assets|examples|evals)/[A-Za-z0-9_./-]+)`")
MAX_SKILL_LINES = 500


def test_paths_in_skill_md_exist():
    missing = [m for m in PATH_RE.findall(SKILL) if not (ROOT / m.rstrip(".,")).exists()]
    assert not missing


def test_description_is_within_the_limit():
    match = re.search(r"^description: (.*)$", SKILL, re.M)
    assert match is not None
    assert len(match.group(1)) <= 1024 and "<" not in match.group(1) and ">" not in match.group(1)


def test_skill_md_is_short():
    assert SKILL.count("\n") <= MAX_SKILL_LINES


def test_every_reference_is_routed_from_skill_md():
    references = ROOT / "references"
    orphans = [p.name for p in references.glob("*.md") if p.name not in SKILL and p.name != "sources.md"] if references.is_dir() else []
    assert not orphans


def test_every_script_has_a_test():
    scripts = ROOT / "scripts"
    tested = {p.stem.removeprefix("test_") for p in (ROOT / "tests").glob("test_*.py")}
    untested = [p.stem for p in scripts.glob("*.py") if p.stem not in tested] if scripts.is_dir() else []
    assert not untested


def test_no_em_dashes():
    assert not [p.name for p in MARKDOWN if "—" in p.read_text(encoding="utf-8")]


def test_evals_are_well_formed():
    data = json.loads((ROOT / "evals" / "evals.json").read_text(encoding="utf-8"))
    assert data["skill_name"] == ROOT.name
    ids = [case["id"] for case in data["evals"]]
    assert len(ids) == len(set(ids))
    assert all(case["prompt"] and case["expectations"] for case in data["evals"])


def test_trigger_evals_have_both_kinds():
    cases = json.loads((ROOT / "evals" / "trigger_evals.json").read_text(encoding="utf-8"))
    assert {case["should_trigger"] for case in cases} == {True, False}
