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
    block = re.search(r"^description:(.*?)^(?=[A-Za-z_-]+:|---)", SKILL, re.MULTILINE | re.DOTALL)
    assert block is not None
    text = re.sub(r"^\s*[>|][+-]?\s*$", "", block.group(1).strip(), count=1, flags=re.MULTILINE).strip()
    assert text and len(text) <= 1024 and "<" not in text and ">" not in text


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
    name = re.search(r"^name:\s*(\S+)", SKILL, re.MULTILINE)
    assert name is not None and data["skill_name"] == name.group(1)
    ids = [case["id"] for case in data["evals"]]
    assert len(ids) == len(set(ids))
    assert all(case["prompt"] and case["expectations"] for case in data["evals"])


def test_trigger_evals_have_both_kinds():
    cases = json.loads((ROOT / "evals" / "trigger_evals.json").read_text(encoding="utf-8"))
    assert {case["should_trigger"] for case in cases} == {True, False}


def test_validator_codes_match_the_formats_table():
    import validate_audit

    doc_codes = set(re.findall(r"^\s+(AS\d{3})\s", validate_audit.__doc__ or "", re.MULTILINE))
    table = (ROOT / "references" / "formats.md").read_text(encoding="utf-8")
    table_codes = set(re.findall(r"^\| (AS\d{3}) \|", table, re.MULTILINE))
    source = (ROOT / "scripts" / "validate_audit.py").read_text(encoding="utf-8")
    emitted = set(re.findall(r'Finding\("(AS\d{3})"', source))
    assert doc_codes == table_codes == emitted


def test_vocabularies_match_formats_md():
    import audit_common

    text = (ROOT / "references" / "formats.md").read_text(encoding="utf-8")
    for vocabulary in (audit_common.VERDICTS, audit_common.KINDS, audit_common.SEARCH_PURPOSES):
        assert all(f"`{word}`" in text for word in vocabulary), sorted(vocabulary)


def test_skill_md_names_every_script():
    assert all(f"scripts/{p.name}" in SKILL for p in (ROOT / "scripts").glob("*.py") if p.name != "audit_common.py")


def test_review_cap_in_docs_matches_code():
    import audit_common

    assert audit_common.MAX_REVIEW_ROUNDS == 3
    assert "3 rounds" in SKILL and "3 rounds" in (ROOT / "references" / "review-loop.md").read_text(encoding="utf-8")
