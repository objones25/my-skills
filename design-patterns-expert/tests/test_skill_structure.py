"""Structural checks: the self-claims SKILL.md makes about its own files stay true."""
from __future__ import annotations

import re
from pathlib import Path

import design_smells
import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / "SKILL.md").read_text()
MD_FILES = [ROOT / "SKILL.md", ROOT / "CHANGELOG.md", ROOT / "AUDIT.md",
            *sorted((ROOT / "references").glob("*.md"))]
PATH_RE = re.compile(r"`((?:references|scripts|examples|evals|tests)/[A-Za-z0-9_./-]+)`")


def _frontmatter() -> str:
    match = re.match(r"---\n(.*?)\n---\n", SKILL, re.DOTALL)
    assert match, "SKILL.md has no frontmatter"
    return match.group(1)


def test_frontmatter_has_name_version_test_command_and_negative_scope() -> None:
    fm = _frontmatter()
    assert re.search(r'^name: "?design-patterns-expert"?$', fm, re.MULTILINE)
    assert re.search(r"^  version: \d+\.\d+\.\d+$", fm, re.MULTILINE)
    assert re.search(r"^  test-command: .*pytest", fm, re.MULTILINE)
    description = re.search(r"^description: (.*)$", fm, re.MULTILINE)
    assert description and "Not for" in description.group(1)


def test_changelog_has_an_entry_for_the_current_version() -> None:
    found = re.search(r"^  version: (\S+)$", _frontmatter(), re.MULTILINE)
    assert found, "no metadata.version"
    version = found.group(1)
    assert re.search(rf"^## {re.escape(version)} \(\d{{4}}-\d{{2}}-\d{{2}}\)", (ROOT / "CHANGELOG.md").read_text(),
                     re.MULTILINE)


@pytest.mark.parametrize("md", MD_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_backticked_paths_exist(md: Path) -> None:
    missing = [rel for rel in (m.group(1).split("#")[0].rstrip(".,") for m in PATH_RE.finditer(md.read_text()))
               if not (ROOT / rel).exists()]
    assert not missing, missing


def test_reference_table_lists_every_reference_file_and_no_others() -> None:
    listed = set(re.findall(r"^\| `(references/[a-z-]+\.md)` \|", SKILL, re.MULTILINE))
    on_disk = {f"references/{p.name}" for p in (ROOT / "references").glob("*.md")}
    assert listed == on_disk


def test_smell_kind_count_in_skill_md_matches_the_scanner() -> None:
    stated = re.search(r"It reports (\d+) kinds of signal", SKILL)
    assert stated and int(stated.group(1)) == len(design_smells.KINDS)


def test_claim_check_counts_match_verify_python_claims() -> None:
    n_checks = len(re.findall(r'^@check\("', (ROOT / "scripts" / "verify_python_claims.py").read_text(),
                              re.MULTILINE))
    idioms = (ROOT / "references" / "python-idioms.md").read_text()
    assert re.search(rf"\({n_checks} checks\)", SKILL)
    assert re.search(rf"run it to see all {n_checks}\b", idioms)


def test_every_script_is_exercised_by_a_test() -> None:
    tests_text = "\n".join(p.read_text() for p in (ROOT / "tests").glob("test_*.py"))
    scripts = [*(ROOT / "scripts").glob("*.py"), ROOT / "evals" / "grade.py"]
    untested = [p.name for p in scripts if p.stem not in tests_text]
    assert not untested, untested
