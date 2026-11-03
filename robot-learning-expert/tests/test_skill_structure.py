"""Structural checks: every file path the skill mentions exists, frontmatter is valid, style rules hold."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MD_FILES = [ROOT / "SKILL.md", *sorted((ROOT / "references").rglob("*.md"))]
# Paths that live outside the skill (installed packages, other repos).
EXTERNAL = re.compile(
    r"^(CS291K|mprl|mjlab|rsl_rl|modules/|algorithms/|massively-parallel-rl|results-main|student\.py|"
    r"assignment_1|a2_trajopt|notes\d|docs/|src/|g1_constants|velocity_env_cfg|config/|tasks/)"
)
PATH_RE = re.compile(r"`((?:references|scripts|examples|assets|tests|evals|concepts|training|papers)/[A-Za-z0-9_./-]+)`")


def _resolve(md: Path, rel: str) -> bool:
    candidates = [ROOT / rel, ROOT / "references" / rel, md.parent / rel]
    return any(c.exists() for c in candidates)


@pytest.mark.parametrize("md", MD_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_backticked_paths_exist(md):
    text = md.read_text()
    missing = []
    for m in PATH_RE.finditer(text):
        rel = m.group(1).split("#")[0].rstrip(".,")
        if EXTERNAL.match(rel):
            continue
        if not _resolve(md, rel):
            missing.append(rel)
    assert not missing, f"{md.name} mentions missing files: {sorted(set(missing))}"


def test_frontmatter():
    text = (ROOT / "SKILL.md").read_text()
    m = re.match(r"^---\nname: ([a-z0-9-]+)\ndescription: (.+?)\n---\n", text, re.S)
    assert m, "SKILL.md needs name and description frontmatter"
    assert m.group(1) == "robot-learning-expert" == ROOT.name
    assert len(m.group(2)) <= 1024
    assert "<" not in m.group(2) and ">" not in m.group(2)


def test_skill_md_is_short():
    assert len((ROOT / "SKILL.md").read_text().splitlines()) <= 200


@pytest.mark.parametrize("md", MD_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_em_dashes(md):
    assert "\u2014" not in md.read_text(), f"em dash in {md.name}"


def test_evals_json_wellformed():
    data = json.loads((ROOT / "evals" / "evals.json").read_text())
    assert data["skill_name"] == "robot-learning-expert"
    ids = [e["id"] for e in data["evals"]]
    assert len(ids) == len(set(ids))
    for e in data["evals"]:
        assert e["prompt"] and e["expectations"]


def test_every_script_has_a_test():
    scripts = {p.stem for p in (ROOT / "scripts").glob("*.py")}
    tests = {p.stem.removeprefix("test_") for p in (ROOT / "tests").glob("test_*.py")}
    assert scripts <= tests, f"untested scripts: {sorted(scripts - tests)}"


GENERAL_FILES = [ROOT / "SKILL.md", *sorted((ROOT / "references").rglob("*.md")), *sorted((ROOT / "assets").glob("*")),
                 *sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "examples").glob("*.py")), ROOT / "evals" / "evals.json"]
COURSE_FRAMING = re.compile(r"CS ?291K|syllabus|student\.py|Gradescope|tutor mode|plagiari|\bA[34]\b \(|Owen's|group project|pre-lecture")


@pytest.mark.parametrize("path", GENERAL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_skill_is_not_framed_around_one_course(path):
    if path.name == "CHANGELOG.md":
        pytest.skip("history may mention the old course framing")
    hits = COURSE_FRAMING.findall(path.read_text())
    assert not hits, f"{path.name} still has course-specific framing: {sorted(set(hits))}"
