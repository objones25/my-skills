"""Keep this skill's files consistent with each other. Runs with pytest and the standard library."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / "SKILL.md").read_text(encoding="utf-8")
MARKDOWN = [p for p in ROOT.rglob("*.md") if "evals" not in p.relative_to(ROOT).parts]
PATH_RE = re.compile(r"`((?:references|scripts|assets|examples|evals|tests)/[A-Za-z0-9_./-]+)`")
MAX_SKILL_LINES = 150


def _frontmatter() -> str:
    m = re.match(r"^---\n(.*?)\n---\n", SKILL, re.DOTALL)
    assert m, "SKILL.md needs a frontmatter block"
    return m.group(1)


def test_paths_in_markdown_exist():
    missing = []
    for md in MARKDOWN:
        for rel in PATH_RE.findall(md.read_text(encoding="utf-8")):
            rel = rel.rstrip(".,")
            if not ((ROOT / rel).exists() or (ROOT / "references" / rel).exists() or (md.parent / rel).exists()):
                missing.append(f"{md.relative_to(ROOT)}: {rel}")
    assert not missing


def test_description_is_within_the_limit_and_names_negative_scope():
    m = re.search(r"^description: (.+)$", _frontmatter(), re.MULTILINE)
    assert m is not None
    text = m.group(1)
    assert len(text) <= 1024 and "<" not in text and ">" not in text
    assert "Not for" in text


def test_test_command_pins_the_versions_the_changelog_reports():
    m = re.search(r'^  test-command: "(.+)"$', _frontmatter(), re.MULTILINE)
    assert m is not None, "metadata.test-command is missing"
    command = m.group(1)
    for pin in ("jax==", "gymnax==", "flax==", "optax==", "distrax=="):
        assert pin in command, f"test-command does not pin {pin.rstrip('=')}"
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    jax_pin = re.search(r"jax==([0-9.]+)", command)
    gymnax_pin = re.search(r"gymnax==([0-9.]+)", command)
    assert jax_pin is not None and gymnax_pin is not None
    assert f"jax {jax_pin.group(1)}" in changelog and f"gymnax {gymnax_pin.group(1)}" in changelog


def test_skill_md_is_short():
    assert SKILL.count("\n") <= MAX_SKILL_LINES


def test_every_reference_is_routed_from_skill_md():
    orphans = [p.name for p in (ROOT / "references").glob("*.md") if p.name not in SKILL and p.name != "sources.md"]
    assert not orphans


def test_every_script_is_exercised_by_a_test():
    tests_text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py"))
    untested = [p.stem for p in (ROOT / "scripts").glob("*.py") if p.stem not in tests_text]
    assert not untested


def test_no_em_dashes():
    assert not [p.name for p in MARKDOWN if "—" in p.read_text(encoding="utf-8")]


def test_has_extend_section():
    assert "## Extend this skill" in SKILL


def test_evals_are_well_formed():
    data = json.loads((ROOT / "evals" / "evals.json").read_text(encoding="utf-8"))
    assert data["skill_name"] == ROOT.name
    ids = [case["id"] for case in data["evals"]]
    assert len(ids) == len(set(ids))
    assert all(case["prompt"].strip() and case["expectations"] for case in data["evals"])
    assert any(case.get("type") == "control" for case in data["evals"]), "needs a control case"
    for case in data["evals"]:
        for f in case.get("files", []):
            assert (ROOT / f).exists(), f


def test_case_folders_are_listed_in_evals_json():
    data = json.loads((ROOT / "evals" / "evals.json").read_text(encoding="utf-8"))
    listed = {case["id"] for case in data["evals"]}
    folders = {p.parent.name for p in (ROOT / "evals").glob("*/case.yaml")}
    assert folders <= listed, f"case folders missing from evals.json: {sorted(folders - listed)}"


def test_trigger_evals_have_both_kinds():
    cases = json.loads((ROOT / "evals" / "trigger_evals.json").read_text(encoding="utf-8"))
    assert all(set(c) == {"query", "should_trigger"} and c["query"].strip() for c in cases)
    assert sum(c["should_trigger"] for c in cases) >= 8
    assert sum(not c["should_trigger"] for c in cases) >= 8
    assert len({c["query"] for c in cases}) == len(cases)


def test_sources_and_audit_exist():
    assert (ROOT / "references" / "sources.md").exists()
    assert (ROOT / "AUDIT.md").exists()
