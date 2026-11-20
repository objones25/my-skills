"""Structural checks on the skill itself: paths resolve, frontmatter valid, style holds, evals well formed.

Each check is a function of a skill root, so the same code runs on the real skill and on
a deliberately broken temporary copy (`skill_copy`) that proves it can fail.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import SKILL_ROOT

ROOT = SKILL_ROOT
SKILL = (ROOT / "SKILL.md").read_text()

PATH_RE = re.compile(
    r"`((?:references|scripts|assets|examples|selftest|evals|papers)/[A-Za-z0-9_./-]+)"
    r"(?:#[^\s`]*)?"  # an anchor: `references/x.md#section`
    r"(?:\s[^`]*)?`"  # trailing arguments: `scripts/x.py --flag tests/`
)
REF_NAME_RE = re.compile(r"`([A-Za-z0-9_-]+\.md)`")
ROOT_DOCS = {"SKILL.md", "README.md", "INSTALL.md", "CHANGELOG.md", "AUDIT.md", "CLAUDE.md"}
STEP_HEADING_RE = re.compile(r"^\*\*(\d+)\. ([^*]+)\*\*", re.MULTILINE)
STEP_REF_RE = re.compile(r"[Ww]orkflow step (\d+)(?::\s+([A-Za-z ]+))?")
GATE_COUNT_RE = re.compile(r"(\d+) hard gates")
SCRIPT_SELFTEST_TESTS_MIN = 3
EXPECTATION_CHARS_MIN = 10
SKILL_MD_LINES_MAX = 260


# --------------------------------------------------------------------------
# Checks. Each takes a skill root and returns what is wrong (empty means fine).
# --------------------------------------------------------------------------
def md_files(root: Path) -> list[Path]:
    return [
        root / "SKILL.md", root / "INSTALL.md", root / "AUDIT.md",
        *sorted((root / "references").rglob("*.md")), *sorted((root / "examples").rglob("*.md")),
    ]


def _exists(root: Path, md: Path, rel: str) -> bool:
    rel = rel.rstrip(".,:")
    return any(c.exists() for c in (root / rel, root / "references" / rel, md.parent / rel))


def missing_backticked_paths(root: Path, md: Path) -> list[str]:
    return sorted({m for m in PATH_RE.findall(md.read_text()) if not _exists(root, md, m)})


def unknown_bare_names(root: Path, md: Path) -> list[str]:
    known = {p.name for p in (root / "references").rglob("*.md")} | ROOT_DOCS
    return sorted({n for n in REF_NAME_RE.findall(md.read_text()) if n not in known})


def _asserting_tests(source: str) -> int:
    """Count test functions that can fail: a non-constant assert, pytest.raises or pytest.fail."""
    count = 0
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        for inner in ast.walk(node):
            asserts = isinstance(inner, ast.Assert) and not isinstance(inner.test, ast.Constant)
            raises = isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr in {"raises", "fail"}
            if asserts or raises:
                count += 1
                break
    return count


def scripts_without_a_real_selftest(root: Path) -> list[str]:
    """A script needs selftest/test_<stem>.py that names the script and holds real assertions."""
    problems = []
    scripts = sorted((root / "scripts").glob("*.py"))
    if not scripts:
        return ["no scripts found: nothing was checked"]
    for script in scripts:
        test = root / "selftest" / f"test_{script.stem}.py"
        if not test.exists():
            problems.append(f"{script.name}: no selftest/{test.name}")
            continue
        source = test.read_text()
        if script.name not in source:
            problems.append(f"{script.name}: {test.name} never mentions {script.name}")
        elif (n := _asserting_tests(source)) < SCRIPT_SELFTEST_TESTS_MIN:
            problems.append(f"{script.name}: {test.name} has {n} asserting tests, needs {SCRIPT_SELFTEST_TESTS_MIN}")
    return problems


def eval_expectation_problems(case: dict) -> list[str]:
    problems = []
    if not str(case.get("prompt", "")).strip():
        problems.append("empty prompt")
    expectations = case.get("expectations")
    if not isinstance(expectations, list) or not 3 <= len(expectations) <= 5:
        return [*problems, "expectations must be a list of 3 to 5 items"]
    for item in expectations:
        if not isinstance(item, str) or len(item.strip()) < EXPECTATION_CHARS_MIN:
            problems.append(f"expectation is not a sentence: {item!r}")
    texts = [e for e in expectations if isinstance(e, str)]
    if len(set(texts)) != len(texts):
        problems.append("duplicate expectations")
    return problems


def unmarked_diagnoses(root: Path) -> list[str]:
    files = [p for p in sorted((root / "examples" / "diagnoses").glob("*.md")) if p.name != "README.md"]
    if not files:
        return ["no diagnosis files found: nothing was checked"]
    return [p.name for p in files if "*Illustrative" not in p.read_text()]


def wrong_step_references(root: Path) -> list[str]:
    """A reference to 'workflow step N' must use a step that exists; with a title it must name that step."""
    steps = {int(n): title.lower() for n, title in STEP_HEADING_RE.findall((root / "SKILL.md").read_text())}
    wrong = []
    for md in md_files(root):
        for number, phrase in STEP_REF_RE.findall(md.read_text()):
            title = steps.get(int(number))
            if title is None:
                wrong.append(f"{md.name}: there is no workflow step {number}")
                continue
            if not phrase:
                continue
            title_words = {w for w in re.findall(r"[a-z]+", title) if len(w) > 4}
            if not title_words <= set(re.findall(r"[a-z]+", phrase.lower())):
                wrong.append(f"{md.name}: step {number} is {title!r}, not {phrase.strip()!r}")
    return wrong


def gate_count_mismatches(root: Path) -> list[str]:
    """Every file that states 'N hard gates' must agree with the rule headings; INSTALL.md must state it."""
    actual = len(re.findall(r"^### Rule ", (root / "references" / "rules.md").read_text(), re.MULTILINE))
    stated = {
        md.relative_to(root).as_posix(): [int(n) for n in GATE_COUNT_RE.findall(md.read_text())]
        for md in md_files(root) if md.name != "AUDIT.md"  # AUDIT.md quotes past counts
    }
    if not stated["INSTALL.md"]:
        return ["INSTALL.md does not state the gate count"]
    return [f"{name} says {n} hard gates, rules.md has {actual}" for name, counts in stated.items() for n in counts if n != actual]


CLEAN_AUDIT_RE = re.compile(r"^clean: no findings \((\d+) file\(s\) checked\)$", re.MULTILINE)


def own_tests_not_clean_under_own_audit(root: Path) -> list[str]:
    """The skill's own `selftest/` and `assets/` must pass the skill's own `scripts/audit_tests.py`."""
    problems = []
    for folder in ("selftest", "assets"):
        run = subprocess.run(
            [sys.executable, str(root / "scripts" / "audit_tests.py"), str(root / folder)],
            cwd=root, capture_output=True, text=True, timeout=120, check=False,
        )
        checked = CLEAN_AUDIT_RE.search(run.stdout)
        if run.returncode != 0 or checked is None or int(checked.group(1)) == 0:
            problems.append(f"{folder}/: audit_tests.py exit {run.returncode}: {(run.stdout + run.stderr).strip()}")
    return problems


def mutmut_disagreements(root: Path) -> list[str]:
    """The final mutmut run is quoted in the reference and in the wallet README; they must agree."""
    reference = (root / "references" / "mutation-testing.md").read_text()
    readme = (root / "examples" / "wallet" / "README.md").read_text()
    runs = re.findall(r"(\d+)/(\d+)\s+killed (\d+)\s+survived (\d+)", reference)
    if not runs:
        return ["references/mutation-testing.md quotes no mutmut result"]
    done, total, killed, survived = runs[-1]
    problems = []
    if done != total or int(killed) + int(survived) != int(total):
        problems.append(f"reference result does not add up: {done}/{total} killed {killed} survived {survived}")
    for needle, where, text in (
        (f"{killed} of {total} killed", "wallet README", readme),
        (f"mutmut reports {survived} survivors", "wallet README", readme),
        (f"What the remaining {survived} teach", "reference", reference),
    ):
        if needle not in text:
            problems.append(f"{where} lacks {needle!r}")
    return problems


# --------------------------------------------------------------------------
# The real skill
# --------------------------------------------------------------------------
MD_FILES = md_files(ROOT)


@pytest.mark.parametrize("md", MD_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_backticked_paths_exist(md):
    """CHANGELOG.md is left out: history names files that were later renamed or removed."""
    assert missing_backticked_paths(ROOT, md) == []


def test_changelog_has_no_em_dashes():
    assert "—" not in (ROOT / "CHANGELOG.md").read_text()


@pytest.mark.parametrize("md", [ROOT / "SKILL.md", *sorted((ROOT / "references").glob("*.md"))], ids=lambda p: p.name)
def test_bare_reference_names_exist(md):
    assert unknown_bare_names(ROOT, md) == []


def test_frontmatter():
    m = re.match(r"^---\nname: ([a-z0-9-]+)\ndescription: (.+?)\n(?:metadata:\n(?:  .+\n)+)?---\n", SKILL, re.DOTALL)
    assert m, "SKILL.md must open with name and description frontmatter"
    assert m.group(1) == "pytest-expert"
    assert len(m.group(2)) <= 1024
    assert "<" not in m.group(2)
    assert ">" not in m.group(2)


def test_test_command_runs_the_selftests():
    """skill-builder's packager runs metadata.test-command in the unpacked copy; without it the zip is never tested."""
    m = re.search(r"^metadata:\n(?:  .+\n)*?  test-command: \"(.+)\"\n", SKILL, re.MULTILINE)
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
    assert len(SKILL.splitlines()) <= SKILL_MD_LINES_MAX


@pytest.mark.parametrize("md", MD_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_em_dashes(md):
    assert "—" not in md.read_text()


def test_the_gate_count_is_stated_the_same_everywhere():
    assert gate_count_mismatches(ROOT) == []


def test_every_script_has_a_selftest():
    assert scripts_without_a_real_selftest(ROOT) == []


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
    assert eval_expectation_problems(case) == []


def test_examples_are_marked_illustrative():
    assert unmarked_diagnoses(ROOT) == []


def test_workflow_step_references_match_skill_md():
    assert wrong_step_references(ROOT) == []


def test_mutmut_result_is_stated_the_same_everywhere():
    assert mutmut_disagreements(ROOT) == []


def test_the_skills_own_tests_pass_its_own_audit():
    assert own_tests_not_clean_under_own_audit(ROOT) == []


def test_wallet_readme_passed_count_matches_the_collected_tests():
    """The wallet README quotes the default `pytest` count; a test added or removed makes it stale."""
    pytest.importorskip("hypothesis", reason="the wallet suite needs hypothesis")
    pytest.importorskip("pytest_cov", reason="the wallet's pyproject addopts need pytest-cov")
    wallet = ROOT / "examples" / "wallet"
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-cov", "-p", "no:cacheprovider", "-p", "no:randomly"],
        cwd=wallet, capture_output=True, text=True, timeout=120, check=False,
    )
    selected = re.search(r"(\d+)(?:/\d+)? tests? collected", run.stdout)
    assert selected, run.stdout + run.stderr
    row = next(line for line in (wallet / "README.md").read_text().splitlines() if line.startswith("| `pytest` |"))
    stated = re.search(r"(\d+) passed", row)
    assert stated, row
    assert int(stated.group(1)) == int(selected.group(1))


# --------------------------------------------------------------------------
# Each check can fail: a broken temporary copy of the skill, never the real files
# --------------------------------------------------------------------------
@pytest.fixture
def skill_copy(tmp_path) -> Path:
    dest = tmp_path / "skill"
    shutil.copytree(
        ROOT, dest,
        ignore=shutil.ignore_patterns("__pycache__", ".hypothesis", ".git", "*.bak", "mutants", ".coverage*", ".pytest_cache"),
    )
    return dest


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text() + text)


def test_the_copy_itself_passes_every_check(skill_copy):
    assert missing_backticked_paths(skill_copy, skill_copy / "SKILL.md") == []
    assert unknown_bare_names(skill_copy, skill_copy / "SKILL.md") == []
    assert scripts_without_a_real_selftest(skill_copy) == []
    assert wrong_step_references(skill_copy) == []
    assert gate_count_mismatches(skill_copy) == []
    assert unmarked_diagnoses(skill_copy) == []
    assert mutmut_disagreements(skill_copy) == []
    assert own_tests_not_clean_under_own_audit(skill_copy) == []


@pytest.mark.parametrize(
    "snippet",
    [
        "see `references/nope.md`",
        "see `references/nope.md#section`",
        "run `scripts/nope.py --flag value`",
        "run `scripts/nope.py tests/` first",
        "see `references/nope.md#section and more`",
    ],
    ids=["plain", "anchor", "arguments", "arguments-with-directory", "anchor-then-words"],
)
def test_a_backticked_path_to_a_missing_file_is_found(skill_copy, snippet):
    md = skill_copy / "SKILL.md"
    _append(md, f"\n{snippet}\n")
    assert missing_backticked_paths(skill_copy, md) == [snippet.split("`")[1].split("#")[0].split()[0]]


@pytest.mark.parametrize(
    "snippet",
    ["`references/rules.md#rule-1`", "`scripts/audit_tests.py --format json tests/`", "`examples/wallet`", "`references/rules.md`"],
    ids=["anchor", "arguments", "directory", "plain"],
)
def test_a_backticked_path_to_an_existing_file_stays_clean(skill_copy, snippet):
    md = skill_copy / "SKILL.md"
    _append(md, f"\n{snippet}\n")
    assert missing_backticked_paths(skill_copy, md) == []


@pytest.mark.parametrize("name", ["nonexistent-ref.md", "9-nonexistent.md", "Nonexistent_Ref.md"], ids=["lowercase", "digit-first", "capitals-and-underscore"])
def test_an_unknown_bare_markdown_name_is_found(skill_copy, name):
    md = skill_copy / "SKILL.md"
    _append(md, f"\nsee `{name}`\n")
    assert unknown_bare_names(skill_copy, md) == [name]


def test_known_bare_markdown_names_stay_clean(skill_copy):
    md = skill_copy / "SKILL.md"
    _append(md, "\nsee `rules.md`, `README.md` and `CHANGELOG.md`\n")
    assert unknown_bare_names(skill_copy, md) == []


def test_a_selftest_file_replaced_by_a_stub_is_found(skill_copy):
    (skill_copy / "selftest" / "test_diff_guard.py").write_text("def test_x():\n    assert True\n")
    problems = scripts_without_a_real_selftest(skill_copy)
    assert problems == ["diff_guard.py: test_diff_guard.py never mentions diff_guard.py"]


def test_a_selftest_that_names_the_script_but_asserts_nothing_real_is_found(skill_copy):
    stub = 'SCRIPT = "diff_guard.py"\n\n\ndef test_a():\n    assert True\n\n\ndef test_b():\n    pass\n'
    (skill_copy / "selftest" / "test_diff_guard.py").write_text(stub)
    assert scripts_without_a_real_selftest(skill_copy) == ["diff_guard.py: test_diff_guard.py has 0 asserting tests, needs 3"]


def test_a_script_without_a_selftest_file_is_found(skill_copy):
    (skill_copy / "scripts" / "brand_new.py").write_text("print('hi')\n")
    assert scripts_without_a_real_selftest(skill_copy) == ["brand_new.py: no selftest/test_brand_new.py"]


def test_an_empty_scripts_directory_is_not_reported_as_fine(skill_copy):
    for script in (skill_copy / "scripts").glob("*.py"):
        script.unlink()
    assert scripts_without_a_real_selftest(skill_copy) == ["no scripts found: nothing was checked"]


def test_a_selftest_with_three_asserting_tests_is_accepted(skill_copy):
    body = 'import pytest\nSCRIPT = "diff_guard.py"\n\n\ndef test_a():\n    assert 1 + 1 == 2\n\n\ndef test_b():\n    assert "x" in "xy"\n\n\ndef test_c():\n    with pytest.raises(ZeroDivisionError):\n        1 / 0\n'
    (skill_copy / "selftest" / "test_diff_guard.py").write_text(body)
    assert scripts_without_a_real_selftest(skill_copy) == []


def test_a_tenth_diagnosis_without_the_marker_is_found(skill_copy):
    (skill_copy / "examples" / "diagnoses" / "10-late-case.md").write_text("# A diagnosis\n\nNo marker here.\n")
    assert unmarked_diagnoses(skill_copy) == ["10-late-case.md"]


def test_an_empty_diagnoses_directory_is_not_reported_as_fine(skill_copy):
    for md in (skill_copy / "examples" / "diagnoses").glob("*.md"):
        md.unlink()
    assert unmarked_diagnoses(skill_copy) == ["no diagnosis files found: nothing was checked"]


def _step_5_title() -> str:
    return next(t.strip() for n, t in STEP_HEADING_RE.findall(SKILL) if n == "5").lower()


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        ("Workflow step 5: install the gates", "step 5 is"),
        ("workflow step 5: prove nothing at all", "step 5 is"),
        ("Workflow step 9: whatever", "there is no workflow step 9"),
        ("Workflow step 9 covers it", "there is no workflow step 9"),
    ],
    ids=["capital-and-wrong-title", "first-word-right-rest-wrong", "no-such-step", "no-colon-no-such-step"],
)
def test_a_wrong_workflow_step_reference_is_found(skill_copy, snippet, expected):
    md = skill_copy / "references" / "rules.md"
    _append(md, f"\n{snippet}\n")
    wrong = wrong_step_references(skill_copy)
    assert len(wrong) == 1 and expected in wrong[0], wrong


@pytest.mark.parametrize(
    "snippet",
    ["Workflow step 5: prove the tests you just wrote can fail", "workflow step 5 (the mutation step)", "Workflow step 5: prove the tests can fail."],
    ids=["longer-phrase-with-the-title-words", "no-colon-existing-step", "capital-and-exact-title"],
)
def test_a_correct_workflow_step_reference_stays_clean(skill_copy, snippet):
    assert _step_5_title().startswith("prove the tests can fail")
    _append(skill_copy / "references" / "rules.md", f"\n{snippet}\n")
    assert wrong_step_references(skill_copy) == []


def test_a_gate_count_that_differs_in_skill_md_is_found(skill_copy):
    _append(skill_copy / "SKILL.md", "\nThe skill has 7 hard gates.\n")
    actual = len(re.findall(r"^### Rule ", (skill_copy / "references" / "rules.md").read_text(), re.MULTILINE))
    assert gate_count_mismatches(skill_copy) == [f"SKILL.md says 7 hard gates, rules.md has {actual}"]


def test_a_gate_count_missing_from_install_md_is_found(skill_copy):
    install = skill_copy / "INSTALL.md"
    install.write_text(GATE_COUNT_RE.sub("many hard gates", install.read_text()))
    assert gate_count_mismatches(skill_copy) == ["INSTALL.md does not state the gate count"]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ({"prompt": "p", "expectations": ["", "", ""]}, 4),
        ({"prompt": "p", "expectations": [1, 2, 3]}, 3),
        ({"prompt": "p", "expectations": ["ok", "fine and long enough", "also long enough here"]}, 1),
        ({"prompt": "p", "expectations": ["same sentence here", "same sentence here", "another sentence here"]}, 1),
        ({"prompt": "   ", "expectations": ["first long sentence", "second long sentence", "third long sentence"]}, 1),
        ({"prompt": "p", "expectations": ["only one long sentence"]}, 1),
    ],
    ids=["empty-strings", "non-strings", "too-short", "duplicates", "blank-prompt", "too-few"],
)
def test_a_malformed_eval_case_is_found(case, expected):
    assert len(eval_expectation_problems(case)) == expected, eval_expectation_problems(case)


def test_a_well_formed_eval_case_is_clean():
    case = {"prompt": "Why is my suite green?", "expectations": ["names the cause", "runs the audit script", "shows the command"]}
    assert eval_expectation_problems(case) == []


def test_a_mutmut_figure_that_disagrees_between_reference_and_readme_is_found(skill_copy):
    readme = skill_copy / "examples" / "wallet" / "README.md"
    readme.write_text(re.sub(r"\d+ of (\d+) killed", r"1 of \1 killed", readme.read_text()))
    problems = mutmut_disagreements(skill_copy)
    assert len(problems) == 1 and problems[0].startswith("wallet README lacks '") and "killed" in problems[0]


def test_a_mutmut_survivor_count_that_disagrees_is_found(skill_copy):
    readme = skill_copy / "examples" / "wallet" / "README.md"
    readme.write_text(re.sub(r"mutmut reports \d+ survivors", "mutmut reports 99 survivors", readme.read_text()))
    problems = mutmut_disagreements(skill_copy)
    assert len(problems) == 1 and "mutmut reports" in problems[0]


def test_a_reference_without_a_mutmut_result_is_found(skill_copy):
    ref = skill_copy / "references" / "mutation-testing.md"
    ref.write_text(re.sub(r"\d+/\d+\s+killed \d+\s+survived \d+", "no result", ref.read_text()))
    assert mutmut_disagreements(skill_copy) == ["references/mutation-testing.md quotes no mutmut result"]


@pytest.mark.parametrize("folder", ["selftest", "assets"])
def test_an_assertion_free_test_in_the_skills_own_tests_is_found(skill_copy, folder):
    (skill_copy / folder / "test_planted.py").write_text("def test_planted():\n    compute()\n")
    problems = own_tests_not_clean_under_own_audit(skill_copy)
    assert len(problems) == 1
    assert problems[0].startswith(f"{folder}/: audit_tests.py exit 1:")
    assert "test_planted.py:1: NO_ASSERT" in problems[0]
