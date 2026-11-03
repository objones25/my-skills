"""Shared helpers: build throwaway skills and run the bundled scripts on them."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

GOOD_DESCRIPTION = ("Check widgets for damage. Use when the user asks to inspect a widget or says "
                    "'is this widget broken'. Not for gadgets; use gadget-expert.")


def write_skill(root: Path, name: str = "demo-skill", description: str = GOOD_DESCRIPTION,
                body: str = "# Demo\n\nDo the thing.\n") -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    quoted = json.dumps(description)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {quoted}\n---\n\n{body}", encoding="utf-8")
    return skill


def add_core_files(skill: Path) -> None:
    (skill / "CHANGELOG.md").write_text("# Changelog\n\n## 1.0.0, 2026-11-03\n\nFirst release.\n", encoding="utf-8")
    evals = skill / "evals"
    evals.mkdir(exist_ok=True)
    (evals / "evals.json").write_text(json.dumps({
        "skill_name": skill.name,
        "evals": [{"id": 1, "prompt": "Inspect this widget", "expected_output": "A verdict",
                   "files": [], "expectations": ["States whether the widget is damaged"]}],
    }), encoding="utf-8")
    (evals / "trigger_evals.json").write_text(json.dumps([
        {"query": "is this widget broken", "should_trigger": True},
        {"query": "fix my gadget", "should_trigger": False},
    ]), encoding="utf-8")


@pytest.fixture
def skill(tmp_path) -> Path:
    """A minimal skill that meets every core requirement."""
    made = write_skill(tmp_path)
    add_core_files(made)
    return made


@pytest.fixture
def run_script():
    def run(name: str, *args, cwd=None):
        return subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)], cwd=cwd,
                              capture_output=True, text=True, timeout=120)
    return run


@pytest.fixture
def codes(run_script):
    """Run check_skill.py on a skill and return the set of rule codes it reports."""
    def run(path: Path, *extra) -> set[str]:
        result = run_script("check_skill.py", path, "--json", *extra)
        assert result.returncode in (0, 1), result.stderr
        return {finding["code"] for finding in json.loads(result.stdout)["findings"]}
    return run
