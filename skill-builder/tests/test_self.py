"""This skill must meet the standard it sets."""
from __future__ import annotations

from conftest import SCRIPTS, SKILL_ROOT

import sys
import subprocess


def _check(*extra):
    # SB012: running these tests can itself leave __pycache__ in the working folder.
    return subprocess.run([sys.executable, str(SCRIPTS / "check_skill.py"), str(SKILL_ROOT), "--ignore", "SB012", *extra],
                          capture_output=True, text=True, timeout=60)


def test_passes_its_own_check_with_no_warnings():
    result = _check("--strict")
    assert result.returncode == 0, result.stdout


def test_trigger_evals_have_at_least_eight_of_each_kind():
    import json
    cases = json.loads((SKILL_ROOT / "evals" / "trigger_evals.json").read_text(encoding="utf-8"))
    assert sum(c["should_trigger"] for c in cases) >= 8
    assert sum(not c["should_trigger"] for c in cases) >= 8


def test_standard_documents_every_checker_code():
    sys.path.insert(0, str(SCRIPTS))
    from check_skill import RULES
    text = (SKILL_ROOT / "references" / "standard.md").read_text(encoding="utf-8")
    assert not [code for code in RULES if code not in text]
