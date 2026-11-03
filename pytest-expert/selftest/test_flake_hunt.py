"""flake_hunt.py: order dependence and nondeterminism are told apart; a clean suite is clean."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest
from conftest import SKILL_ROOT

SCRIPT = SKILL_ROOT / "scripts" / "flake_hunt.py"
pytestmark = pytest.mark.skipif(importlib.util.find_spec("pytest_randomly") is None, reason="needs pytest-randomly")

LEAKY_STATE = {
    "state.py": "RATES = {}\n",
    "tests/test_a_sets.py": "import state\n\n\ndef test_sets_rate():\n    state.RATES['EUR'] = 0.5\n    assert state.RATES['EUR'] == 0.5\n",
    "tests/test_b_reads.py": "import state\n\n\ndef test_rate_table_starts_empty():\n    assert state.RATES == {}\n",
}
ALTERNATING = {
    "tests/test_alt.py": (
        "from pathlib import Path\n\n\n"
        "def test_fails_every_other_run(tmp_path_factory):\n"
        "    counter = Path(__file__).with_name('runs.txt')\n"
        "    n = int(counter.read_text()) if counter.exists() else 0\n"
        "    counter.write_text(str(n + 1))\n"
        "    assert n % 2 == 0\n"
    ),
}


def hunt(tmp_path, files, *args):
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    cmd = [sys.executable, str(SCRIPT), "--runs", "6", "--format", "json", *args, "--", "-p", "no:asyncio", "tests/"]
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=600,
                          env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"}, check=False)
    return proc.returncode, json.loads(proc.stdout)


def test_order_dependent_test_is_labelled_order_dependent(tmp_path):
    code, report = hunt(tmp_path, LEAKY_STATE, "--repeat-seed", "2")
    assert code == 1
    assert [(f["test"], f["kind"]) for f in report["flaky"]] == [("tests.test_b_reads::test_rate_table_starts_empty", "order-dependent")]


def test_nondeterministic_test_is_labelled_nondeterministic(tmp_path):
    code, report = hunt(tmp_path, ALTERNATING, "--repeat-seed", "2")
    assert code == 1
    assert [f["kind"] for f in report["flaky"]] == ["nondeterministic"]


def test_clean_suite_exits_0(tmp_path):
    files = {"tests/test_ok.py": "def test_ok():\n    assert 1 + 1 == 2\n"}
    assert hunt(tmp_path, files) == (0, {"tests": 1, "flaky": []})
