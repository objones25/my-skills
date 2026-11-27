"""Tests for the per-test duration budget in assets/conftest.py.

Each test copies the asset conftest into a hermetic project, runs pytest on it in a
subprocess, and reads the per-test outcome from the JUnit XML report. Inner tests that
must be slow busy-wait for 0.3 s against a 0.1 s budget; inner tests that must pass run
against a 30 s budget, so neither side is near the threshold.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
ASSET_CONFTEST = SKILL_ROOT / "assets" / "conftest.py"

BUSY = "import time\n\ndef busy(seconds):\n    end = time.perf_counter() + seconds\n    while time.perf_counter() < end:\n        pass\n"
TIGHT_BUDGET_S = "0.1"
LOOSE_BUDGET_S = "30"


def run_pytest(tmp_path: Path, test_source: str, budget_s: str) -> subprocess.CompletedProcess[str]:
    """Run pytest on one test file beside the asset conftest, writing JUnit XML to report.xml."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    shutil.copy(ASSET_CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_inner.py").write_text(BUSY + test_source, encoding="utf-8")
    env = {**os.environ, "PYTEST_EXPERT_BUDGET_S": budget_s}
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "-p", "no:cacheprovider", f"--junitxml={tmp_path / 'report.xml'}", "test_inner.py"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False,
    )


def run_project(tmp_path: Path, test_source: str, budget_s: str) -> dict[str, ET.Element]:
    """Run pytest via run_pytest; map each test name to its <testcase>."""
    run_pytest(tmp_path, test_source, budget_s)
    root = ET.parse(tmp_path / "report.xml").getroot()  # noqa: S314, RUF100 the pytest subprocess above wrote this file (trusted input)
    return {case.get("name", ""): case for case in root.iter("testcase")}


def outcome(case: ET.Element) -> str:
    for tag in ("failure", "error", "skipped"):
        if case.find(tag) is not None:
            return tag
    return "passed"


def message(case: ET.Element) -> str:
    node = case.find("failure")
    if node is None:
        node = case.find("error")
    assert node is not None, "expected a failure or error element"
    return node.get("message", "") + (node.text or "")


def test_unmarked_test_over_budget_fails_naming_phase_and_budget(tmp_path):
    cases = run_project(tmp_path, "\ndef test_heavy():\n    busy(0.3)\n    assert True\n", TIGHT_BUDGET_S)

    assert outcome(cases["test_heavy"]) == "failure"
    assert "test_inner.py::test_heavy call took" in message(cases["test_heavy"])
    assert "(budget 0.1s)" in message(cases["test_heavy"])


def test_slow_fixture_setup_over_budget_is_charged_to_setup(tmp_path):
    source = "\nimport pytest\n\n@pytest.fixture\ndef model():\n    busy(0.3)\n    return 1\n\ndef test_uses_model(model):\n    assert model == 1\n"

    cases = run_project(tmp_path, source, TIGHT_BUDGET_S)

    assert outcome(cases["test_uses_model"]) == "error"
    assert "test_uses_model setup took" in message(cases["test_uses_model"])


@pytest.mark.parametrize("marker", ["slow", "integration", "e2e", "gpu", "symbolic"])
def test_tests_marked_for_deselection_are_exempt(tmp_path, marker):
    source = f"\nimport pytest\n\n@pytest.mark.{marker}\ndef test_heavy():\n    busy(0.3)\n    assert True\n"

    cases = run_project(tmp_path, source, TIGHT_BUDGET_S)

    assert outcome(cases["test_heavy"]) == "passed"


def test_module_level_slow_mark_exempts_every_test_in_the_file(tmp_path):
    source = "\nimport pytest\n\npytestmark = pytest.mark.slow\n\ndef test_heavy():\n    busy(0.3)\n    assert True\n"

    cases = run_project(tmp_path, source, TIGHT_BUDGET_S)

    assert outcome(cases["test_heavy"]) == "passed"


def test_test_within_budget_passes(tmp_path):
    cases = run_project(tmp_path, "\ndef test_heavy():\n    busy(0.3)\n    assert True\n", LOOSE_BUDGET_S)

    assert outcome(cases["test_heavy"]) == "passed"


def test_a_test_that_already_failed_keeps_its_own_failure(tmp_path):
    cases = run_project(tmp_path, "\ndef test_heavy():\n    busy(0.3)\n    assert 1 == 2\n", TIGHT_BUDGET_S)

    assert outcome(cases["test_heavy"]) == "failure"
    assert "budget" not in message(cases["test_heavy"])
    assert "assert 1 == 2" in message(cases["test_heavy"])


def test_budget_of_zero_disables_the_check(tmp_path):
    cases = run_project(tmp_path, "\ndef test_heavy():\n    busy(0.3)\n    assert True\n", "0")

    assert outcome(cases["test_heavy"]) == "passed"


def test_hypothesis_call_phase_is_left_to_its_per_example_deadline(tmp_path):
    source = (
        "\nfrom hypothesis import given, settings, strategies as st\n\n"
        "@settings(max_examples=10, deadline=None, database=None)\n"
        "@given(st.integers())\n"
        "def test_property(n):\n    busy(0.03)\n    assert isinstance(n, int)\n"
    )

    cases = run_project(tmp_path, source, TIGHT_BUDGET_S)

    assert outcome(cases["test_property"]) == "passed"


def test_hypothesis_test_with_a_slow_fixture_is_still_charged_for_setup(tmp_path):
    source = (
        "\nimport pytest\nfrom hypothesis import given, settings, strategies as st\n\n"
        "@pytest.fixture(scope='module')\ndef model():\n    busy(0.3)\n    return 1\n\n"
        "@settings(max_examples=2, database=None)\n"
        "@given(st.integers())\n"
        "def test_property(model, n):\n    assert model == 1\n"
    )

    cases = run_project(tmp_path, source, TIGHT_BUDGET_S)

    assert outcome(cases["test_property"]) == "error"
    assert "test_property setup took" in message(cases["test_property"])


def test_slow_fixture_teardown_over_budget_is_charged_to_teardown(tmp_path):
    source = "\nimport pytest\n\n@pytest.fixture\ndef db():\n    yield 1\n    busy(0.3)\n\ndef test_uses_db(db):\n    assert db == 1\n"

    cases = run_project(tmp_path, source, TIGHT_BUDGET_S)

    assert outcome(cases["test_uses_db"]) == "error"
    assert "test_uses_db teardown took" in message(cases["test_uses_db"])


def test_xfail_tests_are_left_to_xfail(tmp_path):
    source = "\nimport pytest\n\n@pytest.mark.xfail(reason='known', strict=False)\ndef test_heavy():\n    busy(0.3)\n    assert True\n"

    result = run_pytest(tmp_path, source, TIGHT_BUDGET_S)

    assert result.returncode == 0, result.stdout
    assert "budget" not in result.stdout


@pytest.mark.parametrize("raw", ["abc", "nan", "inf", "-1"])
def test_invalid_budget_stops_the_run_naming_the_variable(tmp_path, raw):
    result = run_pytest(tmp_path, "\ndef test_ok():\n    assert True\n", raw)

    assert result.returncode != 0
    assert f"PYTEST_EXPERT_BUDGET_S must be a finite number of seconds >= 0, got {raw!r}" in result.stdout + result.stderr


def test_empty_budget_means_the_default(tmp_path):
    cases = run_project(tmp_path, "\ndef test_heavy():\n    busy(0.3)\n    assert True\n", "")

    assert outcome(cases["test_heavy"]) == "passed"
