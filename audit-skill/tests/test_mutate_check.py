"""mutate_check.py: survivors are reported for weak tests, all killed for strong ones."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from conftest import SKILL_ROOT

SCRIPT = SKILL_ROOT / "scripts" / "mutate_check.py"
SOURCE = "def shipping_cents(subtotal):\n    if subtotal >= 5000:\n        return 0\n    return 499\n"
STRONG = (
    "from shop import shipping_cents\n\n\n"
    "def test_boundary_is_free():\n    assert shipping_cents(5000) == 0\n\n\n"
    "def test_just_under_pays():\n    assert shipping_cents(4999) == 499\n"
)
WEAK = "from shop import shipping_cents\n\n\ndef test_big_order_is_free():\n    assert shipping_cents(9000) == 0\n"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "shop.py").write_text(SOURCE)
    (tmp_path / "tests").mkdir()
    return tmp_path


def run(project, *extra):
    cmd = [sys.executable, str(SCRIPT), "shop.py", *extra, "--", sys.executable, "-m", "pytest", "-x", "-q",
           "-p", "no:cacheprovider", "tests/"]
    env = {"PYTHONPATH": str(project), "PATH": "/usr/bin:/bin", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    return subprocess.run(cmd, cwd=project, capture_output=True, text=True, timeout=300, env=env, check=False)


def test_strong_tests_kill_every_mutant(project):
    (project / "tests" / "test_shop.py").write_text(STRONG)
    proc = run(project)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "mutants killed (100%)" in proc.stdout


def test_weak_tests_leave_survivors_and_exit_1(project):
    (project / "tests" / "test_shop.py").write_text(WEAK)
    proc = run(project)
    assert proc.returncode == 1
    assert "SURVIVED line 2: >= -> >" in proc.stdout


def test_json_output_lists_each_mutant(project):
    (project / "tests" / "test_shop.py").write_text(WEAK)
    proc = run(project, "--format", "json")
    payload = json.loads(proc.stdout)
    statuses = {m["status"] for m in payload["mutants"]}
    assert statuses == {"SURVIVED", "killed"}


def test_source_is_restored_after_the_run(project):
    (project / "tests" / "test_shop.py").write_text(WEAK)
    run(project)
    assert (project / "shop.py").read_text() == SOURCE
    assert not list(project.glob("*.bak"))


def test_red_baseline_exits_2(project):
    (project / "tests" / "test_shop.py").write_text("def test_red():\n    assert 1 + 1 == 3\n")
    assert run(project).returncode == 2
