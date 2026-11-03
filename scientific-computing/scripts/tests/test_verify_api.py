"""Regression tests for verify_api.py, run against real numpy/scipy releases.

Each case builds a throwaway environment with `uv run --with`, so the tests need
`uv` and network access to PyPI; they skip otherwise. Point VERIFY_API at another
copy of the script to test it instead of the one next to this file.

    python -m pytest scripts/tests -q
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(os.environ.get(
    "VERIFY_API", Path(__file__).resolve().parents[1] / "verify_api.py"))

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")


def run(numpy: str, scipy: str, python: str = "3.13") -> tuple[int, str]:
    cmd = ["uv", "run", "-q", "--no-project", "--python", python,
           "--with", f"numpy=={numpy}", "--with", f"scipy=={scipy}",
           "python", "-I", str(SCRIPT)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout


def test_pinned_versions_report_no_drift():
    code, out = run("2.4.4", "1.17.1")
    assert code == 0, out
    assert "0 problems" in out


def test_numpy_25_drift_is_reported():
    # numpy 2.5 removed np.row_stack and added np.sort(descending=); the old
    # script printed "0 problems" for numpy 2.5.4 + scipy 1.17.1.
    code, out = run("2.5.4", "1.17.1")
    assert code == 1, out
    assert "DRIFT    numpy.sort" in out


def test_scipy_118_behavior_changes_are_reported():
    code, out = run("2.5.4", "1.18.1")
    assert code == 1, out
    assert "CHANGED  batched scipy.linalg.svd returns a list" in out
    assert "CHANGED  cho_factor returns a bool" in out
    assert "CHANGED  scipy.linalg.lstsq residues" in out


def test_sentinel_defaults_print_without_memory_addresses():
    _, out = run("2.5.4", "1.18.1")
    assert "object object at 0x" not in out


def test_numpy_20_to_23_not_called_numpy_1x():
    code, out = run("2.3.5", "1.17.1")
    assert code == 1, out
    assert "STILL PRESENT: np.in1d, np.trapz" in out
    assert "numpy 1.x" not in out
