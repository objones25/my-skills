"""machine_stamp.py prints the date, machine and versions a result must be stated with."""
from __future__ import annotations

import re

from conftest import SCRIPTS  # noqa: F401


def test_stamp_has_a_date_and_architecture(run_script):
    line = run_script("machine_stamp.py").stdout.strip()
    assert re.match(r"^On \d{4}-\d{2}-\d{2} on \S.* with Python \d+\.\d+\.\d+", line)


def test_stamp_includes_requested_package_versions(run_script):
    line = run_script("machine_stamp.py", "--package", "pip").stdout.strip()
    assert re.search(r"pip \d", line) or "pip not installed" in line


def test_stamp_reports_a_missing_package_honestly(run_script):
    line = run_script("machine_stamp.py", "--package", "no-such-package-xyz").stdout.strip()
    assert "no-such-package-xyz not installed" in line


def test_stamped_result_satisfies_the_checker(skill, run_script, codes):
    stamp = run_script("machine_stamp.py").stdout.strip()
    (skill / "CHANGELOG.md").write_text(f"# Changelog\n\n{stamp}: 40 passed.\n", encoding="utf-8")
    assert "SB050" not in codes(skill)
