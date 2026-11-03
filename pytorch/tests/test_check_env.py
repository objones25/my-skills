"""check_env.py runs on any install and reports the torch version it found."""
from __future__ import annotations

import subprocess
import sys

import torch


def test_check_env_reports_installed_torch(skill_root):
    proc = subprocess.run([sys.executable, str(skill_root / "scripts" / "check_env.py")],
                          capture_output=True, text=True, timeout=300, check=True)
    assert torch.__version__ in proc.stdout


def test_check_env_names_the_accelerator_line(skill_root):
    proc = subprocess.run([sys.executable, str(skill_root / "scripts" / "check_env.py")],
                          capture_output=True, text=True, timeout=300, check=True)
    assert any(line.startswith("accelerator ") for line in proc.stdout.splitlines())
