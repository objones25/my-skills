"""Shared helpers for the pytest-expert Claude Code hooks. Stdlib only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "audit_tests.py"
GUARD = HERE / "diff_guard.py"
MAX_CHARS = 9000  # Claude Code caps hook text at 10,000 characters
SKIP_DIRS = {".git", ".venv", "venv", ".tox", ".nox", "node_modules", "build", "dist", "mutants", "__pycache__", ".hypothesis"}


def project_dir(event: dict) -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or ".").resolve()


def project_python(project: Path) -> str:
    """The interpreter that has the project's test dependencies, not whatever python3 is first on PATH."""
    override = os.environ.get("PYTEST_EXPERT_PYTHON")
    if override:
        return override
    for candidate in (project / ".venv" / "bin" / "python", project / "venv" / "bin" / "python",
                      project / ".venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def pytest_command(python: str, *targets: str) -> list[str]:
    cmd = [python, "-m", "pytest", "-q", "-x", "--tb=short", "-p", "no:randomly", "-p", "no:cacheprovider"]
    has_cov = subprocess.run([python, "-c", "import pytest_cov"], capture_output=True).returncode == 0
    return cmd + (["--no-cov"] if has_cov else []) + list(targets)


def tail(text: str, limit: int = MAX_CHARS) -> str:
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def state_path(event: dict, project: Path) -> Path:
    """Per-session, per-project state file shared by the SessionStart and Stop hooks."""
    key = hashlib.sha256(f"{event.get('session_id', '')}:{project}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"pytest-expert-stop-{key}.json"


def read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def git_head(project: Path) -> str | None:
    proc = subprocess.run(["git", "-C", str(project), "rev-parse", "--verify", "HEAD^{commit}"],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None
