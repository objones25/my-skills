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
MAX_CHARS = 9000  # Claude Code caps hook text at 10,000 characters
# Config files the Stop gate fingerprints: a change to any of these can turn a green run red or lower the bar
# (fail_under, addopts, deselects). Union of the files pytest 9.2 reads (_pytest/config/findpaths.py) and the
# ones coverage reads; selftest/test_hooks.py checks it against the lists in scripts/.
CONFIG_FILES = (
    "pytest.toml", ".pytest.toml", "pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg",
    ".coveragerc", ".coveragerc.toml",
)
SKIP_DIRS = {".git", ".venv", "venv", ".tox", ".nox", "node_modules", "build", "dist", "mutants", "__pycache__", ".hypothesis"}


def find_script(name: str) -> Path | None:
    """audit_tests.py / diff_guard.py: next to the hook (as installed in .claude/hooks), else in the
    skill's scripts/ folder (the layout as shipped, assets/claude/hooks beside scripts/). None if neither."""
    if (HERE / name).is_file():
        return HERE / name
    skill_root = HERE.parents[2] if len(HERE.parents) > 2 else None
    if skill_root and (skill_root / "SKILL.md").is_file() and (skill_root / "scripts" / name).is_file():
        return skill_root / "scripts" / name
    return None


def missing_script_message(name: str) -> str:
    return (f"{name} was not found next to the hook ({HERE}) or in the skill's scripts/ folder, so that check "
            f"could not run. Fix: copy pytest-expert/scripts/{name} into {HERE}.")


def run_detector(label: str, cmd: list[str], *, timeout_s: int, cwd: Path) -> tuple[str | None, str | None]:
    """Run audit_tests.py or diff_guard.py. Contract: exit 0 = clean, 1 = findings on stdout,
    anything else (exit 2, a crash, a timeout, a missing interpreter) = could not check.

    Returns (findings_text, could_not_check_reason); both None means clean."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=cwd, check=False)
    except subprocess.TimeoutExpired:
        return None, f"{label} did not finish within {timeout_s}s, so it could not check anything."
    except OSError as exc:
        return None, f"{label} could not be started ({exc}); set PYTEST_EXPERT_PYTHON to a working interpreter."
    if proc.returncode == 0:
        return None, None
    if proc.returncode == 1 and proc.stdout.strip():
        return proc.stdout, None
    detail = (proc.stderr or proc.stdout).strip()
    return None, f"{label} could not check (exit {proc.returncode}): {tail(detail, 1500) or 'no output'}"


def int_env(name: str, default: int, *, allow_zero: bool = False) -> tuple[int, str | None]:
    """A positive integer (or zero, with allow_zero) from the environment; on a bad value return the default and a warning."""
    raw = os.environ.get(name)
    if raw is None:
        return default, None
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < (0 if allow_zero else 1):
        kind = "a non-negative" if allow_zero else "a positive"
        return default, f"{name}={raw!r} is not {kind} integer; using the default {default}."
    return value, None


def emit(**fields: object) -> None:
    print(json.dumps(fields))


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
    try:
        has_cov = subprocess.run([python, "-c", "import pytest_cov"], check=False, capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        has_cov = False  # the pytest run that follows reports the broken interpreter
    return cmd + (["--no-cov"] if has_cov else []) + list(targets)


def tail(text: str, limit: int = MAX_CHARS) -> str:
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def state_path(event: dict, project: Path) -> Path:
    """Per-session, per-project state file shared by the SessionStart and Stop hooks."""
    key = hashlib.sha256(f"{event.get('session_id', '')}:{project}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"pytest-expert-stop-{key}.json"


def read_state(path: Path) -> dict:
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def read_event() -> dict | None:
    """The hook's JSON event from stdin, or None when it is missing, malformed or not an object."""
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return event if isinstance(event, dict) else None


def git_head(project: Path) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(project), "rev-parse", "--verify", "HEAD^{commit}"],
                              check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None
