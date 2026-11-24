"""Self-tests for the pytest-expert tooling. Run: pytest selftest/ -q"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]


def load_script(relative: str):
    path = SKILL_ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def audit_mod():
    return load_script("scripts/audit_tests.py")


@pytest.fixture
def run_findings(tmp_path, audit_mod):
    """Write a test module (and optional extra files) into a hermetic project and audit it.

    The project root carries a `.git` marker, so the audit never looks above it: a seeding
    conftest.py, a strict pyproject.toml or a directory called `integration` above `tmp_path`
    (for example under `--basetemp`) cannot change the result. `under` places the project
    below `tmp_path`, `root` narrows the audited path below the project, and `extra` maps
    relative paths to file text.
    """

    def _run(
        source: str,
        name: str = "test_case.py",
        extra: dict[str, str] | None = None,
        *,
        under: str = "",
        root: str = "",
    ):
        project = tmp_path / under if under else tmp_path
        (project / ".git").mkdir(parents=True, exist_ok=True)
        for rel, text in (extra or {}).items():
            target = project / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.encode("utf-8"))
        return audit_mod.audit(project / root if root else project)

    return _run


@pytest.fixture
def run_audit(run_findings):
    """Same as run_findings, but returns the sorted list of codes found."""

    def _run(*args, **kwargs) -> list[str]:
        return sorted(f.code for f in run_findings(*args, **kwargs))

    return _run
