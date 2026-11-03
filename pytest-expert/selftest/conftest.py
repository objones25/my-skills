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
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def audit_mod():
    return load_script("scripts/audit_tests.py")


@pytest.fixture
def run_audit(tmp_path, audit_mod):
    """Write a test module (and optional extra files) and return the codes found."""

    def _run(source: str, name: str = "test_case.py", extra: dict[str, str] | None = None) -> list[str]:
        for rel, text in (extra or {}).items():
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        return [f.code for f in audit_mod.audit(tmp_path)]

    return _run
