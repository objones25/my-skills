from __future__ import annotations

import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("scripts", "examples", "examples/ports_adapters", "evals"):
    sys.path.insert(0, str(ROOT / sub))


@pytest.fixture
def write(tmp_path: Path) -> Callable[[str, str], Path]:
    """write("pkg/mod.py", source) -> path; dedents the source."""
    def _write(rel: str, source: str = "") -> Path:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip())
        return path
    return _write
