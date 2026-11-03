#!/usr/bin/env python3
"""Claims the skill makes about itself. Needs pytest; the frontmatter test skips without pyyaml."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

def test_frontmatter_parses_with_a_strict_yaml_parser() -> None:
    yaml = pytest.importorskip("yaml")
    text = (Path(__file__).parent.parent / "SKILL.md").read_text(encoding="utf-8")
    block = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert block is not None
    data = yaml.safe_load(block.group(1))
    assert data["name"] == "negative-space-programming"
    assert data["description"].startswith("Write and review code")


def test_shipped_helper_and_audit_script_audit_clean() -> None:
    """The audit script's docstring promises this; hold it to that."""
    import subprocess
    import sys

    root = Path(__file__).parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "audit_negative_space.py"),
         str(root / "assets" / "negative_space.py"), str(root / "scripts" / "audit_negative_space.py"),
         "--min-assertions", "1"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout
