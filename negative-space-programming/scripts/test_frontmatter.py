#!/usr/bin/env python3
"""SKILL.md frontmatter must be valid YAML. Needs pytest and pyyaml; skips without pyyaml."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


def test_frontmatter_parses_with_a_strict_yaml_parser() -> None:
    text = (Path(__file__).parent.parent / "SKILL.md").read_text(encoding="utf-8")
    block = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert block is not None
    data = yaml.safe_load(block.group(1))
    assert data["name"] == "negative-space-programming"
    assert data["description"].startswith("Write and review code")
