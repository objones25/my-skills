"""Replay every fuzz corpus and crash input through the harness oracle.

Each crash Atheris ever found becomes a permanent regression test here, and
this file runs on every commit without needing Atheris installed.
"""

import importlib.util
from pathlib import Path

import pytest

FUZZ_DIR = Path(__file__).resolve().parents[2] / "fuzz"
INPUTS = sorted(p for d in ("corpus/parse_amount", "crashes") for p in (FUZZ_DIR / d).glob("*") if p.is_file())

pytestmark = pytest.mark.security


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("fuzz_parse_amount", FUZZ_DIR / "fuzz_parse_amount.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corpus_is_not_empty():
    assert len(INPUTS) >= 5


@pytest.mark.parametrize("path", INPUTS, ids=lambda p: p.name)
def test_fuzz_input_satisfies_the_oracle(harness, path):  # audit: ignore[NO_ASSERT] the oracle raises AssertionError itself
    harness.test_one_input(path.read_bytes())
