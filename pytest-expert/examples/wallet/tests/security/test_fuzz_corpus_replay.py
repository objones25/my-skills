"""Replay every fuzz corpus and crash input through the harness oracle.

Each crash Atheris ever found becomes a permanent regression test here, and
this file runs on every commit without needing Atheris installed.

The oracle in fuzz/fuzz_parse_amount.py accepts AmountError for any input, so on its
own it cannot tell a parser that rejects everything from a correct one. The corpus
files are therefore named by what the parser must do: a valid_* input must parse to
the value in EXPECTED_VALUE, an invalid_* input must raise AmountError.
"""

import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest
from wallet.money import AmountError, parse_amount

FUZZ_DIR = Path(__file__).resolve().parents[2] / "fuzz"
CORPUS = sorted(p for p in (FUZZ_DIR / "corpus" / "parse_amount").glob("*") if p.is_file())
CRASHES = sorted(p for p in (FUZZ_DIR / "crashes").glob("*") if p.is_file())

EXPECTED_VALUE = {
    "valid_plain": Decimal("12.50"),
    "valid_suffix": Decimal("12.50"),
    "valid_min": Decimal("0.01"),
    "valid_max": Decimal("1000000.00"),
}

pytestmark = pytest.mark.security


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("fuzz_parse_amount", FUZZ_DIR / "fuzz_parse_amount.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corpus_has_both_kinds_and_every_name_says_which():
    names = [p.name for p in CORPUS]
    assert [n for n in names if not n.startswith(("valid_", "invalid_"))] == []
    assert sum(n.startswith("valid_") for n in names) >= 4
    assert sum(n.startswith("invalid_") for n in names) >= 6


def test_every_valid_entry_has_an_expected_value():
    valid = {p.name for p in CORPUS if p.name.startswith("valid_")}
    assert valid == set(EXPECTED_VALUE)


@pytest.mark.parametrize("path", [p for p in CORPUS if p.name.startswith("valid_")], ids=lambda p: p.name)
def test_valid_corpus_input_parses_to_its_expected_value(path):
    assert parse_amount(path.read_bytes().decode("utf-8")) == EXPECTED_VALUE[path.name]


@pytest.mark.parametrize("path", [p for p in CORPUS if p.name.startswith("invalid_")], ids=lambda p: p.name)
def test_invalid_corpus_input_is_rejected(path):
    with pytest.raises(AmountError):
        parse_amount(path.read_bytes().decode("utf-8"))


@pytest.mark.parametrize("path", [*CORPUS, *CRASHES], ids=lambda p: p.name)
def test_fuzz_input_satisfies_the_oracle(harness, path):  # audit: ignore[NO_ASSERT] the oracle raises AssertionError itself
    harness.test_one_input(path.read_bytes())
