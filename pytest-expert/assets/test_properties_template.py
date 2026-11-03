"""Property-based test patterns, runnable as-is. Copy the pattern, swap in your function.

A property is a rule that holds for every input; Hypothesis generates inputs,
finds a failing one, shrinks it to the smallest failing case, and replays it on
every later run. Each pattern below targets one kind of bug.

Run: pytest assets/test_properties_template.py -p no:randomly
"""

from __future__ import annotations

import math
import re

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule


# --------------------------------------------------------------------------
# Toy system under test (replace with your own code)
# --------------------------------------------------------------------------
class AmountError(ValueError):
    """The only error the parser may raise."""


def format_cents(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def parse_cents(text: str) -> int:
    match = re.fullmatch(r"([0-9]{1,9})\.([0-9]{2})", text.strip())
    if not match:
        raise AmountError(f"invalid amount: {text!r}")
    return int(match[1]) * 100 + int(match[2])


def normalize_owner(name: str) -> str:
    return " ".join(name.split()).casefold()


def fee_cents(cents: int) -> int:
    """1%, minimum 25 cents. Integer arithmetic, the fast path."""
    return max(25, cents // 100)


def fee_cents_reference(cents: int) -> int:
    """Slow but obviously correct version, used as an oracle."""
    return max(25, math.floor(cents / 100)) if cents < 2**52 else max(25, cents // 100)


cents = st.integers(min_value=0, max_value=10**11 - 1)


# 1. Round trip: decode(encode(x)) == x. Catches lossy or asymmetric conversions.
@given(cents)
def test_round_trip(value):
    assert parse_cents(format_cents(value)) == value


# 2. Oracle / differential: fast implementation agrees with a reference.
@given(cents)
def test_fast_fee_matches_reference(value):
    assert fee_cents(value) == fee_cents_reference(value)


# 3. Invariant: output always inside its contract.
@given(cents)
def test_fee_is_at_least_the_minimum(value):
    assert fee_cents(value) >= 25


# 4. Idempotence: applying twice equals applying once.
@given(st.text())
def test_normalize_is_idempotent(name):
    assert normalize_owner(normalize_owner(name)) == normalize_owner(name)


# 5. Metamorphic: a known transformation of the input has a known effect on the output.
@given(cents, st.sampled_from([" ", "\t", "\n "]))
def test_surrounding_whitespace_does_not_change_the_value(value, pad):
    assert parse_cents(pad + format_cents(value) + pad) == value


# 6. Exception contract + postcondition: any input either raises the documented
#    error or returns a valid result. Never anything else.
@given(st.text(max_size=30))
@example("")
@example("1.5")
@example("\u0661.00")  # Arabic-Indic digit: rejected because the regex uses [0-9], not \d
def test_parser_raises_only_amount_error(text):
    try:
        value = parse_cents(text)
    except AmountError:
        return
    assert value >= 0


# 7. Known special values always run via @example; random draws hit them rarely.
@given(st.floats(allow_nan=True, allow_infinity=True))
@example(math.nan)
@example(math.inf)
@example(-0.0)
def test_isfinite_agrees_with_definition(x):
    assert math.isfinite(x) == (not math.isnan(x) and not math.isinf(x))


# 8. Stateful: random operation sequences checked against a simple model.
class CounterModel(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.real: dict[str, int] = {}
        self.model_total = 0

    @rule(key=st.sampled_from("abc"), amount=st.integers(min_value=1, max_value=100))
    def add(self, key, amount):
        self.real[key] = self.real.get(key, 0) + amount
        self.model_total += amount

    @invariant()
    def totals_agree(self):
        assert sum(self.real.values()) == self.model_total


TestCounterModel = CounterModel.TestCase


# Settings for an expensive property: fewer examples, no deadline.
@settings(max_examples=25, deadline=None)
@given(st.lists(cents, max_size=50))
def test_sum_of_fees_is_monotonic_in_list_length(values):
    assert sum(map(fee_cents, [*values, 0])) >= sum(map(fee_cents, values))


@pytest.mark.parametrize("text", ["12.50", "0.00"])
def test_example_based_tests_still_document_specific_cases(text):
    assert format_cents(parse_cents(text)) == text
