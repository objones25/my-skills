"""quote_match.py: a quote counts only if it is really in the fetched text."""

from __future__ import annotations

import pytest
from conftest import run_script
from quote_match import quote_found, quote_problem

TEXT = (
    "The results show a statistically signifi-\ncant correlation between mutant detection and real fault "
    "detection, independent of code coverage. 73% of the real faults were coupled to common mutants.\n"
    "We call this a “well-known” effect — see Section 4.\n"
)


@pytest.mark.parametrize("quote", [
    "independent of code coverage",
    "73% of the real faults were coupled to common mutants",
    "a statistically significant correlation between mutant detection",
    'We call this a "well-known" effect - see Section 4.',
    "The results show ... coupled to common mutants",
    "INDEPENDENT OF CODE COVERAGE",
])
def test_found(quote):
    assert quote_found(quote, TEXT)


@pytest.mark.parametrize("quote", [
    "93% of the real faults were coupled to common mutants",
    "coupled to common mutants ... The results show",
    "dependent on code coverage only",
])
def test_not_found(quote):
    assert not quote_found(quote, TEXT)


def test_short_quotes_are_a_problem():
    assert quote_problem("73% of faults") is not None
    assert quote_problem("independent of code coverage ... 73%") is not None
    assert quote_problem("73% of the real faults were coupled to common mutants") is None


def test_cli_exit_codes(tmp_path):
    src = tmp_path / "s.txt"
    src.write_text(TEXT, encoding="utf-8")
    assert run_script("quote_match.py", src, "independent of code coverage").returncode == 0
    assert run_script("quote_match.py", src, "dependent on code coverage only").returncode == 1
    assert run_script("quote_match.py", tmp_path / "missing.txt", "x").returncode == 2


@pytest.mark.parametrize("source,quote", [
    ("The --strict option is not deprecated in pytest 9.", "The --strict option is ... deprecated in pytest 9"),
    ("Mutants were not a valid substitute for real faults.", "Mutants were [...] a valid substitute for real faults"),
    ("Mutants were never a valid substitute for real faults.", "Mutants were ... a valid substitute for real faults"),
    ("We trained on 10⁵ examples in total.", "We trained on 105 examples in total"),
    ("A range of 10 - 20 cases was used here.", "A range of 10 -20 cases was used here"),
])
def test_meaning_changing_quotes_are_rejected(source, quote):
    assert not quote_found(quote, source)


def test_ellipsis_gap_is_bounded():
    source = "The first finding is stated here clearly. " + "Filler sentence. " * 100 + "The last finding is here."
    assert not quote_found("The first finding is stated ... The last finding is here", source)
    near = "The first finding is stated here clearly. Then the last finding is here."
    assert quote_found("The first finding is stated ... the last finding is here", near)


@pytest.mark.parametrize("word", ["rarely", "only", "seldom", "hardly"])
def test_hedges_cannot_be_elided(word):
    source = f"In our tests this method {word} improves accuracy on standard benchmarks."
    assert not quote_found("this method ... improves accuracy on standard benchmarks", source)
