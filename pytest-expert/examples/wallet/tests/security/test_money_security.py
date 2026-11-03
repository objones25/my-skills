"""Security regressions for the parsing boundary: size limits, odd encodings, special values."""

import pytest
from wallet.money import AmountError, parse_amount

pytestmark = pytest.mark.security


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1" * 100_000, "too long"),
        ("1e999999999", "invalid amount"),
        ("NaN", "invalid amount"),
        ("Infinity", "invalid amount"),
        ("\u0661\u0662", "invalid amount"),
        ("\uff11\uff12", "invalid amount"),
    ],
    ids=["huge-input", "huge-exponent", "nan", "infinity", "arabic-indic-digits", "fullwidth-digits"],
)
def test_hostile_amount_text_is_rejected(text, message):
    with pytest.raises(AmountError, match=message):
        parse_amount(text)


@pytest.mark.parametrize("text", ["5 USD\n", "\t5\n"], ids=["trailing-newline", "tabs-and-newline"])
def test_surrounding_whitespace_is_stripped_not_smuggled(text):
    assert str(parse_amount(text)) == "5.00"
