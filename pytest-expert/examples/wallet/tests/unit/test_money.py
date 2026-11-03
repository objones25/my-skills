from decimal import Decimal

import pytest
from wallet.money import AmountError, parse_amount, transfer_fee_cents

PARTNER = 4200000091
REGULAR = 1700000079


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12.50", Decimal("12.50")),
        ("12.50 USD", Decimal("12.50")),
        ("  7 ", Decimal("7.00")),
        ("0.01", Decimal("0.01")),
        ("1000000", Decimal("1000000.00")),
    ],
    ids=["plain", "usd-suffix", "padded-integer", "minimum", "maximum"],
)
def test_parse_amount_accepts_valid_text(text, expected):
    assert parse_amount(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "abc", "-5", "0", "0.00", "1e3", "1000000.01", "12.345", "12,50"],
    ids=["empty", "letters", "negative", "zero", "zero-cents", "exponent", "over-max", "three-places", "comma"],
)
def test_parse_amount_rejects_invalid_text(text):
    with pytest.raises(AmountError, match="amount"):
        parse_amount(text)


@pytest.mark.parametrize(
    ("amount_cents", "account", "expected"),
    [(10_000, REGULAR, 100), (1_000, REGULAR, 25), (5_000, PARTNER, 5), (50_000, PARTNER, 0)],
    ids=["one-percent", "minimum-fee", "partner-rebate", "partner-rebate-floors-at-zero"],
)
def test_transfer_fee_cents(amount_cents, account, expected):
    fee = transfer_fee_cents(amount_cents, account)
    assert (fee, type(fee)) == (expected, int)


@pytest.mark.parametrize(
    ("amount_cents", "account"),
    [(0, REGULAR), (100, REGULAR + 1), (100, 123)],
    ids=["zero-amount", "bad-check-digit", "too-short"],
)
def test_transfer_fee_rejects_bad_input(amount_cents, account):
    with pytest.raises(AmountError, match=r"amount|account"):
        transfer_fee_cents(amount_cents, account)


SMALLEST_TEN_DIGIT = 1000000064
LARGEST_TEN_DIGIT = 9999999952
NINE_DIGIT_WITH_CHECK = 999999967
ELEVEN_DIGIT_WITH_CHECK = 10000000049


def test_transfer_fee_accepts_one_cent():
    assert transfer_fee_cents(1, REGULAR) == 25


@pytest.mark.parametrize("account", [SMALLEST_TEN_DIGIT, LARGEST_TEN_DIGIT], ids=["smallest", "largest"])
def test_transfer_fee_accepts_both_ends_of_the_account_range(account):
    assert transfer_fee_cents(10_000, account) == 100


@pytest.mark.parametrize("account", [NINE_DIGIT_WITH_CHECK, ELEVEN_DIGIT_WITH_CHECK], ids=["nine-digits", "eleven-digits"])
def test_transfer_fee_rejects_valid_check_digit_outside_ten_digits(account):
    with pytest.raises(AmountError, match="invalid account"):
        transfer_fee_cents(10_000, account)


def test_parse_amount_rejects_non_text_with_a_clear_message():
    with pytest.raises(AmountError, match="amount must be text, got float"):
        parse_amount(12.5)


def test_parse_amount_accepts_exactly_32_characters():
    assert parse_amount(" " * 27 + "12.50") == Decimal("12.50")


def test_parse_amount_rejects_33_characters():
    with pytest.raises(AmountError, match="too long"):
        parse_amount(" " * 28 + "12.50")
