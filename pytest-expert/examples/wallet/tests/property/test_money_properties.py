"""Properties: rules that must hold for every input, checked on generated inputs."""

from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st
from wallet.money import CENT, MAX_AMOUNT, AmountError, parse_amount, transfer_fee_cents

amounts = st.decimals(min_value=CENT, max_value=MAX_AMOUNT, places=2)
accounts = st.integers(min_value=10**9, max_value=10**10 - 1)


@given(amounts)
def test_round_trip_through_text(amount):
    assert parse_amount(f"{amount:f}") == amount


@given(amounts, st.sampled_from(["", " ", "\t", "  "]))
def test_whitespace_and_suffix_do_not_change_the_value(amount, pad):
    assert parse_amount(f"{pad}{amount:f} USD{pad}") == parse_amount(f"{amount:f}")


@given(st.text(max_size=40))
def test_any_text_either_parses_to_a_valid_amount_or_raises_amount_error(text):
    try:
        value = parse_amount(text)
    except AmountError:
        return
    assert Decimal(0) < value <= MAX_AMOUNT
    assert value == value.quantize(CENT)


@given(st.integers(min_value=1, max_value=10**8), accounts)
def test_fee_is_never_negative(amount_cents, account):
    try:
        fee = transfer_fee_cents(amount_cents, account)
    except AmountError:
        return
    assert fee >= 0
