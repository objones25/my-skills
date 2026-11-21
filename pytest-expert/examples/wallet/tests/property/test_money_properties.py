"""Properties: rules that must hold for every input, checked on generated inputs."""

from decimal import Decimal

from hypothesis import given
from hypothesis import strategies as st
from wallet.money import CENT, MAX_AMOUNT, AmountError, parse_amount, transfer_fee_cents

amounts = st.decimals(min_value=CENT, max_value=MAX_AMOUNT, places=2)

# Account numbers must be ten digits and satisfy account % 97 == 1. Drawing any ten-digit
# integer and returning early on AmountError would skip the fee code on every draw (a
# random number passes the check once in 97), so the strategy builds valid numbers.
_FIRST_K = -(-(10**9 - 1) // 97)
_LAST_K = (10**10 - 2) // 97
_PARTNER_FIRST_K = -(-(42 * 10**8 - 1) // 97)
_PARTNER_LAST_K = (43 * 10**8 - 2) // 97
valid_accounts = st.integers(min_value=_FIRST_K, max_value=_LAST_K).map(lambda k: 97 * k + 1)
partner_accounts = st.integers(min_value=_PARTNER_FIRST_K, max_value=_PARTNER_LAST_K).map(lambda k: 97 * k + 1)


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


@given(valid_accounts)
def test_the_account_strategy_only_builds_accounts_the_fee_accepts(account):
    assert 10**9 <= account < 10**10
    assert transfer_fee_cents(100, account) >= 0  # raises AmountError if the check digit is wrong


@given(partner_accounts)
def test_the_partner_strategy_only_builds_partner_accounts(account):
    assert account // 10**8 == 42
    assert transfer_fee_cents(100, account) >= 0


@given(st.integers(min_value=1, max_value=10**8), st.one_of(valid_accounts, partner_accounts))
def test_fee_is_never_negative(amount_cents, account):
    # No try/except: every drawn input is valid, so an AmountError here is a failure,
    # and the assertion below runs on every example.
    assert transfer_fee_cents(amount_cents, account) >= 0
