"""The fee property from tests/property, driven by an SMT solver instead of random draws.

Deselected by default. Run: pytest -m symbolic -p no:randomly
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from wallet.money import AmountError, transfer_fee_cents

pytest.importorskip("hypothesis_crosshair_provider", reason="pip install hypothesis-crosshair")


@settings(backend="crosshair", max_examples=200, deadline=None, database=None)
@given(st.integers(min_value=1, max_value=10**8), st.integers(min_value=10**9, max_value=10**10 - 1))
def test_fee_is_never_negative_for_any_solver_found_input(amount_cents, account):
    try:
        fee = transfer_fee_cents(amount_cents, account)
    except AmountError:
        return
    assert fee >= 0
