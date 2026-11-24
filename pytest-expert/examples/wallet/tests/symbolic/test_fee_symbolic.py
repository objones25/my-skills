"""The fee property from tests/property, driven by an SMT solver instead of a strategy.

tests/property builds valid accounts with a strategy that knows the mod-97 rule. This
test knows nothing about it: it draws plain ten-digit integers and lets the solver find
the ones that pass the check, so it also guards the day the strategy and the code drift apart.

Deselected by default. Needs hypothesis-crosshair (the "symbolic" extra); without it the
module is skipped. Run: pytest -m symbolic --no-cov -p no:randomly
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
