"""Reference test module: every rule from SKILL.md, in working code.

One toy system is threaded through the whole file so the patterns compose
rather than sitting in isolation -- a `Wallet` with a balance, a withdrawal
rule, and a currency-rate lookup that hits the network (the boundary we stub).

Run it:
    pytest assets/test_example.py -v
    python scripts/audit_tests.py assets/     # must report zero findings
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from unittest import mock

import pytest

# --------------------------------------------------------------------------
# The system under test. In a real project this lives in src/, not here.
# --------------------------------------------------------------------------


class InsufficientFunds(ValueError):
    """Withdrawal exceeds the available balance."""


@dataclasses.dataclass
class Wallet:
    balance: float = 0.0
    currency: str = "USD"

    def deposit(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError(f"deposit must be positive, got {amount}")
        self.balance += amount
        return self.balance

    def withdraw(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError(f"withdrawal must be positive, got {amount}")
        if amount > self.balance:
            raise InsufficientFunds(
                f"cannot withdraw {amount} from balance {self.balance}"
            )
        self.balance -= amount
        return self.balance


def fetch_rate(currency: str) -> float:
    """The I/O boundary. Never called for real in a unit test."""
    import urllib.request

    with urllib.request.urlopen(f"https://rates.example/{currency}") as resp:
        return float(resp.read())


def convert(wallet: Wallet, to_currency: str) -> float:
    return wallet.balance * fetch_rate(to_currency)


def print_receipt(wallet: Wallet) -> None:
    """Writes to stdout. Printing is this function's actual contract."""
    print(f"receipt: {wallet.balance} {wallet.currency}")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def empty_wallet() -> Wallet:
    """Function-scoped by default: each test gets its own object.

    That default is the right one. A session-scoped mutable fixture would let
    test A's deposit change what test B sees, which is exactly the order
    dependence rule 11 forbids.
    """
    return Wallet()


@pytest.fixture
def funded_wallet(empty_wallet: Wallet) -> Wallet:
    """Fixtures compose: this one requests another by name.

    Building the funded case on top of the empty one keeps the setup honest --
    it goes through the same deposit() the production code uses, so a bug in
    deposit() shows up here rather than being papered over by hand-set state.
    """
    empty_wallet.deposit(100.0)
    return empty_wallet


@pytest.fixture
def make_wallet() -> Callable[..., Wallet]:
    """Factory fixture: for when a test needs several, or needs to vary one.

    Returning a *function* instead of an object is the escape hatch when a
    plain fixture would force you into a fixed shape.
    """

    def _make(balance: float = 0.0, currency: str = "USD") -> Wallet:
        return Wallet(balance=balance, currency=currency)

    return _make


# --------------------------------------------------------------------------
# Rules 1-3: one behavior, declarative name, arrange/act/assert
# --------------------------------------------------------------------------


def test_deposit_increases_balance(empty_wallet: Wallet) -> None:
    # Arrange
    amount = 25.0

    # Act
    result = empty_wallet.deposit(amount)

    # Assert
    assert result == pytest.approx(25.0)
    assert empty_wallet.balance == pytest.approx(25.0)


def test_withdraw_reduces_balance(funded_wallet: Wallet) -> None:
    remaining = funded_wallet.withdraw(30.0)
    assert remaining == pytest.approx(70.0)


# --------------------------------------------------------------------------
# Rule 4: no loops in test bodies -- parametrize instead
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("starting", "withdrawal", "expected"),
    [
        (100.0, 100.0, 0.0),      # exact balance is allowed
        (100.0, 0.01, 99.99),     # smallest meaningful withdrawal
        (100.0, 99.99, 0.01),     # leaves a residue
    ],
    ids=["withdraw-all", "withdraw-minimum", "leave-residue"],
)
def test_withdraw_boundaries(starting: float, withdrawal: float, expected: float) -> None:
    """ids= turns the report into readable names.

    Without them the report says test_withdraw_boundaries[100.0-0.01-99.99],
    which tells you the inputs but not the intent. With them, a failing case
    is `withdraw-minimum` and you know what broke before opening the file.
    """
    wallet = Wallet(balance=starting)
    assert wallet.withdraw(withdrawal) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Rules 8-10: exact values, approx for floats, specific exceptions
# --------------------------------------------------------------------------


def test_repeated_deposits_accumulate_exactly(empty_wallet: Wallet) -> None:
    """0.1 + 0.2 != 0.3 in binary floating point. approx encodes the tolerance.

    Written as `assert balance == 0.3` this test fails on every machine on
    earth. Written as `assert balance > 0.29` it also passes for a broken
    implementation returning 1000.0.
    """
    empty_wallet.deposit(0.1)
    empty_wallet.deposit(0.2)
    assert empty_wallet.balance == pytest.approx(0.3)


def test_withdraw_over_balance_raises_insufficient_funds(funded_wallet: Wallet) -> None:
    """match= is what makes this assertion mean something.

    Without it, the test passes if withdraw() raises InsufficientFunds for any
    reason at all -- including a typo that raises it on every call.
    """
    with pytest.raises(InsufficientFunds, match=r"cannot withdraw 500\.0"):
        funded_wallet.withdraw(500.0)

    # The failed withdrawal must not have moved money.
    assert funded_wallet.balance == pytest.approx(100.0)


@pytest.mark.parametrize("bad_amount", [0.0, -1.0, -0.01])
def test_withdraw_rejects_non_positive_amounts(funded_wallet: Wallet, bad_amount: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        funded_wallet.withdraw(bad_amount)


def test_raises_exposes_the_exception_for_further_assertions(empty_wallet: Wallet) -> None:
    """`as excinfo` when you need to inspect the exception itself."""
    with pytest.raises(InsufficientFunds) as excinfo:
        empty_wallet.withdraw(1.0)

    assert excinfo.value.args[0] == "cannot withdraw 1.0 from balance 0.0"
    assert isinstance(excinfo.value, ValueError)  # documents the hierarchy


# --------------------------------------------------------------------------
# Rules 14, 19-22: test doubles at the boundary
# --------------------------------------------------------------------------


def test_convert_uses_the_fetched_rate(
    funded_wallet: Wallet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """monkeypatch is the first choice: no mock library, no spec to get wrong.

    Note *where* it patches -- the name `fetch_rate` as resolved in this
    module, which is where convert() looks it up. Patching
    `urllib.request.urlopen` would work too but couples the test to an
    implementation detail two layers down.
    """
    monkeypatch.setattr(__name__ + ".fetch_rate", lambda _currency: 1.5)

    assert convert(funded_wallet, "EUR") == pytest.approx(150.0)


def test_convert_passes_the_target_currency_through(funded_wallet: Wallet) -> None:
    """When the *call* is the behavior under test, a mock earns its place.

    autospec=True makes the double adopt fetch_rate's real signature, so a
    call with the wrong number of arguments fails here instead of passing and
    breaking in production.
    """
    with mock.patch(__name__ + ".fetch_rate", autospec=True) as fake_rate:
        fake_rate.return_value = 0.8

        convert(funded_wallet, "GBP")

    # Real assertion methods. `fake_rate.called_once_with("GBP")` would be a
    # Mock attribute that is truthy no matter what -- the FAKE_MOCK_ASSERT trap.
    fake_rate.assert_called_once_with("GBP")


def test_autospec_rejects_a_wrong_signature() -> None:
    """Proof that autospec is doing something, not just decoration."""
    with (
        mock.patch(__name__ + ".fetch_rate", autospec=True) as fake_rate,
        pytest.raises(TypeError, match="too many positional arguments"),
    ):
        fake_rate("USD", "extra")


# --------------------------------------------------------------------------
# Rules 16-17: tmp_path and monkeypatch for filesystem and environment
# --------------------------------------------------------------------------


def test_wallet_serialises_to_a_file(tmp_path, funded_wallet: Wallet) -> None:
    """tmp_path is a pathlib.Path unique to this test, cleaned up automatically.

    Writing to a relative path here would litter the repo and make the test
    order-dependent the moment two tests pick the same filename.
    """
    target = tmp_path / "wallet.txt"
    target.write_text(f"{funded_wallet.balance}:{funded_wallet.currency}")

    assert target.read_text() == "100.0:USD"
    assert target.parent.exists()


def test_default_currency_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """setenv is undone at teardown, so the next test starts clean.

    `os.environ["APP_CURRENCY"] = "JPY"` would leak into every later test in
    the process, and the failure would surface somewhere unrelated.
    """
    monkeypatch.setenv("APP_CURRENCY", "JPY")

    import os

    assert os.environ["APP_CURRENCY"] == "JPY"


# --------------------------------------------------------------------------
# Rules 23-25: markers with reasons, strict xfail
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_many_deposits_stay_accurate(empty_wallet: Wallet) -> None:
    """Marked slow, so the default selection skips it. Run with `-m slow`."""
    for _ in range(10_000):
        empty_wallet.deposit(0.01)
    assert empty_wallet.balance == pytest.approx(100.0)


@pytest.mark.xfail(
    strict=True,
    reason="Wallet has no currency conversion on deposit yet; see issue #421",
)
def test_deposit_converts_foreign_currency() -> None:
    """strict=True is what makes xfail useful rather than a hiding place.

    When someone implements the feature, this test starts passing -- and
    strict turns that unexpected pass into a FAILURE that says 'remove the
    xfail marker'. Non-strict xfail would report xpass and stay green, and the
    marker would sit there forever.
    """
    wallet = Wallet(currency="USD")
    wallet.deposit(100.0)
    assert wallet.currency == "EUR"


@pytest.mark.skipif(
    not hasattr(Wallet, "transfer"),
    reason="transfer() not implemented yet; tracked in issue #422",
)
def test_transfer_moves_between_wallets() -> None:
    """skipif with a reason: the report says why, not just 's'."""
    pytest.fail("unreachable until transfer() exists")


# --------------------------------------------------------------------------
# Output as behavior: capsys and caplog
# --------------------------------------------------------------------------


def test_audit_log_records_the_withdrawal(
    caplog: pytest.LogCaptureFixture, funded_wallet: Wallet
) -> None:
    """When logging *is* the contract, assert on it -- do not print.

    caplog captures log records; capsys captures stdout/stderr. Both give you
    a real assertion. A bare print() gives you nothing a test can check, which
    is why rule PRINT bans it.
    """
    import logging

    with caplog.at_level(logging.INFO):
        logging.getLogger("wallet").info("withdraw %s", 30.0)
        funded_wallet.withdraw(30.0)

    assert "withdraw 30.0" in caplog.text
    assert caplog.records[0].levelno == logging.INFO


def test_receipt_is_printed(capsys: pytest.CaptureFixture[str], funded_wallet: Wallet) -> None:
    """The *code* prints; the test asserts on what was captured.

    Note the shape here. The test never calls print() itself -- that would be
    the PRINT violation, and it would assert nothing. It calls the function
    whose job is to print, then reads stdout back and checks it.
    """
    print_receipt(funded_wallet)

    captured = capsys.readouterr()
    assert captured.out == "receipt: 100.0 USD\n"
    assert captured.err == ""


# --------------------------------------------------------------------------
# The factory fixture in use
# --------------------------------------------------------------------------


def test_two_wallets_are_independent(make_wallet: Callable[..., Wallet]) -> None:
    a = make_wallet(balance=50.0)
    b = make_wallet(balance=10.0, currency="EUR")

    a.withdraw(50.0)

    assert a.balance == pytest.approx(0.0)
    assert b.balance == pytest.approx(10.0)
    assert b.currency == "EUR"
