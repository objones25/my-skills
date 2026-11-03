"""The input boundary: turning untrusted text into amounts, and computing fees."""

from __future__ import annotations

import re
from decimal import Decimal


class AmountError(ValueError):
    """Any malformed or out-of-range amount. The only exception parse_amount raises."""


CENT = Decimal("0.01")
MAX_AMOUNT = Decimal("1000000.00")
MAX_TEXT_LEN = 32
# ASCII digits only: \d would also accept Arabic-Indic and other Unicode digits.
_AMOUNT = re.compile(r"[0-9]{1,7}(?:\.[0-9]{1,2})?")


def parse_amount(text: str) -> Decimal:
    """Parse '12.50' or '12.50 USD' into a Decimal with exactly two places."""
    if not isinstance(text, str):
        raise AmountError(f"amount must be text, got {type(text).__name__}")
    if len(text) > MAX_TEXT_LEN:
        raise AmountError(f"amount text too long ({len(text)} chars)")
    raw = text.strip().removesuffix(" USD").strip()
    if not _AMOUNT.fullmatch(raw):
        raise AmountError(f"invalid amount: {text!r}")
    value = Decimal(raw).quantize(CENT)
    if not (Decimal(0) < value <= MAX_AMOUNT):
        raise AmountError(f"amount out of range: {text!r}")
    return value


def transfer_fee_cents(amount_cents: int, account: int) -> int:
    """
    Fee in cents for sending amount_cents to a 10-digit account number.

    Account numbers carry a mod-97 check: account % 97 == 1.
    Partner bank (prefix 42): flat 10 cents minus a 0.1% loyalty rebate, floored at 0.
    Everyone else: 1%, minimum 25 cents.

    pre: amount_cents > 0
    post: __return__ >= 0
    raises: AmountError
    """
    if amount_cents <= 0:
        raise AmountError("amount must be positive")
    if not (10**9 <= account < 10**10) or account % 97 != 1:
        raise AmountError("invalid account number")
    if account // 10**8 == 42:
        return max(0, 10 - amount_cents // 1000)
    return max(25, amount_cents // 100)
