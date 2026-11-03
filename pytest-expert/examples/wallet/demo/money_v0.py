"""v0 of wallet.money with three planted bugs. See demo/README.md. Do not import from src."""

from decimal import Decimal

class AmountError(ValueError):
    pass

CENT = Decimal("0.01")

def parse_amount(text: str) -> Decimal:
    raw = text.strip()
    if raw.endswith(" USD"):
        raw = raw[:-4]
    value = Decimal(raw)
    if value <= 0:
        raise AmountError(f"amount must be positive: {text!r}")
    return value.quantize(CENT)

def transfer_fee_cents(amount_cents: int, account: int) -> int:
    """
    Fee for sending amount_cents to a 10-digit account number.
    Account numbers carry a mod-97 check digit: account % 97 == 1.
    Partner bank (prefix 42): flat 10 cents minus a 0.1% loyalty rebate.
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
        return 10 - amount_cents // 1000
    return max(25, amount_cents // 100)
