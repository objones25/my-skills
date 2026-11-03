"""Atheris harness for demo/money_v0.parse_amount (the buggy version).

Run from examples/wallet:  python demo/fuzz_v0.py -runs=100000 -artifact_prefix=/tmp/v0-
"""

import sys
from decimal import Decimal

import atheris

with atheris.instrument_imports():
    import money_v0


def test_one_input(data: bytes) -> None:
    text = data.decode("utf-8", errors="replace")
    try:
        value = money_v0.parse_amount(text)
    except money_v0.AmountError:
        return
    if not value > Decimal(0):
        raise AssertionError(f"parse_amount({text!r}) returned {value!r}")


atheris.Setup(sys.argv, test_one_input)
atheris.Fuzz()
