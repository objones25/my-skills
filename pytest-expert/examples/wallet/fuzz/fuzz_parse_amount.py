"""Coverage-guided fuzz target for wallet.money.parse_amount (Atheris).

Run (needs pip install atheris):
    python fuzz/fuzz_parse_amount.py fuzz/corpus/parse_amount -max_total_time=60 -artifact_prefix=fuzz/crashes/

The oracle is the same contract the property tests check: parse_amount either
raises AmountError or returns a positive, in-range, two-place Decimal. Any other
exception, or a wrong value, is a crash. tests/security/test_fuzz_corpus_replay.py
replays the corpus and every saved crash under plain pytest.
"""

from __future__ import annotations

import sys
from decimal import Decimal

if __name__ == "__main__":
    import atheris

    with atheris.instrument_imports(include=["wallet"]):
        from wallet import money
else:
    from wallet import money


def check(text: str) -> None:
    try:
        value = money.parse_amount(text)
    except money.AmountError:
        return
    if not (Decimal(0) < value <= money.MAX_AMOUNT) or value != value.quantize(money.CENT):
        raise AssertionError(f"parse_amount({text!r}) returned {value!r}")


def test_one_input(data: bytes) -> None:
    check(data.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
