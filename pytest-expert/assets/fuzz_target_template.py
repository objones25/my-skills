"""Atheris fuzz target template. Copy to fuzz/fuzz_<name>.py and swap in your function.

Coverage-guided fuzzing: Atheris mutates bytes, keeps any input that reaches new
code, and reports an uncaught exception or a failed oracle as a crash.

Run (pip install atheris):
    python fuzz/fuzz_<name>.py fuzz/corpus/<name> -max_total_time=60 -artifact_prefix=fuzz/crashes/
Replay a crash without Atheris: call test_one_input(Path(crash).read_bytes()) from a pytest test.

Design rules:
  * The oracle is the function's contract, not "does not crash". Assert the postcondition.
  * Decode bytes deterministically (utf-8, errors="replace") so pytest can replay crashes.
  * Keep the target fast: no I/O, no sleeps, no network. Aim for thousands of execs/second.
"""

from __future__ import annotations

import sys


class ConfigError(ValueError):
    """The only error parse_pairs may raise."""


def parse_pairs(text: str) -> dict[str, str]:
    """Replace this toy with your own parser: 'a=1;b=2' -> {'a': '1', 'b': '2'}."""
    result: dict[str, str] = {}
    for chunk in filter(None, text.split(";")):
        key, sep, value = chunk.partition("=")
        if not sep or not key.isidentifier():
            raise ConfigError(f"bad pair: {chunk!r}")
        result[key] = value
    return result


def check(text: str) -> None:
    try:
        pairs = parse_pairs(text)
    except ConfigError:
        return
    rebuilt = ";".join(f"{k}={v}" for k, v in pairs.items())
    if parse_pairs(rebuilt) != pairs:
        raise AssertionError(f"round trip failed for {text!r}")


def test_one_input(data: bytes) -> None:
    check(data.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    import atheris

    atheris.instrument_func(parse_pairs)
    atheris.instrument_func(check)
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
