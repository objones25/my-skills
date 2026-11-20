# Worked example: wallet

A small package tested at every layer the skill covers. The code is deliberately
tiny so each layer's purpose is visible.

| Module | What it does | Risk it carries |
|---|---|---|
| `src/wallet/money.py` | parses untrusted amount text; computes transfer fees | malformed input, exact-value logic (mod-97 check digit) |
| `src/wallet/core.py` | wallet balance, withdraw, transfer | state, invariants (money is conserved) |
| `src/wallet/store.py` | JSON files under a base directory; SQLite store | path traversal, SQL and wildcard injection, unsafe deserialization |

## Layout

```
tests/
├── conftest.py      copy of assets/conftest.py (seeding, network guard, profiles, markers); the only
│                    difference is APP_ENV_PREFIXES, which is ("WALLET_",) here
├── unit/            example tests, parametrized; test_conftest_safety_net.py proves the guards act
├── property/        Hypothesis properties and a stateful model
├── security/        payload tables, security properties, fuzz corpus replay
├── integration/     real SQLite on disk; auto-marked, deselected by default
└── symbolic/        Hypothesis with the CrossHair backend; auto-marked, deselected by default
fuzz/
├── fuzz_parse_amount.py   Atheris target
└── corpus/parse_amount/   seed inputs (crashes go in fuzz/crashes/)
demo/                      v0 modules with planted bugs and the tests that find them
```

## Running each layer

Install from this directory with `pip install -e ".[test,symbolic,fuzz,mutation]"`.

Measured on 2026-11-20, macOS arm64, Python 3.13.2, pytest 9.2.0, pytest-cov 7.1.0 (coverage 7.17.3),
pytest-randomly 5.1.0, Hypothesis 6.170.17, crosshair-tool 0.0.114, hypothesis-crosshair 0.0.32,
mutmut 3.8.1. Without hypothesis-crosshair installed the default `pytest` run reports
`135 passed, 1 skipped, 2 deselected` instead (the symbolic module skips itself); the passed
count and the coverage are the same.

| Command | Runs | Measured result |
|---|---|---|
| `pytest` | unit, property, security, corpus replay | 135 passed, 3 deselected, 100% branch coverage (2026-11-20) |
| `pytest -m integration --no-cov` | SQLite on disk | 2 passed (2026-11-20) |
| `pytest -m symbolic --no-cov -p no:randomly` | solver-backed fee property | 1 passed, needs `hypothesis-crosshair` (2026-11-20) |
| `python ../../scripts/audit_tests.py tests` | the audit | clean, exit 0 (2026-11-20) |
| `python fuzz/fuzz_parse_amount.py fuzz/corpus/parse_amount -max_total_time=60 -artifact_prefix=fuzz/crashes/` | Atheris | 16.5 million runs (machine not recorded) in 46 s, no crash; not rerun on 2026-11-20 because Atheris could not be installed on this machine |
| `crosshair check --analysis_kind=PEP316 src/` | contract check on docstrings | no findings, exit 0 (2026-11-20) |
| `python ../../scripts/mutate_check.py src/wallet/money.py --function transfer_fee_cents -- python -m pytest -x -q -p no:randomly --no-cov tests/unit` | targeted mutation | 32 of 34 killed; 2 equivalent (2026-11-20, 74 s) |
| `mutmut run --max-children 4` then `mutmut results` | whole-package mutation | 190 of 196 killed (2026-11-20, 12 s, run in a scratch copy of this directory) |

`crosshair check` needs `src` on the import path: run it with `PYTHONPATH=src` set in
the environment.

## Surviving mutants

mutmut reports 6 survivors, and all six are equivalent: no test can tell them apart
from the original.

| Survivor | Why it is equivalent |
|---|---|
| `encoding="utf-8"` in `save_wallet` mutated to `None`, omitted, and `"UTF-8"` | `json.dumps` escapes non-ASCII, so the file is pure ASCII under any encoding; codec names are case-insensitive |
| `decode("utf-8")` mutated to `decode("UTF-8")` in `load_wallet` | same: codec names are case-insensitive |
| `10**9 <= account` mutated to `<`, and `account < 10**10` mutated to `<=` in `transfer_fee_cents` | `10**9 % 97` is 34 and `10**10 % 97` is 49, so neither boundary value passes the mod-97 check either way |

Earlier versions of this example left seven non-equivalent survivors (unanchored
`match=` patterns, and the `load_wallet` consistency check) as exercises. They now have
tests: `tests/unit/test_money.py` anchors the fee messages, and
`tests/security/test_store_security.py` loads files with a wrong owner, a negative,
NaN and infinite balance, and a zero balance (which must succeed).
`references/mutation-testing.md` walks through how survivors are classified.

## The demo

`demo/README.md` shows each technique finding a planted bug in v0 code, with the
captured output.
