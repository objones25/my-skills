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
├── conftest.py      copy of assets/conftest.py (seeding, network guard, profiles, markers)
├── unit/            example tests, parametrized
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

| Command | Runs | Measured result |
|---|---|---|
| `pytest` | unit, property, security, corpus replay | 87 passed, 97% branch coverage |
| `pytest -m integration --no-cov` | SQLite on disk | 2 passed |
| `pytest -m symbolic --no-cov -p no:randomly` | solver-backed fee property | 1 passed |
| `python ../../scripts/audit_tests.py tests` | the audit | clean |
| `python fuzz/fuzz_parse_amount.py fuzz/corpus/parse_amount -max_total_time=60 -artifact_prefix=fuzz/crashes/` | Atheris | 16.5 million runs (machine not recorded) in 46 s, no crash |
| `crosshair check --analysis_kind=PEP316 src/` | contract check on docstrings | no findings |
| `python ../../scripts/mutate_check.py src/wallet/money.py --function transfer_fee_cents -- python -m pytest -x -q -p no:randomly --no-cov tests/unit` | targeted mutation | 32 of 34 killed; 2 equivalent |
| `mutmut run --max-children 4` then `mutmut results` | whole-package mutation | 183 of 194 killed |

`crosshair check` needs `src` on the import path: run it with `PYTHONPATH=src` set in
the environment.

## Exercises left on purpose

mutmut reports 11 survivors. Seven are worth a test:

| Survivor | Missing test |
|---|---|
| error messages mutated to `"XX...XX"` in `transfer_fee_cents` | anchor the `match=` pattern (`r"^amount must be positive$"`) |
| `balance < 0` to `<= 0` in `load_wallet` | load a stored wallet with a zero balance (must succeed) and a negative one (must fail) |
| `or` to `and` in the same check | load a file whose `owner` does not match its name |

The rest are equivalent mutants. `references/mutation-testing.md` walks through how
each was classified.

## The demo

`demo/README.md` shows each technique finding a planted bug in v0 code, with the
captured output.
