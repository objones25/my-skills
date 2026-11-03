# Demo: planted bugs, and which technique finds each

The v0 modules here are wrong on purpose. `test_v0_findings.py` aims one technique at
each bug; six of its seven tests are expected to fail. Nothing in this directory is
collected by the normal test run (`testpaths = ["tests"]`).

Run from `examples/wallet/`:

```
PYTHONPATH=demo python -m pytest demo -p no:randomly --no-cov -q
```

## The bugs

| v0 file | Bug | Found by | Missed by |
|---|---|---|---|
| `wallet_v0.py` | `withdraw(nan)` passes both checks (`nan <= 0` and `nan > balance` are both false) and the balance becomes `nan` | Hypothesis with `@example(math.nan)`, every run | plain `st.floats()`: NaN appeared in about 1% of draws, so one run passed 100 of 100 examples |
| `money_v0.py` | `parse_amount("")` leaks `decimal.InvalidOperation` instead of `AmountError` | Hypothesis `st.text()`; Atheris on its first input | example tests |
| `money_v0.py` | `parse_amount("0.001")` passes the positivity check, then rounds to `0.00` | Hypothesis (reported alongside the leak as a second distinct failure) | example tests |
| `money_v0.py` | partner-bank fee `10 - amount // 1000` goes negative, only for accounts that pass a mod-97 check | Hypothesis with `backend="crosshair"`; `crosshair check` | Hypothesis random search, 3 of 3 runs at 1,000 examples |
| `store_v0.py` | SQL built with an f-string | the injection payload tests; ruff `S608` without running anything | |

## Captured output

Test summary:

```
FAILED demo/test_v0_findings.py::test_withdraw_never_leaves_a_non_finite_balance_random_only
FAILED demo/test_v0_findings.py::test_withdraw_never_leaves_a_non_finite_balance
FAILED demo/test_v0_findings.py::test_parse_amount_raises_only_amount_error
FAILED demo/test_v0_findings.py::test_fee_never_negative_solver_search - asse...
FAILED demo/test_v0_findings.py::test_get_returns_nothing_for_injection_payloads[tautology]
FAILED demo/test_v0_findings.py::test_get_returns_nothing_for_injection_payloads[like-wildcard]
6 failed, 1 passed in 4.03s
```

The one that passed is `test_fee_never_negative_random_search`: random search missing
the checksum-gated bug. `..._random_only` failed in this run and passed in an earlier
one; that inconsistency is the reason to use `@example`.

Values found:

```
amount = 11000, account = 4200000091
E   assert -1 >= 0
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
E   assert ('alice', '10.00') is None
```

Atheris:

```
PYTHONPATH=demo python demo/fuzz_v0.py -runs=200000 -artifact_prefix=/tmp/v0-
```

```
 === Uncaught Python exception: ===
InvalidOperation: [<class 'decimal.ConversionSyntax'>]
artifact_prefix='/tmp/v0-'; Test unit written to /tmp/v0-crash-da39a3ee5e6b4b0d3255bfef95601890afd80709
```

CrossHair on the docstring contract (from `demo/`):

```
crosshair check --analysis_kind=PEP316 --per_condition_timeout=20 money_v0.transfer_fee_cents
```

```
money_v0.py:27: error: false when calling transfer_fee_cents(11000, 4200000091) (which returns -1)
```

Static analysis:

```
ruff check --select S608 --output-format concise demo/
```

```
demo/store_v0.py:13:32: S608 Possible SQL injection vector through string-based query construction
```

The fixed versions of each module are in `src/wallet/`.

## Re-run notes (November 2026)

Re-run on Python 3.13 with pytest 9.2.0, Hypothesis 6.170.14, CrossHair 0.0.112: same
6 failed, 1 passed. Two things changed underneath:

- Python 3.13 emits `ResourceWarning` for a `sqlite3.Connection` that is garbage
  collected without `close()` (3.12 does not; checked with `-X dev` on both). Under
  `filterwarnings = ["error"]` that surfaced as `PytestUnraisableExceptionWarning` and
  crashed pytest at exit. The store test now closes its connection in a fixture.
- Hypothesis 6.170 writes `.hypothesis/patches/<date>--<hash>.patch` after a failure.
  `git apply` on it adds each failing input as an `@example(...).via('discovered
  failure')`, which is the "turn findings into permanent tests" step done for you.
  Review the patch before applying; delete it for planted-bug demos like this one.
