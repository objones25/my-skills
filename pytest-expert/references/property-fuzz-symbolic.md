# Property-based testing, fuzzing, and symbolic execution

Three ways to make the computer pick the inputs. They find different bugs, and the
difference is measurable. Every output below is from a real run on
`examples/wallet` (Hypothesis 6.168.3, CrossHair 0.0.111, hypothesis-crosshair
0.0.30, Atheris 3.1.0, Python 3.12, October 2026). Re-run November 2026 on Hypothesis
6.170.14, CrossHair 0.0.112, hypothesis-crosshair 0.0.31, Python 3.13: same findings;
the solver search took about 0.4 s of test time in 3 of 3 runs (was about 1.7 s).

## Contents

- [Terms](#terms)
- [Which tool finds which bug](#which-tool-finds-which-bug)
- [Hypothesis patterns](#hypothesis-patterns)
- [Hypothesis gotchas we hit](#hypothesis-gotchas-we-hit)
- [CrossHair: solver-backed search](#crosshair-solver-backed-search)
- [Atheris: coverage-guided fuzzing](#atheris-coverage-guided-fuzzing)
- [Turning findings into permanent tests](#turning-findings-into-permanent-tests)
- [Running them in CI](#running-them-in-ci)
- [Sources](#sources)

## Terms

| Term | Meaning |
|---|---|
| **Property** | A rule that must hold for every input, such as "parsing the formatted value returns the value". |
| **Property-based testing (PBT)** | Generate many inputs, check the property on each. Hypothesis is the Python standard. |
| **Shrinking** | After a failure, Hypothesis searches for the smallest input that still fails, so the report shows `0.001`, not a 40-character string. |
| **Strategy** | A Hypothesis object that describes how to generate values (`st.integers()`, `st.text()`). |
| **Oracle** | Whatever decides pass or fail for a generated input. In PBT and fuzzing it is the property, not a hard-coded expected value. |
| **Coverage-guided fuzzing** | Mutate input bytes; keep any input that executes new code; repeat. Atheris (built on libFuzzer) does this for Python. |
| **Corpus** | The saved set of inputs a fuzzer has found interesting. Seed it with a few valid examples. |
| **Symbolic execution** | Run the code on symbolic values and ask a constraint solver (an SMT solver, a kind of theorem prover) for concrete inputs that reach each path. CrossHair does this for Python. |
| **Contract** | Pre- and postconditions written on a function (`pre:`, `post:` in a docstring). CrossHair searches for inputs that satisfy the precondition and break the postcondition. |

## Which tool finds which bug

`examples/wallet/demo/` holds v0 versions of the wallet with planted bugs, and
`demo/test_v0_findings.py` aims each technique at them.

| Bug in v0 | Example tests | Hypothesis (random) | Hypothesis + CrossHair | Atheris |
|---|---|---|---|---|
| `parse_amount("")` leaks `decimal.InvalidOperation` instead of `AmountError` | missed | **found** | n/a | **found** (first input) |
| `parse_amount("0.001")` rounds to `0.00` after the positivity check | missed | **found** | n/a | found if the oracle checks positivity |
| `withdraw(nan)` corrupts the balance (NaN fails both comparisons) | missed | **found only sometimes** | n/a | n/a |
| Partner-bank fee goes negative, but only for accounts that pass a mod-97 check | missed | **missed, 3 of 3 runs at 1,000 examples** | **found in ~0.4 s, 3 of 3 runs** (1.7 s on CrossHair 0.0.111) | not tried |

The pattern: random generation finds malformed-input bugs quickly. A bug behind an
exact-value condition (a checksum, a magic constant, a narrow numeric band) is
nearly invisible to random draws and trivial for a solver. Use both.

The solver run, as reported by pytest:

```
amount = 11000, account = 4200000091
...
E       assert -1 >= 0
FAILED demo/test_v0_findings.py::test_fee_never_negative_solver_search
```

The same bug from standalone CrossHair, checking the docstring contract:

```
error: false when calling transfer_fee_cents(11000, 4200000091) (which returns -1)
```

## Hypothesis patterns

`assets/test_properties_template.py` has a runnable version of each.

| Pattern | Property | Catches |
|---|---|---|
| Round trip | `parse(format(x)) == x` | lossy or asymmetric conversion |
| Oracle (differential) | `fast(x) == slow_but_obvious(x)` | optimization bugs |
| Invariant | output always inside its contract | off-by-one, sign errors |
| Idempotence | `f(f(x)) == f(x)` | normalizers that keep changing |
| Metamorphic | a known change to the input has a known effect on the output | bugs with no exact oracle |
| Exception contract + postcondition | raises only the documented error, else the result is valid | leaked internal exceptions |
| Stateful | random operation sequences agree with a simple model | ordering and state bugs |

The exception-contract shape, from `tests/property/test_money_properties.py`:

```python
@given(st.text(max_size=40))
def test_any_text_either_parses_to_a_valid_amount_or_raises_amount_error(text):
    try:
        value = parse_amount(text)
    except AmountError:
        return
    assert Decimal(0) < value <= MAX_AMOUNT
    assert value == value.quantize(CENT)
```

Catch only the documented exception. Catching `Exception` would hide exactly the
leak this test exists to find, and the audit reports it as `SWALLOWED_FAILURE`.

### Check what the generator produces

A property that passes every run may simply never reach the interesting inputs. In a
study of 30 practitioners, 11 admitted they did not check their generators, and the
most common validation, planting a bug to see the property fail, was used by 7
[PBTP §4.6] (`papers/pbt-in-practice.md`). Two cheap checks, both run on Hypothesis
6.170.14:

- Label inputs with `event(...)` and run `pytest --hypothesis-show-statistics`. For
  `st.text(max_size=20)` labelled empty / has whitespace / no whitespace, the report
  was `72.00%, no whitespace`, `27.00%, has whitespace`, `1.00%, empty`: the empty
  string, a classic edge case, came up in 1% of draws. Pin it with `@example("")`.
- Run `mutate_check.py` on a function covered only by properties. A surviving mutant
  in a branch means the generator never sends input down it.

The same study found differential properties (fast implementation against a slow,
obvious one) the most used kind, by 17 of 30 [PBTP §4.3]; reach for one whenever a
simple reference exists.

## Hypothesis gotchas we hit

**Rare values need `@example`.** In 300 draws, `st.floats()` produced NaN 3 times
(about 1%). The first demo run of the NaN property passed 100 of 100 examples and
missed the bug; the next run failed. Known special values belong in `@example`,
which runs on every execution:

```python
@given(st.floats())
@example(math.nan)
@example(math.inf)
def test_withdraw_never_leaves_a_non_finite_balance(amount): ...
```

**`allow_nan=False` can hide the bug you care about.** The original skill's
property example used `st.floats(..., allow_nan=False)` against a Wallet that
corrupts its balance on NaN. Exclude a value only when the function's contract
excludes it.

**`target` is reserved in stateful rules.** `@rule(source=wallets, target=wallets)`
does not pass a `target` argument; `target=` names the bundle the rule's return
value goes into. The run failed with `TypeError: WalletMachine.move() missing 1
required positional argument: 'target'`. Use another name (`dest`).

**Function-scoped fixtures and `@given`.** A test that takes `tmp_path` runs the
fixture once for all generated examples, so state leaks between examples
(Hypothesis raises the `function_scoped_fixture` health check). Use
`tmp_path_factory` or build the resource inside the test, as
`test_wallet_path_never_escapes_the_base_directory` does.

**The `Failing test case:` block printed blank values** in these runs
(`amount=,`). The real values are in the traceback locals just above it.

**Deadlines flake in CI.** The default per-example deadline is 200 ms. The `ci` and
`nightly` profiles in `assets/conftest.py` set `deadline=None`.

| Profile | Use | Settings |
|---|---|---|
| `dev` (default) | local runs | 100 examples |
| `ci` | every push | 300 examples, no deadline, `print_blob=True` |
| `nightly` | scheduled | 5,000 examples, no deadline |
| `crosshair` | solver search | `backend="crosshair"` (registered only if installed) |

Select with `pytest --hypothesis-profile=ci` or `HYPOTHESIS_PROFILE=ci`.

## CrossHair: solver-backed search

Two ways to use it.

**As a Hypothesis backend.** Same test, solver instead of random draws:

```python
@settings(backend="crosshair", max_examples=200, deadline=None, database=None)
@given(st.integers(min_value=1, max_value=10**8), st.integers(min_value=10**9, max_value=10**10 - 1))
def test_fee_is_never_negative_for_any_solver_found_input(amount_cents, account): ...
```

Requires `pip install hypothesis-crosshair`. The import name is
`hypothesis_crosshair_provider`, which is what `pytest.importorskip` must check.
In the example these tests live in `tests/symbolic/`, are auto-marked `symbolic`,
and are deselected by default because each takes seconds.

**Standalone, against docstring contracts:**

```python
def transfer_fee_cents(amount_cents: int, account: int) -> int:
    """
    pre: amount_cents > 0
    post: __return__ >= 0
    raises: AmountError
    """
```

```
crosshair check --analysis_kind=PEP316 --per_condition_timeout=20 src/
```

Declare expected exceptions with `raises:`. Without that line, CrossHair reports the
function's own documented validation as an error, and that report stops the search
before the real bug:

```
error: AmountError: invalid account number when calling transfer_fee_cents(1, 0)
```

On the fixed code the same command exits 0 with no output.

Limits: CrossHair reasons symbolically about built-in types, user classes, and
much of the standard library. It falls back to concrete execution for C
extensions, I/O, and anything it cannot model, where it behaves like a slower
random tester. Point it at pure logic.

## Atheris: coverage-guided fuzzing

Template: `assets/fuzz_target_template.py`. Worked target:
`examples/wallet/fuzz/fuzz_parse_amount.py`.

```
python fuzz/fuzz_parse_amount.py fuzz/corpus/parse_amount -max_total_time=60 -artifact_prefix=fuzz/crashes/
```

Against v0 it failed on the very first input, the empty string (the crash file is
named with the SHA-1 of empty input):

```
 === Uncaught Python exception: ===
InvalidOperation: [<class 'decimal.ConversionSyntax'>]
artifact_prefix='/tmp/v0-'; Test unit written to /tmp/v0-crash-da39a3ee5e6b4b0d3255bfef95601890afd80709
```

Against the fixed parser:

```
#2097152	pulse  cov: 11 ft: 11 corp: 5/43b lim: 4096 exec/s: 419430 rss: 43Mb
Done 16527893 runs in 46 second(s)
```

`cov: 11` means 11 code edges reached. The fixed parser is a strict regex plus two
range checks, so coverage saturated at once and 16.5 million further executions
found nothing. A flat `cov` on a large target would instead mean instrumentation is
missing; check that the target module is imported inside
`atheris.instrument_imports()`.

Harness rules:

| Rule | Why |
|---|---|
| Assert the postcondition, not just "no crash" | Otherwise wrong answers pass. |
| Decode bytes with `utf-8, errors="replace"` | pytest can replay a crash file without Atheris. |
| Import the target inside `instrument_imports()` | Without it Atheris has no coverage signal. |
| No I/O or sleeps in the target | Throughput is the whole method. |
| Seed the corpus with a few valid and invalid inputs | The fuzzer starts from real structure. |

Atheris can also drive an existing Hypothesis test:
`atheris.Setup(sys.argv, test_fn.hypothesis.fuzz_one_input)`.

HypoFuzz is a third option that adds coverage guidance to Hypothesis tests across a
whole suite. Commercial-bias note: version 25.11.1 ships under a custom license
that is free only for non-commercial use, and it names CI for commercial products
as requiring a paid license. Atheris (Apache-2.0) and CrossHair (MIT) have no such
restriction.

## Turning findings into permanent tests

A finding that is not a test will come back.

1. Hypothesis saves failing examples in `.hypothesis/` and replays them locally.
   For CI, copy the shrunk input into an `@example(...)` on the test.
2. Atheris writes `crash-<sha1>` files. Commit them to `fuzz/crashes/`;
   `tests/security/test_fuzz_corpus_replay.py` replays every corpus and crash file
   through the harness oracle on every commit, with no Atheris needed.
3. A CrossHair counterexample becomes an ordinary parametrized unit test.

## Running them in CI

| When | What | Where |
|---|---|---|
| Every push | Hypothesis `ci` profile, corpus replay | `assets/github/workflows/tests.yml` |
| Nightly | Hypothesis `nightly`, `-m symbolic`, `crosshair check`, 5 min Atheris per target | `assets/github/workflows/nightly.yml` |

## Sources

- Hypothesis documentation, settings and stateful testing: https://hypothesis.readthedocs.io/
- hypothesis-crosshair: https://github.com/pschanely/hypothesis-crosshair
- CrossHair, contract syntaxes and limits: https://crosshair.readthedocs.io/
- Atheris (Google), README and Hypothesis integration: https://github.com/google/atheris
- HypoFuzz: https://hypofuzz.com/docs/ (license read from the installed 25.11.1 package)
