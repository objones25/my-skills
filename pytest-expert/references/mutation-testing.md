# Mutation testing

Coverage says a line ran. Mutation testing asks whether any test would notice if
that line were wrong. It makes one small change to the code (a **mutant**), runs
the tests, and records whether some test failed (**killed**) or everything still
passed (**survived**). A survivor is a change no test detects: usually a missing
test, sometimes an equivalent mutant that no input can tell apart from the original
(see [Reading survivors](#reading-survivors)).

## Contents

- [Why it beats coverage as a quality signal](#why-it-beats-coverage-as-a-quality-signal)
- [Two tools, two jobs](#two-tools-two-jobs)
- [mutate_check.py: prove one function's tests can fail](#mutate_checkpy-prove-one-functions-tests-can-fail)
- [mutmut: the whole package](#mutmut-the-whole-package)
- [Reading survivors](#reading-survivors)
- [In CI](#in-ci)
- [Sources](#sources)

## Why it beats coverage as a quality signal

- Inozemtseva and Holmes (ICSE 2014) found only a low to moderate correlation
  between coverage and test-suite effectiveness once suite size is controlled for,
  and concluded coverage should not be used as a quality target. Chen et al. (ASE
  2020, with Holmes as a co-author) dispute the size control itself: test-set size is
  a consequence of the test goal, not a variable to fix. At equal size, tests chosen
  for coverage were as strong as or stronger than tests chosen for mutants until
  coverage was satisfied; mutants added the most when used after coverage had
  "exhausted its usefulness" [§6.4, Fig. 5; §8]. Shrinking a fault-detecting test
  set by mutants mostly kept its fault detection, while shrinking it by coverage
  lost much of it [Fig. 7] (`papers/coverage-mutation-real-faults.md`).
- Just et al. (FSE 2014), across 357 real faults in 5 programs, found a
  statistically significant correlation between mutant detection and real-fault
  detection, independent of coverage. 73% of the real faults were coupled to
  common mutants.

Coverage remains useful as a floor: an uncovered line's mutants survive trivially.

One limit: mutation scoring needs a green suite on code believed correct. It cannot
judge a test whose purpose is to expose a bug that is already in the code, which is
why `mutate_check.py` refuses a red baseline (`papers/llm-tests-proxy-metrics.md`,
[Zhao26, footnote 1 on RQ2 in §3.1]).

Survivors have three acceptable outcomes, not two: killed by a new test, equivalent
(no input can tell the mutant apart), or unproductive (killable, but the test would
pin an arbitrary detail such as a buffer size and become a change detector), each
with a one-line reason. Google's mutation work treats most raw mutants as
unproductive and suppresses them (`papers/google-mutation-at-scale.md`).

## Two tools, two jobs

| Tool | Scope | Speed | Use when |
|---|---|---|---|
| `scripts/mutate_check.py` | one file, optionally one function | seconds to a minute | workflow step 5: prove the tests you just wrote can fail |
| mutmut 3 | whole package | seconds on the example (under 10 s), and it grows with mutants times suite time | nightly, or before trusting a suite |

`mutate_check.py` is stdlib only. Operators: comparison boundaries (`<` and `<=`,
`==` and `!=`), arithmetic swaps, integer constant +1, `True`/`False`, `and`/`or`,
dropping `not`, `max`/`min` and `any`/`all` swaps, and `return x` to `return None`.

## mutate_check.py: prove one function's tests can fail

```
python scripts/mutate_check.py src/wallet/money.py --function transfer_fee_cents -- python -m pytest -x -q -p no:randomly --no-cov tests/unit
```

Everything after `--` is your test command. The file is edited in place and
restored on exit, Ctrl-C, or SIGTERM. Each run gets a fresh `PYTHONPYCACHEPREFIX`:
a same-size mutation written within the same second can otherwise be masked by a
stale `.pyc`, making a mutant look like it survived.

First run on the example's unit tests:

```
28/34 mutants killed (82%), baseline 4.5s
SURVIVED line 47: 0 -> 1  <- no test fails when this changes
SURVIVED line 49: <= -> <  <- no test fails when this changes
SURVIVED line 49: < -> <=  <- no test fails when this changes
SURVIVED line 49: ** -> *  <- no test fails when this changes
SURVIVED line 49: 10 -> 11  <- no test fails when this changes
SURVIVED line 49: 10 -> 11  <- no test fails when this changes
```

Line 47 is `if amount_cents <= 0`; line 49 is `10**9 <= account < 10**10`. No test
used an amount of 1 cent or an account at either end of the 10-digit range. After
adding those boundary tests:

```
32/34 mutants killed (94%), baseline 3.5s
SURVIVED line 49: <= -> <  <- no test fails when this changes
SURVIVED line 49: < -> <=  <- no test fails when this changes
```

These two are **equivalent mutants**: no input can tell them apart from the
original. They only differ at exactly 10⁹ and 10¹⁰, and both of those fail the
mod-97 check (remainders 34 and 49), so the function rejects them either way.
Equivalent mutants are expected; record why and move on.

The baseline must pass first. When boundary constants were mistyped, the script
refused to run:

```
mutate_check: baseline test command did not pass (failed); fix the suite first
```

## mutmut: the whole package

Config, in `pyproject.toml` (mutmut 3.8 deprecated `tests_dir`):

```toml
[tool.mutmut]
source_paths = ["src/wallet"]
also_copy = ["fuzz"]
pytest_add_cli_args = ["-p", "no:randomly", "--no-cov", "-q"]
pytest_add_cli_args_test_selection = ["tests/unit", "tests/property", "tests/security"]
```

mutmut copies the project into `mutants/` and runs there, so anything tests read at
run time (the fuzz corpus here) must be listed in `also_copy`. The first attempt
failed with `test_corpus_is_not_empty` for exactly that reason. Add `mutants/` to
`.gitignore` and to ruff's `extend-exclude`.

| Command | Does |
|---|---|
| `mutmut run --max-children 4` | run every mutant, 4 workers |
| `mutmut results` | list survivors |
| `mutmut show <name>` | diff of one mutant |

Result on the example, first run (72 s, 3.25 mutants/s; machine not recorded):

```
194/194  killed 175  survived 19
```

## Reading survivors

`mutmut show` on a few of the 19:

```
-    if amount > self.balance:
+    if amount >= self.balance:
```
No test withdraws the exact full balance. Real gap; added
`test_withdraw_entire_balance_leaves_exactly_zero`.

```
-        return max(0, 10 - amount_cents // 1000)
+        return max(0, 10 - amount_cents / 1000)
```
The test asserted `fee == 5`, and `5.0 == 5` is true in Python. Asserting
`(fee, type(fee)) == (expected, int)` kills it.

```
-    if len(text) > MAX_TEXT_LEN:
+    if len(text) >= MAX_TEXT_LEN:
```
No test at exactly 32 characters. Added accept-32 and reject-33 tests.

```
-        data = json.loads(raw.decode("utf-8"))
+        data = json.loads(raw.decode("UTF-8"))
```
Equivalent. Ignore.

After the four new tests, on 2026-11-09 on macOS arm64 with Python 3.13.2, mutmut
3.8.1 and pytest 9.2.0 (two runs, 6 s and 9 s; an earlier record of 183 killed and 11
survived in 61 s, machine not recorded, did not reproduce):

```
194/194  killed 181  survived 13
```

What the remaining 13 teach:

| Survivor | Lesson |
|---|---|
| Three `encoding="utf-8"` mutants in `save_wallet` (`None`, omitted, `"UTF-8"`), and `decode("UTF-8")` in `load_wallet` | Equivalent. `json.dumps` escapes non-ASCII by default (`ensure_ascii=True`), so the file is pure ASCII under any encoding, and codec names are case-insensitive. |
| `AmountError("amount must be positive")` mutated to `"XXamount must be positiveXX"`, the same for `"invalid account number"`, and `StoreError(...)` mutated to `StoreError(None)` in `load_wallet` | `pytest.raises(match=...)` uses `re.search`. Anchor it (`match=r"^amount must be positive$"`) when the message is the contract. |
| `balance < 0` mutated to `balance <= 0` and to `balance < 1` in `load_wallet` | No test loads a stored wallet with a zero or negative balance. A security-relevant gap: corrupt files are not covered. |
| Each `or` mutated to `and` in the same check | Same gap: no test for a file whose owner does not match its name, or whose balance is not finite. |
| `10**9 <= account` mutated to `<`, and `account < 10**10` to `<=` | Equivalent (see above). |

These are left in the example on purpose as exercises.

Triage order: a survivor in validation, auth, or money logic is a missing test.
A survivor in a log or error message is a missing test only if callers depend on
the text. An equivalent mutant gets a one-line note.

## In CI

A whole-package run costs roughly mutants times suite time: under 10 s on the example,
but long enough on a real package that it belongs in a nightly job, while
`mutate_check.py` on the functions a change touched is cheap enough to run before
pushing. Google's approach is per change: mutants only on the lines a changelist adds
or changes (`papers/google-mutation-at-scale.md`). `assets/github/workflows/nightly.yml` runs
mutmut nightly and fails if survivors exceed `MUTATION_MAX_SURVIVORS`. Set the
limit at today's count and lower it as you add tests, the same ratchet as the
coverage floor.

## Sources

- Inozemtseva and Holmes, *Coverage Is Not Strongly Correlated with Test Suite Effectiveness*, ICSE 2014: https://cs.ubc.ca/~rtholmes/papers/icse_2014_inozemtseva.pdf
- Just et al., *Are Mutants a Valid Substitute for Real Faults in Software Testing?*, FSE 2014: https://www.cs.ubc.ca/~rtholmes/papers/fse_2014_just.pdf (the umass.edu copy returned 404 on 2026-11-02)
- mutmut: https://github.com/boxed/mutmut (config keys read from the installed 3.8.1 package)
