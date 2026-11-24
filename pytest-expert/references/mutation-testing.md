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
| `scripts/mutate_check.py` | one file, optionally one function | 74 s for the example's 34 mutants (2026-11-20, `examples/wallet/README.md`), and it grows with mutants times suite time | workflow step 5: prove the tests you just wrote can fail |
| mutmut 3 | whole package | 12 s on the example (2026-11-20), and it grows with mutants times suite time | nightly, or before trusting a suite |

`mutate_check.py` is stdlib only. Mutation kinds (the names work with `--skip-kinds` and
`--only-kinds`):

| Kind | Mutation |
|---|---|
| `cmp` | comparison boundaries and negations: `<` and `<=`, `==` and `!=`, `is`, `in` |
| `binop`, `augassign` | arithmetic swaps (`+ - * // / % **`), plain and augmented |
| `boolop` | `and` and `or` |
| `not` | `not x` becomes `x` |
| `bool` | `True` and `False` |
| `int`, `float` | `n` becomes `n + 1`, `c` becomes `c + 1.0` |
| `name` | `max` and `min`, `any` and `all` |
| `return` | `return x` becomes `return None` |
| `matmul` | `@` becomes `*` |
| `shift` | `<<` and `>>` |
| `bitop` | `\|` becomes `&`, `&` becomes `\|`, `^` becomes `&` |
| `neg` | unary minus dropped |

Only code inside a function or class body is mutated; annotations, docstrings and module-level
code are not. `x | None` is never mutated (it can only be a type union), but other runtime `|`
(`isinstance(x, int | str)`, set and dict union) is, and a type-union mutant raises `TypeError`,
which is a weak kill. Drop the `bitop` kind if that noise outweighs the signal. A tool that
mutates more kinds finds more mutants, so a score is comparable only with scores from the same
version of the script and the same kinds.

## mutate_check.py: prove one function's tests can fail

```
python scripts/mutate_check.py src/wallet/money.py --function transfer_fee_cents -- python -m pytest -x -q -p no:randomly --no-cov tests/unit
```

Everything after `--` is your test command. The file is edited in place and
restored on exit, Ctrl-C, or SIGTERM (exit 130). Test runs get `PYTHONDONTWRITEBYTECODE=1`,
so no mutant is ever cached as a `.pyc` that could mask a later same-size mutant and make
it look like it survived. The existing cache for pytest and the stdlib is still read,
so each run does not recompile them.

The test command runs in its own process group, and the whole group is killed on a timeout,
on SIGINT or SIGTERM, and when it exits. The baseline run happens twice and must pass both
times, so the baseline cost is doubled; a flaky suite cannot be scored. The mutant timeout is
`max(--timeout-floor, baseline * --timeout-factor)`, with a floor of 2 s: a
`--timeout-factor` that gives less than 2 s changes nothing unless you also lower
`--timeout-floor`.

Each mutant gets one status, from the test command's exit code (pytest's):

| Status | Meaning | In the score |
|---|---|---|
| `SURVIVED` | exit 0: the tests did not notice | counts against it |
| `killed` | exit 1, or any code listed in `--kill-codes` (default `1`, "tests failed") | counts for it |
| `killed (timeout)` | the run exceeded the timeout | counts for it, and is reported on its own line: a loaded machine also causes it |
| `errored (exit N)` | any other exit: pytest 2 (the mutant broke the import or collection), a crash by signal (negative N) | left out: the tests could not run, so they did not catch anything |

The score is killed over killed plus survived. The text report prints each mutant as it
finishes, with the test ids that killed it (`by test_a, test_b (+2 more)`: the first three).
JSON (`--format json`) has `score`, `killed`, `survived`, `errored`, `timeouts`,
`baseline_seconds` and `timeout_seconds`, and per mutant `index`, `line`, `description`,
`status`, `seconds`, `exit_code` and `killed_by`. The ids come from pytest's short summary and
unittest's `FAIL:` lines in the captured output, so a test command that suppresses the summary
(`-p no:terminal`) reports an empty `killed_by`; the output never changes a status.
`--show-output` prints the last 10 lines of each errored mutant's output (text) or adds an
`output_tail` field to every mutant (JSON). `--function NAME` scopes by a bare name (any
function or class of that name at any depth, with everything nested in it) or `Class.method`
(a path from the top of the module).

Exit status: 0 when no mutant survived (or survivors are at most `--max-survivors`), 1
otherwise, 130 after SIGINT or SIGTERM. 2 means the check could not be done, and is never a
pass: a usage error, an unreadable or unparsable source, a test command that cannot start, a
red or flaky baseline, no mutation sites, or no mutant that could be evaluated (every run
exited outside `--kill-codes`).

First run on the example's unit tests. These two reports are earlier captures: the current
script also prints one `  #N  line L  description  status` line per mutant as it goes and ends
its summary line with `mutant timeout Ns`; the survivor lines and counts are as quoted (the
second report's survivors were confirmed again on 2026-11-20, 32 of 34 killed, in
`examples/wallet/README.md`).

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
refused to run and exited 2 (re-run 2026-11-20 with a command that exits 1):

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

Result on the example, first run (72 s, 3.25 mutants/s; machine not recorded; 194 mutants then, 196 now):

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

After tests for those four, and for the fee messages and `load_wallet`'s consistency checks
(see `examples/wallet/README.md`), on 2026-11-20 on macOS arm64 with Python 3.13.2, mutmut
3.8.1 and pytest 9.2.0 (one run, 12 s, in a scratch copy of the example):

```
196/196  killed 190  survived 6
```

What the remaining 6 teach:

| Survivor | Lesson |
|---|---|
| Three `encoding="utf-8"` mutants in `save_wallet` (`None`, omitted, `"UTF-8"`), and `decode("UTF-8")` in `load_wallet` | Equivalent. `json.dumps` escapes non-ASCII by default (`ensure_ascii=True`), so the file is pure ASCII under any encoding, and codec names are case-insensitive. |
| `10**9 <= account` mutated to `<`, and `account < 10**10` to `<=`, in `transfer_fee_cents` | Equivalent: `10**9 % 97` is 34 and `10**10 % 97` is 49, so neither boundary value passes the mod-97 check either way. |

A message mutant such as `AmountError("amount must be positive")` becoming
`"XXamount must be positiveXX"` survives while `pytest.raises(match=...)` is unanchored,
because `match` uses `re.search`. Anchor it (`match=r"^amount must be positive$"`) when the
message is the contract; that is how the example kills them.

Triage order: a survivor in validation, auth, or money logic is a missing test.
A survivor in a log or error message is a missing test only if callers depend on
the text. An equivalent mutant gets a one-line note.

## In CI

A whole-package run costs roughly mutants times suite time: 12 s on the example (2026-11-20),
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
