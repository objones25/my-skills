# Changelog

## 2026-11-27: keep expensive work out of tests (rule 35, a 2 s duration budget)

The problem: tests that run an expensive function (training, a simulation, a large solve) only to get input for the thing they check, so a full run took 30+ minutes. The skill had a `slow` marker and `--durations`, but nothing told an agent not to do this, and nothing failed when it did.

| Change | Where |
|---|---|
| Rule 35, "A test runs only the computation it is testing", in the hard gates and in `rules.md` (34 to 35 gates) | `SKILL.md`, `references/rules.md`, `INSTALL.md` |
| Workflow step 3 now says never to run an expensive function in a test that is not about it | `SKILL.md` |
| A duration budget: a passing setup, call or teardown phase over 2 s fails unless the test is marked `slow`, `integration`, `e2e`, `gpu` or `symbolic`; the message names the test and the phase. `PYTEST_EXPERT_BUDGET_S` changes it, `0` turns it off, empty means the default, and anything not a finite number >= 0 stops the run with a usage error. xfail tests are exempt, and so is the call phase of Hypothesis tests (detected with `hypothesis.is_hypothesis_test`), because its total scales with `max_examples`; the dev profile's per-example `deadline` checks those locally, and nothing does under the `ci` profile (documented) | `assets/conftest.py`, copied to `examples/wallet/tests/conftest.py` |
| New reference: techniques cheapest first, scope as a cache (and its xdist and parametrize pitfalls), Hypothesis cost, tiny models, finding slow tests, the budget, the dev loop, what not to do, with sources | `references/test-speed.md` |
| The budget is off for nightly jobs and mutmut, where instrumentation slows every test | `assets/github/workflows/nightly.yml`, `references/mutation-testing.md` |
| Speed table updated | `references/ci-cd.md` |
| One eval and one trigger case for a slow suite | `evals/` |

Design decisions, from three research passes (pytest, pytest-xdist, Hypothesis, pytest-timeout, pytest-fail-slow and pytest-testmon docs; Meszaros *Slow Tests*; Software Engineering at Google ch. 11; the Hugging Face transformers testing guide; sources in `references/test-speed.md`):

- **A runtime budget, not a static check.** A static check for calls named `train`, `fit` or `solve` was rejected: `fit` is standard scikit-learn API and cheap on small data, and a name says nothing about input size. The budget measures what actually ran.
- **2 s, not 1 s.** The first version used 1 s and failed the wallet's stateful Hypothesis test (1.05 s on this machine; the reviewer measured 0.54 s on another run), with a security property at 0.85 s close behind. Both are Hypothesis tests, which led to the exemption. 2 s leaves room for a loaded CI runner, so a 0.9 s test does not become flaky.
- **Setup and teardown are checked too.** An expensive fixture is the usual way a cheap test ends up running expensive code. A module or session fixture is charged to whichever test sets it up or tears it down, which varies under pytest-randomly and xdist (documented).
- **Inline hook, not pytest-fail-slow.** The plugin's last release was 2024-06-01, and it uses its own `fail_slow` marker instead of `slow`. The hook is about 20 lines and adds no dependency.

Independent review (a fresh agent given the diff and the sources): every quote and version was confirmed against its source. It found these bugs, now fixed test-first: a non-numeric, `nan`, `inf` or negative budget crashed collection or failed every test; teardown was not checked; a slow non-strict xfail that passed was rewritten to FAILED in the terminal but stayed "skipped" in JUnit XML. It also found claims, now corrected: the Hypothesis exemption is unchecked under `deadline=None` profiles; the shared-fixture charge is arbitrary; coverage and xdist inflate the times; "no false positives" was overstated; the mutmut slowdown was asserted without a measurement; the `timeout` example contradicted `pytest-config.toml`. Added at its suggestion: a path for adopting the budget in an existing slow suite. Not done: `diff_guard.py` does not flag an added `slow` marker or a raised budget; `test-speed.md` says so.

Tests: `selftest/test_conftest_budget.py`, 20 tests that run the asset conftest in a subprocess. The two core tests (an unmarked slow test fails, a slow fixture is charged to setup), the Hypothesis call-phase test and the 7 tests for the review's bugs were seen failing before their code existed. `mutate_check.py` after the review fixes: `pytest_runtest_makereport` 13 of 14 killed, `_parse_budget_s` 9 of 9. The survivor is `<=` to `<` on the duration comparison, which differs only when a duration equals the budget exactly and cannot be tested deterministically (equivalent in practice). Wallet example under the budget: 135 passed, 1 skipped, 2 deselected with the `dev` profile and again with `HYPOTHESIS_PROFILE=ci` (unchanged counts). `selftest`: 1485 passed, on 2026-11-27, macOS arm64, Python 3.13.2, pytest 9.2.0, run with the skill's `test-command`. ruff 0.17.3 is clean on the changed Python files, both with `assets/ruff-pytest.toml` and with `--isolated`.

Not done: evals were not run with or without the skill. The budget was not tried under pytest-xdist in this repo; a research prototype of the same hook gave the same outcomes serially and with `-n 2`, on pytest 9.2.0 with xdist 3.8.0. pytest-testmon's compatibility with xdist and coverage was not checked.

## 2026-11-24: mutate_check.py about 4x faster

Profiled with cProfile on audit-skill's `quote_match.py` (57 mutants): 98% of the time was spent waiting on test runs, and under 1 s in the script itself.

| Cause | Evidence | Fix |
|---|---|---|
| Each test run got a new, empty `PYTHONPYCACHEPREFIX` while `PYTHONDONTWRITEBYTECODE=1` was set, so every run recompiled pytest and the stdlib from source | One pytest run took 0.81-0.90 s that way, 0.28-0.32 s without the prefix | Dropped the prefix; kept `PYTHONDONTWRITEBYTECODE=1`, which already stops a mutant from being cached |
| Timeout floor of 10 s | Two infinite-loop mutants took 10.0 s each against a 0.3 s baseline | `--timeout-floor` now defaults to 2 s (`MUTANT_TIMEOUT_FLOOR_S = 2.0`) |

Benchmark: 62.0 s to 15.4 s of test-run time, with an identical score (66.7%) and an identical verdict for every mutant. `references/mutation-testing.md` now describes the new behaviour. Two new tests in `selftest/test_mutate_check.py`, both seen failing before the fix. `selftest`: 259 passed, 3 skipped (257 before, plus the 2 new tests), macOS arm64, Python 3.13.2, pytest 9.2.0; measured before merging the 2026-11-20 entry below. After the merge the fix was re-applied to that entry's rewritten `run()` (Popen in its own process group) and to the default of its new `--timeout-floor` option, and `test_timeout_floor_is_a_minimum_and_is_reported` now expects the 2 s default instead of 10 s. `selftest` after the merge: 1431 passed (1430 plus that test after its update; `test_mutate_check.py` alone 181 passed), macOS arm64, Python 3.13.12, run with the skill's `test-command` and `-p no:randomly`.

## 2026-11-21: final mutation run, and tests for the gaps it showed

No script changed. The sequential mutation run on the 2026-11-20 tree finished (23 jobs, one at a time, no errored mutants, no timeouts); the full table and the survivor notes are in `AUDIT.md` section 7.

| Target | Killed | Score |
|---|---|---|
| `scripts/audit_tests.py` | 631 of 698 | 90.4% |
| `scripts/diff_guard.py`, 16 selected functions (no whole-file run) | 229 of 274 | 83.6% |
| `scripts/mutate_check.py` | 185 of 204 | 90.7% |
| `scripts/flake_hunt.py` | 85 of 89 | 95.5% |
| Stop gate hook | 97 of 111 | 87.4% |
| `_common.py` (hooks) | 71 of 80 | 88.8% |
| Post-edit hook | 30 of 41 | 73.2% |
| Session-start hook | 3 of 5 | 60.0% |

### Tests added or strengthened (28 new tests, 2 strengthened)

- `selftest/test_diff_guard.py` (462 to 484): which assertions count as able to run (`else` branches, conditional early returns, exits two blocks deep, `while False`, `break` and `continue`, `try` / `except*`); exit code 2 with a message for `FileNotFoundError`, other `OSError` and an internal error; deleting the only file the guard reads; a first test file in a repository that had none; a native `[tool.pytest]` table; a `pytest.ini` with no section; a special case already in the base commit.
- `selftest/test_hooks.py` (111 to 115): the post-edit hook's "fails right now" and "pytest could not run" tests matched text anywhere in the output, so they passed when the hook crashed on the line that builds the message (the crash report quotes that line). They now assert the exact start of the message, pytest's own output, and that nothing else is emitted. New tests: a `file_path` that is not a usable string, and the hook reporting its own crash.
- `selftest/test_audit_tests.py` (450 to 452): only an integration marker on a class excuses a network call; asserting an ordinary call's result is not a weak assertion.

Seeing them fail: a rerun of the five affected `diff_guard.py` functions took `scan_block` from 9 of 26 to 23 of 26, `main` 17 to 19 of 23, `guard` 9 to 10 of 12, `pytest_config` 7 to 9 of 10 and `source_findings` 15 to 18 of 20 (the 16 functions together: 251 of 274, 91.6%). The post-edit hook mutants (5), the `audit_tests.py` mutants (2) and the internal-error mutant of `main` were applied by hand and each failed the new tests. These tests were written after the code they cover, because the code was already right.

### Not done

- No whole-file mutation run of `diff_guard.py`; no whole-job rerun of `audit_tests.py` or the post-edit hook after these tests.
- The remaining survivors were read and grouped (`AUDIT.md` section 7), not each proven equivalent; a few lines in `audit_tests.py` and `diff_guard.py` are listed there as not examined.
- `references/hooks.md` now says 115 hook tests.

### Verification

| Date | Machine | Versions | Check | Result |
|---|---|---|---|---|
| 2026-11-21 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0, Hypothesis 6.170.17 (pytest-randomly 5.1.0 switched off) | `pytest selftest -q -p no:cacheprovider -p no:randomly` | 1457 passed in 185 s (audit 452, diff guard 484, flake hunt 54, hooks 115, mutate_check 179, structure 173); 1429 on 2026-11-20 |
| 2026-11-21 | same | same | `audit_tests.py selftest` | clean: no findings (7 files checked) |

## 2026-11-20: audit of the scripts, hooks, self-tests and wallet tests, and the fixes

An `audit-skill` run (risk scope, stage 1 of 4) checked 311 of 3,323 inventoried items: the four scripts, the Claude Code hooks, their self-tests, the structure tests and the wallet example's tests. It found 48 wrong, 1 stale and 2 disputed items, plus 6 claims that the last review round reported missing and nobody checked, and 172 suggestions. This entry records every change made in response: 269 rows in ten fix logs, A to H (258 fixed, 7 not fixed, 4 already correct). Logs A to F (246 rows) answer the audit; G1, G2, G3 and H (23 rows) are owner-approved follow-ups that resolve 3 of the 7 not-fixed rows and 2 of the 5 partly fixed rows, so 4 rows remain not fixed and 3 remain partly fixed. The method, every finding and what is still unchecked are in `AUDIT.md`. The reference files, paper digests, `SKILL.md`, `INSTALL.md`, the asset templates and the workflows were not part of this audit; later stages cover them.

Fixes were written test first; each fixed row in the logs records what the new test showed against the old code, except one round of `mutate_check.py` additions where the code came first (see `AUDIT.md`). The self-test suite grew from 260 tests (2026-11-10) to the count in the verification block below.

### What changes for people who run the scripts

| Change | Where |
|---|---|
| Exit 2 now means "could not check" in all four scripts, and is never a pass. Before, these cases exited 0 or 1: a path with no test files, an unknown `--ignore` code, an empty repository, a nonexistent test path, `--runs 0`, a test command that does not exist | all four scripts |
| `audit_tests.py` prints `clean: no findings (N file(s) checked)` instead of `clean: no findings` | `audit_tests.py` |
| `mutate_check.py` scores are not comparable with earlier runs (see the `mutate_check.py` section) | `mutate_check.py` |
| `flake_hunt.py` runs 31 seeds by default, not 20 | `flake_hunt.py` |
| The Stop gate runs pytest on the whole project and blocks when a check could not run, where it used to skip silently | hooks |
| New diff guard codes `ASSERT_WEAKENED` (error), `PARAM_HOOK_CHANGED` and `SKIP_VIA_HELPER` (warnings); new audit code `SUPPRESSION_INERT` (warning) and option `--assert-helper` | `diff_guard.py`, `audit_tests.py` |

### scripts/audit_tests.py

26 codes: the 25 of the original audit (the docstring says so and a self-test pins the count) plus `SUPPRESSION_INERT` (log G1, below). Exit status: 0 clean, 1 an error-severity finding remains (any warning with `--warnings-as-errors`), 2 could not check. Findings are sorted by file, line, code and detail, so same-line findings come out in the same order whatever `PYTHONHASHSEED` is.

| Problem | Fix |
|---|---|
| The script's self-description was read as 26 rule codes; the code defines 25 (`SKILL.md` says 25) and the docstring stated no count | The docstring says "It reports 25 rule codes"; a self-test pins the count to the code and to the docstring |
| `FAKE_MOCK_ASSERT` said a default `Mock` auto-creates `called_once`, `not_called` and similar names | Rewritten: a default Mock raises `AttributeError` for these on Python 3.12 and later (and for misspelt `assert_*` names since 3.5). The rule still catches `m.assert_called_once` written without parentheses, `assert not m.not_called`, and misspellings found by similarity to the real names (cutoff 0.88). It stays silent on `Mock(unsafe=True)` and on non-Mock objects |
| `ALWAYS_TRUE` flagged `assert [*r]` and `assert {**r}`, which can be empty and can fail; it missed `assert -1`, f-strings, lambdas, `not False`, `1 == 1`, `x or True`, `len(x) >= 0` | One evaluator decides constant truth; the starred forms are no longer flagged, the missing forms are |
| `NETWORK` fired on any call rooted at a network library (`urllib.parse`, `http.HTTPStatus`, exception classes, `httpx.MockTransport`), and ignored class-level `integration` / `e2e` marks. Its message said "real network call" | It fires only on calls that open a connection (`requests.get`, `httpx.Client`, `urllib.request.urlopen`, `socket.socket`, `socket.create_connection`, `aiohttp.ClientSession`, `http.client.HTTPConnection` and similar) after import aliases are resolved. Class decorators and class-body `pytestmark` exempt their methods. Message reworded |
| The integration/e2e directory exemption depended on how the path was typed and on directories above the project (a checkout under a folder named `integration` was exempt) | Directory names are taken from the resolved path relative to the project root, so `.`, a relative path and an absolute path agree |
| `UNSEEDED_RANDOM` flagged `default_rng(seed=42)` and `RandomState(seed=0)`: only positional arguments were checked | Keyword seeds count; a test, module or autouse fixture that seeds itself is exempt; aliased and from-imported `random` / `numpy.random` are resolved |
| Seeding was looked up from sibling directories, a commented-out call, a never-called seed function, a non-autouse fixture, and with a relative path hid a real parent `conftest.py` | The nearest `conftest.py` files from the test file's directory up to the project root are parsed with `ast`; none of those cases counts as seeding |
| `ENV_MUTATION` missed augmented, tuple, starred and annotated targets, `os.environ \|= ...`, `popitem` and aliased `environ` | Those forms are covered; writes inside `with mock.patch.dict(os.environ, ...)` (or the decorator) are exempt, because the patcher restores them |
| The pytest config lookup did not follow pytest 9.2: `.pytest.ini` and `.pytest.toml` were not read, and a `pytest.ini` that beats `pyproject.toml` was not honoured | The lookup mirrors pytest 9.2.0's `findpaths.py` order per directory (`pytest.toml`, `.pytest.toml`, `pytest.ini`, `.pytest.ini`, `pyproject.toml`, `tox.ini`, `setup.cfg`) |
| `XFAIL_NOT_STRICT`: `strict` was ORed with `strict_xfail`, so `strict = true` with `xfail_strict = false` was read as strict; `xfail(strict=False)` was not reported | `strict_xfail` / `xfail_strict` are read first and `strict` is only the fallback, as in pytest's `skipping.py`; `strict=False` is reported, `strict=True` and a non-constant value are not. The description says so |
| `MUTABLE_MODULE_STATE` missed annotated state, `.sort()`, `.reverse()`, `.popitem()`, nested subscripts and `global X; X += [...]`, and flagged import-time table fills | Those mutations are covered; only mutations inside functions count, and a function that binds the name itself is skipped |
| A UTF-8 file with a byte-order mark was reported as `SYNTAX`; a file with a null byte gave a traceback | Source is decoded with `tokenize.detect_encoding` before parsing; null bytes are a `SYNTAX` finding |
| The directory skip list matched the ancestors of the audited path, so a project under `.../build/...` audited nothing | The skip list applies only below the audited, resolved root |
| `selftest/test_audit_tests.py`: the def-line pragma test passed on the plain same-line rule and never exercised the whole-function range | A `FLOAT_EQ` finding on a body line, plus decorator-line, class-line and "other line is not covered" cases |

Other changes from the suggestions:

- New detections: `NO_ASSERT` for a bare `pytest.raises(...)` / `warns` / `RaisesExc` / `RaisesGroup` statement (they count only as a with-item or with a callable argument), for asserts after an unconditional `return` / `raise` or in a nested function nothing calls, and for a near-miss of a Mock assertion name. `WEAK_ASSERT` also recognises `x != None`, `None is not x`, `not x is None`, `0 < len(x)`, `bool(x)` and an `and` of weak checks, and no longer reports `len(x) > 1`. `BARE_COMPARISON` also fires on `not (a == b)` and on an `and` / `or` of comparisons. `FLOAT_EQ` also fires on sums (`0.1 + 0.2`), containers of floats, `float('0.1')`, `1 / 3` and `x in (0.1, 0.2)`; `float('inf')` stays exempt. NaN detection resolves aliases (`from math import nan`, `numpy as npy`, `jax.numpy`) and accepts `float('-nan')`, `np.float32('nan')` and `Decimal('NaN')`. `SWALLOWED_FAILURE` covers `contextlib.suppress(AssertionError / Exception / BaseException)`. `BROAD_RAISES` resolves aliases, reads `expected_exception=`, `builtins.Exception` and `RaisesExc`, and treats `match=''`, `'.*'`, `'^.*$'` and `None` as no match. `SLEEP` resolves `import time as t` and `from time import sleep as nap`, and no longer flags an unimported bare `sleep` (an injected fake). `SYS_PATH` covers `sys.path += [...]`, `sys.path[0:0] = ...`, `sys.path = ...`, `remove`, `pop`, `clear` and `from sys import path`. `SKIP_NO_REASON` covers class decorators, module and class-body `pytestmark`, `reason=''` and `skip('')`. `PATCH_NO_SPEC` treats `autospec=False` and `spec=None` as no double. `UNITTEST_CLASS` resolves `IsolatedAsyncioTestCase`, `import unittest as ut` and subclasses of a flagged class in the same module. `SHADOWED_TEST_NAME` treats one definition per `if` / `else` branch as fine and reports a repeated `class TestA`. `BRANCHING` also flags `match` and `async for` over a name or literal.
- Removed false positive: `YIELD_FIXTURE` message no longer says "deprecated and deprecated since 6.2"; it keeps "pytest 9.2 schedules removal for pytest 10 or 11".
- Arguments: `--ignore` is repeatable and merged, case-insensitive, and an unknown code exits 2 naming the valid codes; `--ignore SYNTAX` exits 2 (an unparsable file cannot be made clean).
- Could not check, exit 2 with one line on stderr: no test files (including "all in skipped directories"), an unreadable file or directory, a broken symlink, a path that is neither file nor directory, an unparsable `pytest.ini`, and any unexpected exception in `main()` (it used to be a traceback and exit 1, the same as "findings"). With no path argument the message is `audit_tests: a test directory or file is required (usage: audit_tests.py PATH)`.
- On Python 3.10 (no `tomllib`) a TOML config is not parsed and a `no tomllib` line is printed on stderr; the exit status is unchanged. There is no `tomli` fallback.
- `--format github` escapes `%`, CR and LF in messages and `:` and `,` in properties; the SARIF `uri` is percent-encoded.
- `NO_ASSERT` message now reads `<test> asserts nothing that can fail`.

Added by owner-approved changes (log G1; these resolve the two log-A rows that were not fixed pending a design decision). New warning code `SUPPRESSION_INERT`, reported at the pragma's own line, for an `# audit: ignore[...]` pragma that suppresses nothing: an unknown code (with the nearest valid code suggested, difflib cutoff 0.6), empty or blank brackets, a lower- or mixed-case code, `ALL`, a code pragmas cannot suppress (`SUPPRESSION_NO_REASON`, `SUPPRESSION_INERT`, `SYNTAX`), or a well-formed pragma with no matching finding in its line, scope or file (stale). One finding per pragma. A pragma cannot suppress it; `--ignore SUPPRESSION_INERT` works; staleness is judged before `--ignore`. New flag `--assert-helper PATTERN` (repeatable) and `[tool.pytest-expert] assert_helpers = [...]` in `pyproject.toml`: a call to a matching helper counts as an assertion for `NO_ASSERT` only (`fnmatch`, case-sensitive; a pattern without a dot matches the callee's final name, one with a dot the whole dotted callee, as written or alias-resolved; the call must be live, as for `assert`). A pattern that would match every call (`*`, `?`, `*.*`, `[a-z]`), an empty one, an unknown key or a non-list value exits 2 with a reason. The patterns in effect are echoed in the text summary, in the SARIF run properties and in a GitHub `::notice`; the JSON output is a plain list as before unless helpers are in effect, and then it is `{"findings": [...], "assert_helpers": [...]}` (a consumer that passes `--assert-helper` must handle both shapes). Without `tomllib` the config table is not read and a warning says so. The tests were red first (56 red and 3 green for the helper option, the 3 being negative checks; 31 red for the new code); the `[tool.pytest-expert]` table is also read next to a `pytest.ini` and counts as a project-root marker.

### scripts/diff_guard.py

Exit status: 0 clean, 1 an error finding (any warning with `--warnings-as-errors`), 2 could not check. New code `ASSERT_WEAKENED` (error): a test kept its checks but lost exact comparisons (`==` loosened to truthiness, `is not None`, `in`, `<`), or `pytest.raises(ValueError)` widened to `Exception`, lost `match=` or got `match='.*'`; it is reported only when the check count did not drop. Two warnings were added by owner-approved changes (log G2; they resolve the two log-B rows that were fixed in part pending a decision). `PARAM_HOOK_CHANGED`: a `pytest_generate_tests` function that is new in a test file or `conftest.py`, or whose normalised AST (decorators, signature, statements; docstring dropped) differs from every hook in the same base file; hooks follow file renames; removal is not reported. `SKIP_VIA_HELPER`: a base test gains a call to, or a request for, a helper or non-autouse fixture that can reach `pytest.skip` / `xfail` / `importorskip` / `SkipTest`, or an existing helper that tests already used gains such a skip (one warning per helper, with the number of tests that use it); new tests are not compared; callables in a `conftest.py` that gained a skip are exempt because that is `SKIP_ADDED`. Non-autouse skipping fixtures newly requested by a test were not covered before. `--list-codes` prints 19 codes. Not verified: per the lead, the worker confirmed its near-miss cases pass but not that they can fail (its hand mutation of the live script was blocked); the new tests were seen failing before the code existed (`assert [] == [('PARAM_HOOK_CHANGED', 'warning')]`), and no mutation run covers the new code.

| Problem | Fix |
|---|---|
| Git path lists were read without `-z`, so non-ASCII names came back quoted and were dropped: a deleted or gutted `tests/test_ä.py` and a unicode-named source file gaining a constant `__eq__` went unreported | NUL-separated everywhere (also `--no-renames` and paired renames); both are reported |
| A renamed or moved test file was `TEST_FILE_DELETED` | A pure rename or move (`git mv` or plain `mv`, exact or at least 60% similar) is paired and followed; a file replaced by an unrelated one is still reported. A moved test inherits the module-level skips of its new file when facts are compared |
| `parametrize('a,b', [rows])` rewritten as `argvalues=[rows]` was reported as 3 rows to 0 | Rows are read from the positional or `argvalues=` list, a module constant, class decorators and `pytestmark`; a trimmed `argvalues=` list is `CASES_REDUCED`; equal-count row swaps are an `EXPECTATION_CHANGED` warning. Fixture `params=[...]` is not counted |
| Skip detection saw only some forms | One resolver follows import aliases and assignments and counts module `pytestmark` (single, list, annotated, `+=`, `append`, `insert`), class-body `pytestmark`, module-level `pytest.skip(...)` and `np = pytest.importorskip(...)`, statements under module-level `if` / `try`, unittest skips, `SkipTest` raises, `self.skipTest`, `pytest.param(marks=...)` and autouse skipping fixtures. Tests defined under module- or class-level `if` / `try` / `with` / `for` / `match` are found |
| A skip marker written as a call was counted twice, so adding `reason=` to a bare skip or `strict=True` to a bare xfail was an error | Counted once |
| `CONFIG_FILES` omitted `.pytest.ini` and `.pytest.toml`; the attached `-k'not bad'` form in `addopts` was not seen | Config is read from the file pytest uses (pytest 9.2 order) and a new higher-priority file is compared with the old effective config. `addopts` tokens include attached `-k` / `-m` forms, `--collect-only` / `--co`, `--no-cov`, `--noconftest` and `--ignore-glob` (space form); lost `python_files` / `python_classes` / `python_functions` patterns and gained `norecursedirs` entries are `CONFIG_NARROWED` |
| Coverage floor: deleting the config file that carried `fail_under`, or adding a `.coveragerc` with `fail_under=0`, reported clean; the space form `--cov-fail-under 90` was missed | The floor is the effective one per directory (coverage's file order; `--cov-fail-under` in `addopts` overrides). A lowered, removed or deleted floor and a gained `omit` entry are `COVERAGE_LOWERED`; moving the floor to another file is clean; a floor the base never had is not checked (the code text says so) |
| `expected_literals` used `ast.walk` breadth-first order, so wrapping an assert in a `with` gave a false `EXPECTATION_CHANGED`; an unhashable literal crashed it | Sorted by position; `approx(...)` and `snapshot(...)` unwrapped; left-hand literals and `assertEqual`-style arguments collected; unhashable literals fall back to source text. When the literal count differs, base literals no longer asserted are reported |
| `SPECIAL_CASE`: a negative literal never matched the source side | A unary minus folds into the number; reversed comparisons (`41 == x`), `not in` and `is not` are read |
| `COLLECTION_NARROWED` missed annotated `collect_ignore`, `.insert`, `.append`, `.extend`, `+=`, `collect_ignore_glob`, swapped entries and changed hook bodies | All are seen, with `pytest_collect_file` and `pytest_pycollect_makeitem`; a skip gained in `conftest.py` is `SKIP_ADDED` |
| A binary snapshot under `__snapshots__` crashed the run with `UnicodeDecodeError`; a latin-1 test file with a coding cookie crashed it | File contents are read as bytes and decoded with PEP 263 rules; a file that cannot be decoded is a `SYNTAX` finding. The `SYNTAX` text no longer claims every such file fails to collect |
| Constant `__eq__` detection missed `not <const>`, ternaries with constant branches, `bool(<const>)`, prelude assignments and lambda assignments | All are caught; a real comparison and a guarded return stay clean |
| Trivial asserts missed chained self-comparisons, `1 < 2`, `not <falsy const>`, `or` with a trivial operand, `isinstance(x, object)` and `approx` with an absurd tolerance | All are `ASSERT_TRIVIALIZED` |
| Unreachable asserts counted as checks | Asserts after `return` / `raise` / `break` / `continue`, under constant-false `if` / `while`, and in a nested function nothing calls do not count; any call starting with `assert` (`assertEqual`, `assert_called_with`) is a check |
| `# audit: ignore` inside a docstring counted as a pragma | Pragmas are counted from `tokenize` comment tokens |

Other changes: `--allow` is repeatable and merged; an unknown `--base` says `--base 'x' is not a commit in this repository (<git's message>)`; `--format github` escapes `%`, CR, LF and `:` `,` in properties. Exit 2 (with one line on stderr) for: no test files, snapshots or pytest/coverage config in the base or the working tree ("nothing to compare"), an unparsable TOML or INI config, a non-UTF-8 config, missing `git`, and any internal error. Without `tomllib` (Python before 3.11, no `tomli`) a TOML config ends the run with exit 2 naming the file. Merging asserts with `and` keeps the check count (no `ASSERTS_REDUCED`); collapsing them into a `for` loop or a tuple comparison stays `ASSERTS_REDUCED` on purpose.

### scripts/mutate_check.py

Mutation scores from before and after this change are not comparable: more kinds of site exist, errored mutants leave the score, and annotations and bare constants are no longer mutated.

| Problem | Fix |
|---|---|
| Any non-zero exit of the test command counted as a kill, so a mutant that broke the import (pytest exit 2) was a false kill: `def banner(sep="-" * 3)` with `"-" // 3` was reported killed although no test touches `banner` | Each mutant run is classified by exit code. 0 is `SURVIVED`; a code in `--kill-codes` (default `1`) is `killed`; a timeout is `killed (timeout)`; any other exit (pytest 2, 3, 4, 5, or a signal) is `errored (exit N)`. Errored mutants are left out of the score `killed / (killed + survived)` and have their own summary line. If nothing could be evaluated the exit status is 2 and the message names `--kill-codes` |
| A missing test executable, an unparseable source, a source that is not valid UTF-8 and a baseline command that cannot be started raised an uncaught exception and exited 1, the "survivors" status | One-line `mutate_check: ...` message and exit 2 |
| One baseline run | The baseline runs twice and must pass both times (`baseline is flaky` otherwise, exit 2); the slower run sets the mutant timeout, so the baseline costs twice as much |
| The 10 second timeout floor was fixed and unmentioned | `--timeout-floor` (default 10); the effective timeout is printed and in the JSON as `timeout_seconds`. Timeout kills are labelled `killed (timeout)`, counted separately in `timeouts`, and the summary warns that a slow machine causes them |
| `subprocess.run` killed only the direct child, so test-command grandchildren survived a timeout | The command runs in its own session and process group; the group is killed on timeout, on SIGINT / SIGTERM (the source is restored first, exit 130) and after normal exit. stdin is `/dev/null`. POSIX only |
| `-h` / `--help` without `--` printed the one-line usage and exited 2 | Prints the full help, including the classification and timeout rules, and exits 0 |
| `--function` was applied differently for bare and dotted names | A bare name matches any function or class of that name at any depth plus everything nested in it; a dotted name is a path from the top of the module plus everything nested in it. Closures inside a method are included either way |
| Annotations were excluded only for int and bool sites; bare constant statements were mutated | `SiteFinder._in_annotation` applies to every kind (a `BinOp` inside `Annotated[int, 1 + 2]` is no longer mutated); a bool or int constant that is a statement is not a site. Dead `is_docstring` and an unused `originals` dict removed |
| The docstring called sites `(node, line, kind)` triples | It says `(node id, line, kind, description)` |

New mutation kinds, selectable by name: `matmul` (`@` to `*`, also `@=`), `shift` (`<<` and `>>` swapped, also augmented), `bitop` (`|` to `&`, `&` to `|`, `^` to `&`, also augmented), `neg` (unary minus dropped) and `float` (`c` to `c + 1.0`, skipped when that changes nothing, as for `inf` or `1e300`). The full list is `cmp`, `binop`, `augassign`, `boolop`, `not`, `bool`, `int`, `float`, `name`, `return`, `matmul`, `shift`, `bitop`, `neg`. New flags: `--skip-kinds` and `--only-kinds` (comma-separated, mutually exclusive, exit 2 with the list of known names on an unknown one), `--kill-codes`, `--timeout-floor`, `--show-output`. `X | None` and `None | X` are never mutated (only a type union); other runtime unions such as `isinstance(x, int | str)` become `int & str`, a `TypeError`, so a cheap kill; set and dict `|` is mutated too. Drop `bitop` with `--skip-kinds bitop` if that is noise.

Which test killed a mutant (requested by the owner during the pass): the command's stdout and stderr are read through a pipe, keeping the last 256 KiB. For a plain `killed` mutant, failing test ids are read from pytest's short summary (`FAILED id - msg`, `ERROR id`) and from unittest `FAIL: test (mod.Class.test)` lines. Text report: `killed   by a, b, c (+N more)`, first three. JSON: each mutant has `killed_by` (full list, `[]` when none was recognised, always `[]` for timeouts, errored mutants and survivors). The output never changes the classification. A command that hides the summary (`-p no:terminal`) reports none. `--show-output` prints the last 10 lines of output under each errored mutant; in JSON it adds `output_tail` to every mutant. JSON fields: `source`, `score`, `killed`, `survived`, `errored`, `timeouts`, `baseline_seconds`, `timeout_seconds`, `mutants` (each: `index`, `line`, `description`, `status`, `seconds`, `exit_code`, `killed_by`).

### scripts/flake_hunt.py

| Problem | Fix |
|---|---|
| A test driven by the global `random` module was called `order-dependent` even with `--repeat-seed 3`, because pytest-randomly reseeds `random` per test from the run's seed | With `--repeat-seed`, repeats run with `--randomly-dont-reset-seed` plus a small generated plugin that reseeds `random` (and NumPy) from OS entropy before every test, so an RNG-driven test is `nondeterministic`. The docstring says a plain rerun cannot tell the two apart. Without pytest-randomly the result is always `nondeterministic`, and the text says `fails in runs [..] of N` under a header saying the order is fixed |
| Default `--runs` was 20 in the code and 30 in the docstring | Default is 31, the figure the docstring cites (Gruber et al.: about 31 random-order runs for 95% confidence on order dependence; not re-read against the paper in this pass). The nightly workflow template passes `--runs 31`. 55% more runs than before |
| Exit status 2 was documented for usage errors, but `--runs 0`, a nonexistent path, `-k` / `-m` selecting nothing and pytest collection errors exited 0 ("no flaky tests found") | Exit 2 for `--runs` or `--repeat-seed` below 1 (argparse `positive_int`), pytest exit 4 and 5, collection errors, internal errors, a pytest crash (exit not in 0 to 5, with the output tail; it aborts rather than continuing), an all-skipped suite, and pytest-xdist `-n` other than 0 (refused with an explanation, since workers hide order dependence; xdist enabled through `addopts` is not detected) |
| A flaky test marked `xfail` was not reported | A junit `<skipped type='pytest.xfail'>` is a failure; xpass stays a pass; a steadily xfailing test stays clean |
| `selftest/test_flake_hunt.py` could not tell several broken behaviours from working ones (inverting `outcomes()` kept it green) and skipped silently without pytest-randomly | Rewritten: exact JSON and text, an independent order oracle for `failing_seeds`, all exit-2 paths, `classify()` and `outcomes()` tables. A missing pytest-randomly now fails the test loudly |

### Claude Code hooks (`assets/claude/hooks/`)

The scripts' exit status is the contract for every hook: 0 clean, 1 findings, anything else (exit 2, a crash, a timeout, a missing script, an unknown diff base) means the check could not run, and that is never read as clean. Claude Code treats any exit other than 0 and 2 as a non-blocking error, so the hooks catch their own crashes and turn them into exit 0 with JSON.

| Case | Before | Now |
|---|---|---|
| Event on stdin empty, malformed or not a JSON object (all three hooks) | exit 0, nothing shown | exit 0 with a `systemMessage` saying the gate or check did not run. The Stop gate lets the stop through (without an event there is no project to check), and says so |
| Stop gate: tests not under `tests/` | pytest and the audit ran only on `tests/` | pytest runs on the whole project and decides where the tests are; the audit covers the project root |
| Stop gate: diff guard exits 2 or an unknown diff base | skipped silently | blocks: `could not check`, naming the base and `PYTEST_EXPERT_DIFF_BASE` |
| Stop gate or post-edit: `audit_tests.py` or `diff_guard.py` missing | skipped silently | found next to the hook, else in the skill's `scripts/` folder (when the hook runs from `assets/claude/hooks`); if absent the gate blocks with `<script> was not found ... copy pytest-expert/scripts/<script> into <hooks dir>` and a `systemMessage` |
| Audit or guard times out or crashes | exit 1 (non-blocking) | caught in `run_detector`: the Stop gate blocks with `<script> did not finish within Ns`; exit 1 with empty stdout counts as a crash, not as findings. Timeout `PYTEST_EXPERT_CHECK_TIMEOUT`, default 120 s for the gate, 60 s for post-edit |
| pytest exits 5 (nothing collected) | counted green and remembered as green | a `systemMessage` warning "no tests were collected, so the fast suite checked nothing"; no block (a project without tests must be able to stop); the tree is not recorded as green, so the warning repeats. Deleting every test still blocks through the diff guard (`TEST_FILE_DELETED`) |
| Not a git repository | the diff guard was skipped silently | a visible warning, no block |
| `PYTEST_EXPERT_MAX_BLOCKS=0`, negative or not a number | `0` printed "gave up after 0 attempts; the suite is still red" before anything ran | the three numeric settings (`PYTEST_EXPERT_MAX_BLOCKS`, `PYTEST_EXPERT_STOP_TIMEOUT` default 480, `PYTEST_EXPERT_CHECK_TIMEOUT`) must be positive integers; a bad value is replaced by its default (3 for blocks) and reported in a `systemMessage`; the suite runs. The give-up message names the last problems |
| Any unexpected exception in the Stop gate | traceback, exit 1 (non-blocking) | blocks once with the traceback tail and a `systemMessage`; post-edit and session-start print a `systemMessage` instead |
| Post-edit: relative `file_path` | resolved against the hook's working directory and silently ignored | resolved against the project directory; a non-object `tool_input` gives a `systemMessage` |
| Stop gate fingerprint | any `.py` file | any `.py` file plus the project-root `pytest.toml`, `.pytest.toml`, `pytest.ini`, `.pytest.ini`, `pyproject.toml`, `tox.ini`, `setup.cfg`, `.coveragerc`, `.coveragerc.toml` (one shared `CONFIG_FILES` tuple in `_common.py`; a self-test fails if it drifts from the scripts' lists). Config in subdirectories and lock files are not fingerprinted |
| `stop_hook_active` true | the gate returned before running anything, so a red tree could stop | the early return is gone (log G3, below) |

`stop_hook_active` (log G3; resolves the log-D not-fixed row on an owner decision): the Stop gate no longer allows a stop because the event's `stop_hook_active` is true. A red state blocks until the gate's own counter reaches `PYTEST_EXPERT_MAX_BLOCKS` (default 3) and then gives up with the existing message; when the field is true the block reason starts with `pytest-expert: block N of MAX; the gate stops blocking after block MAX (...)`. A green or unchanged tree is allowed through silently. `PYTEST_EXPERT_MAX_BLOCKS` above `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP` (platform default 8, `0` disables the cap) is lowered to it, with a note in the `systemMessage`, so the gate's give-up comes before the platform override. The counter lives in a state file keyed by a hash of session id and project in the temp directory, is reset only by a green run, a give-up or an escalation, and survives resume and compaction (SessionStart keeps it). The worker read the Claude Code hooks reference and its environment-variable page on 2026-11-20 and quoted: "The `stop_hook_active` field is `true` when Claude Code is already continuing as a result of a stop hook. Check this value or process the transcript to avoid blocking on a condition that will never resolve." and "Claude Code applies an 8-consecutive-continuation cap: after stop hooks have continued the turn eight times in a row, Claude Code overrides the next block and ends the turn. The count of consecutive continuations resets each time Claude calls a tool." Two caveats from the log: it was not exercised against a live Claude Code session, and the counter's state file could be deleted by a shell command, which the platform cap of 8 still bounds (the platform's count resets on each tool call, the gate's does not, so the gate is the stricter limit). The old self-test that asserted a red suite with the field true returns nothing was replaced on purpose; the new tests were red first (5 red, 101 green). `selftest/test_hooks.py` is 107 tests.

`onFailure` (integration, after the logs; from an audit suggestion that had no log row): the `Stop` and `SubagentStop` entries in `assets/claude/settings.json` now set `"onFailure": "block"`, so a gate that cannot start (a wrong path, no `python3`) or that times out blocks the stop. Without it Claude Code carries out the action when a hook fails (hooks reference, "Block the action when a hook fails", read 2026-11-20; needs Claude Code v2.1.295 or later). The post-edit and session-start entries leave it unset. Test: `test_settings_block_the_stop_when_the_gate_itself_cannot_run` (4 cases; the two gate cases failed before the change). Not exercised against a live Claude Code session.

Also: `_common.py` gained `find_script`, `int_env` and `run_detector` (the `AUDIT` and `GUARD` constants are gone). `selftest/test_hooks.py`: `git_project` sets `GIT_CONFIG_GLOBAL` and `GIT_CONFIG_SYSTEM` to `os.devnull` and turns off commit and tag signing, and a missing `git` fails the fixture instead of skipping. The Claude Code hooks reference was fetched on 2026-11-20 for the code (exit-code and JSON semantics); `references/hooks.md` keeps its 2026-11-02 stamp for the doc.

### Wallet example and assets/conftest.py

| Problem | Fix |
|---|---|
| `assets/conftest.py` and the wallet's `tests/conftest.py` network guard blocked `connect`, `getaddrinfo` and `create_connection` but let `socket.connect_ex` and `socket.sendto` (UDP) through | Both are guarded through one shared `allowed()` helper (2- and 3-argument `sendto`). `sendmsg` is listed as a remaining gap. The wallet's `tests/conftest.py` is regenerated from the asset; only `APP_ENV_PREFIXES` differs. New `tests/unit/test_conftest_safety_net.py` checks each guard with closed sockets or numeric hosts, so a disabled guard fails with `OSError` rather than sending a packet |
| `test_fee_is_never_negative` could not fail: random ten-digit accounts all fail the mod-97 check, so the fee code was reached 0 times in 100 examples, and every mutant in the fee body survived | It draws accounts built as `97*k+1` (regular and partner-bank), has no `try`/`except`, and two guard tests prove the strategies only produce accounts the fee accepts. The mutants `max` to `min` and `max(0, ...)` to `max(-1, ...)` in the partner branch now fail it. The symbolic test is kept as the contrast |
| The fuzz corpus replay checked only the oracle, so a wrong value accepted by `parse_amount` passed | Replay classifies by file name: `valid_*` must parse to the value in `EXPECTED_VALUE`, `invalid_*` must raise `AmountError`; the oracle still runs on every input. New seeds `invalid_zero`, `invalid_zero_cents`, `invalid_negative`, `invalid_over_max` (11 seeds now) |
| `Wallet.transfer` was not atomic: a failing target deposit left the source debited, against its docstring | `transfer` remembers the source balance and restores it, then re-raises. Unit and stateful tests for it, plus tests for `deposit`'s return value, an exact transfer, transfer to self and sub-unit amounts |
| `test_store_security.py`: `save_wallet`'s return value and file content, inconsistent files, the corrupt-file branches, the symlink escape, and `SqliteWalletStore.get` / `find_by_prefix` success paths were untested | Tests for each (owner mismatch, negative, NaN and infinite balance, zero balance succeeding, a symlink escape) |
| Unanchored `match=` patterns in the fee tests and the `load_wallet` consistency check were left as exercises (seven non-equivalent mutmut survivors) | Per the wallet README, `tests/unit/test_money.py` anchors the fee messages and `tests/security/test_store_security.py` loads files with a wrong owner, a negative, NaN and infinite balance and a zero balance (which must succeed) |

Wallet figures regenerated on 2026-11-20 on macOS arm64, Python 3.13.2 (README and the references that quote them): `pytest` 135 passed, 3 deselected, 100% branch coverage (was 87 passed, 97%); `pytest -m integration --no-cov` 2 passed; symbolic 1 passed (needs `hypothesis-crosshair`); `mutate_check.py` on `transfer_fee_cents` 32 of 34 killed (2 equivalent, 74 s); mutmut 190 of 196 killed, 6 survivors, all equivalent (was 181 of 194, 13 survivors; 12 s, in a scratch copy). Without `hypothesis-crosshair` the default run is `135 passed, 1 skipped, 2 deselected`. The Atheris figure (16.5 million runs in 46 s) was not rerun, because Atheris does not install on this machine; it is marked machine not recorded.

### Self-tests

| Problem | Fix |
|---|---|
| `test_audit_tests.py` compared code sets, so a wrong line or severity passed | The parametrized test compares the exact sorted `(code, line)` list and each finding's severity over more than 160 cases; every code has a detecting case with the right severity; `--ignore`, all four formats, `--list-codes`, missing path (exit 2), the ignore-file pragma and the INI, TOML and cfg config forms are tested |
| `selftest/conftest.py` `run_audit` audited `tmp_path`, so a seeding `conftest.py`, strict `pyproject.toml` or `integration` directory above the temp dir changed results | A `.git` marker at the project root stops the upward lookups; a test proves hostile ancestors no longer matter |
| `test_diff_guard.py` stayed green with non-equivalent mutants of behaviour its docstring names; a module-level `skipif` hid a missing `git` | Exact `(code, severity)` lists in every case, severities pinned in a table, the exit code for every error code, and legitimate edits assert no findings at all. A missing `git` fails (opt out with `PYTEST_EXPERT_SKIP_GIT_TESTS=1`) |
| `test_hooks.py`: "ignores non-test files" and "skips an unchanged tree" could not fail | Helpers and a red `module.py` must stay silent; a counting test proves the second fire does not rerun the suite; a `.py` edit and each of five config files must rerun it |
| `test_mutate_check.py` had 5 tests | 108 tests when the first round was logged, covering every swap-table entry, `--function`, `--max-survivors`, timeouts, the exits, signals, JSON fields and score arithmetic; the later kinds and `killed_by` work added more (see Verification for the current count) |
| `test_skill_structure.py`: backticked paths with an `#anchor` or trailing arguments were never checked; bare `.md` names starting with a digit, capitals or underscores were skipped; the workflow-step check matched only the first long word of a title | `PATH_RE` accepts both forms; `REF_NAME_RE` matches `[A-Za-z0-9_-]+`; every long word of the step title must appear. The selftest-per-script check requires the file to name the script and hold at least 3 tests with a real assertion; the diagnoses are every `examples/diagnoses/*.md` except `README.md`; evals need strings of at least 10 characters that are distinct; the "N hard gates" check covers every file except `AUDIT.md` and requires `INSTALL.md` to state it; the mutmut figures are read from the reference and compared with the wallet README; the wallet README's default-run passed count is compared with `pytest --collect-only` (skipped, with a reason, without `hypothesis` or `pytest-cov`) |

The skill's own self-tests and assets now pass the skill's own audit (log H). Before: `audit_tests.py selftest` reported 29 findings (BRANCHING 12, FLOAT_EQ 14, NO_ASSERT 1, SLEEP 1, WEAK_ASSERT 1) and `audit_tests.py assets` 1 (a stale pragma, `SUPPRESSION_INERT`). Of these, 22 were real weaknesses and the tests were strengthened, each with a recorded break-it check where one applies: an `if` on a parameter became an arrange table; loops became parametrized cases or comprehension assertions; float comparisons of scores and timeouts use `pytest.approx`; the `git` opt-out became a `skipif` whose body runs `git --version`; a helper's return value is now asserted. 8 were deliberate patterns and got line-scope pragmas with reasons (an `if os.access(...)` guard for chmod under root, exact binary fractions in float equality, a bounded poll for a subprocess that signals only through files). No finding was a detector false positive. The stale pragma in `assets/test_async_template.py` was removed. A new structure test (`test_the_skills_own_tests_pass_their_own_audit` and a check that it can fail) keeps `selftest/` and `assets/` audit-clean. The worker wrote the three lock-in tests after the fixes, not test-first; it showed the real check failing by planting a `FLOAT_EQ` in a self-test.

### Documentation and `metadata.test-command`

- `metadata.test-command` in `SKILL.md` is now `uv run --no-project --with pytest --with pytest-randomly --with hypothesis --with pytest-cov --with pytest-timeout python -m pytest selftest -q -rs`. It was `... --with pytest --with pytest-randomly python -m pytest selftest -q`. The extra packages are needed by the wallet README count check and the flake hunter's tests; `-rs` prints why anything skipped.
- `SKILL.md`: step 2 also copies `scripts/diff_guard.py`; the script table says the mutation checker "says which test killed each mutant" and the flake hunter's default is 31; exit statuses 0, 1 and 2 are described once ("never a pass"); "random generation missed the checksum-gated bug in 3 of 3 runs at 1,000 examples" became "random draws over any ten-digit account missed it in 14 of 15 runs (1,000 examples)" plus the statement that a property which builds valid accounts and the solver both found it (the 14 of 15 is the log's 2026-11-20 rerun of the unconstrained strategy in `demo/test_v0_findings.py`).
- `INSTALL.md`: exit statuses of the four scripts; the self-test environment and what skips without each package; `PYTEST_EXPERT_SKIP_GIT_TESTS`; the wallet install line; the re-run stamp.
- Edited to match the behaviour, messages, figures and exit statuses above (the files `git diff --stat main` lists; the per-change notes are the `doc_changes` fields of the fix logs): `references/hooks.md`, `references/rules.md`, `references/flaky-tests.md`, `references/mutation-testing.md`, `references/llm-written-tests.md`, `references/parametrize-and-doubles.md`, `references/property-fuzz-symbolic.md`, `references/papers/gruber-flaky-python.md`, `references/papers/petrovic-mutation-practices.md`, `examples/diagnoses/01-green-suite-bugs-in-prod.md`, `02-agent-gamed-the-suite.md`, `03-ci-only-flake.md` and its `README.md`, and the wallet and demo READMEs. `references/mutation-testing.md` quotes the new mutmut run (196 mutants, 190 killed, 6 survived, 2026-11-20); its older output blocks keep their earlier capture stamps.
- `assets/test_async_template.py`: two `fake_sleep.delays == [0.5, 1.0]` asserts now use `pytest.approx` so the template stays audit-clean under the wider `FLOAT_EQ`. `assets/github/workflows/nightly.yml`: `flake_hunt.py --runs 31`.
- A second documentation pass for the follow-ups above: `SKILL.md` (still 260 lines), `INSTALL.md`, `references/rules.md`, `references/llm-written-tests.md` and `references/hooks.md` (26 and 19 codes, the helper option, `SUPPRESSION_INERT`, the two warnings, the `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP` row, 107 hook self-tests).
- `AUDIT.md`: new top section for this audit.
- `audit-skill` 0.1.3 carries the re-synced `mutate_check.py`.

### Not changed / still open

Resolved after the logs of the audit pass, by the follow-ups above: the Stop gate and `stop_hook_active` (ignored in favour of the gate's counter), custom assertion helpers (`--assert-helper`), typo'd and stale pragmas (`SUPPRESSION_INERT`), `pytest_generate_tests` and skips through helpers (`PARAM_HOOK_CHANGED`, `SKIP_VIA_HELPER`), and the self-tests' own audit findings (log H).

Still not fixed, with the reason (details in `AUDIT.md`):

- `BRANCHING` exempts `for` over any call (`list(CASES)`, `sorted(CASES)`, `zip(a, b)`), which cannot be told from `range(n)` without evaluating the call. Pinned by a test.
- A trivial pragma reason such as `.` is accepted (checked 2026-11-20: `# audit: ignore[FLOAT_EQ] .` suppresses the finding without a warning); a length check is trivially gamed, so what counts as a reason is a policy question.
- `test_every_code_is_documented` still accepts a backticked code anywhere in `rules.md`; a stricter heading check depends on the final `rules.md` layout.
- `SKILL.md` stays at the 260-line limit; any added line fails `test_skill_md_is_short`.
- Partly fixed and left: fixture `params=[...]` rows are not counted by the diff guard; collapsing asserts into a loop stays an error; dict lookups and `match` / `case` special-casing are not comparisons and are not detected; helpers imported from modules other than `conftest.py` are not followed (as stated by the lead); `integration_tests/` directories are not recognised.
- Also left as is, per the logs: `NETWORK` does not follow session fixtures, helpers or methods on session objects; `UNSEEDED_RANDOM` ignores pytest-randomly; Django `TestCase` is not flagged; a skip mark stored in a variable first is not seen; `np.float64(np.nan)` and `x in [math.nan]` are not seen as NaN; `killed_by` is limited to what pytest or unittest print; xdist enabled through `addopts` is not detected by the flake hunter; the Stop gate fingerprint does not cover config in subdirectories or lock files.
- Not verified: the Stop gate change against a live Claude Code session; the new diff guard warnings are not covered by any mutation run; the Atheris fuzz figure was not rerun; older mutation output blocks quoted in `references/mutation-testing.md` and `examples/diagnoses/` keep their 2026-11-02 capture; the 3,012 items the audit did not check (every reference file, the paper digests, `SKILL.md`, `INSTALL.md`, the asset templates, the workflows) were not audited.

### Verification

| Date | Machine | Versions | Check | Result |
|---|---|---|---|---|
| 2026-11-20 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0, Hypothesis 6.170.17, pytest-cov 7.1.0 (pytest-randomly 5.1.0 switched off) | `pytest selftest -q -p no:cacheprovider -p no:randomly` | 1429 passed, 0 skipped, in 201 s (audit 450, diff guard 462, flake hunt 54, hooks 111, mutate_check 179, structure 173). On the `main` source, same command and machine: 260 passed (audit 56, diff guard 46, flake hunt 3, hooks 17, mutate_check 5, structure 133) |
| 2026-11-20 | macOS arm64 (Apple Silicon) | same, without hypothesis-crosshair | `examples/wallet`: `pytest` | 135 passed, 1 skipped, 2 deselected, 100% branch coverage (on `main`: 87 passed, 1 skipped, 2 deselected, 96.95%) |
| 2026-11-20 | macOS arm64 (Apple Silicon) | same | `examples/wallet`: `pytest -m integration --no-cov` | 2 passed, 1 skipped, 135 deselected |
| 2026-11-20 | macOS arm64 (Apple Silicon) | same | `audit_tests.py` on `examples/wallet/tests`, `assets`, `selftest` | clean (11 files), clean (5 files), clean (7 files); `selftest` had 28 to 29 findings before log H |
| 2026-11-20 | macOS arm64 (Apple Silicon) | same | `audit_tests.py --list-codes`, `diff_guard.py --list-codes` | 26 lines, 19 lines (25 and 16 on `main`) |
| 2026-11-20 | macOS arm64 (Apple Silicon) | same | `audit-skill`: `pytest tests -q` | 215 passed |

Not rerun: Atheris (not installable here); `mutate_check.py` on `transfer_fee_cents` and mutmut on the wallet (figures above are the README's, 2026-11-20); the wallet's symbolic layer (needs `hypothesis-crosshair`). The self-test count rose during the work as follow-up logs landed (1,270, 1,336, 1,422, 1,425). Mutation scores for the fixed scripts are in `AUDIT.md`; the lead's sequential reruns started before the follow-ups and do not cover them.

## 2026-11-19: type fixes for Pyright

`pyright pytest-expert` reported 75 errors and 3 warnings (78 diagnostics) on 2026-11-19 (machine not recorded); it now reports 0 and 0. No runtime behaviour changed: `selftest` gives 257 passed, 3 skipped before and after, and the wallet example suite gives 89 passed, 1 skipped before and after.

| Where | What was wrong | Change |
|---|---|---|
| `assets/conftest.py`, `examples/wallet/tests/conftest.py` | The socket guards typed `address`, `host` and `*args` as `object`, which `socket.getaddrinfo` and `real_connect` do not accept | Typed them `Any` |
| `scripts/audit_tests.py` | `visit_TryStar = visit_Try` did not match the base signature; `fn_stack` was `list[ast.AST]`, so `decorator_list` was unknown | `visit_Try` takes `ast.Try \| ast.TryStar`; `fn_stack` is a list of function nodes |
| `scripts/diff_guard.py` | A stored `isinstance` result did not narrow `node`; `__doc__` may be `None` | Inline the checks with `continue`; assert `__doc__` is set |
| `scripts/mutate_check.py` | `is_docstring` and `in_scope` returned a list, not `bool`; mutation sites read fields off `ast.AST` | `len(...) > 0`; the mutated node is cast to `Any` once |
| `selftest/conftest.py`, `examples/wallet/tests/security/test_fuzz_corpus_replay.py` | `spec_from_file_location` may return `None` | `assert spec is not None and spec.loader is not None` |
| `selftest/test_hooks.py`, `examples/wallet/tests/integration/test_sqlite_file_store.py` | `fire()` and `store.get()` return `None` when nothing is found | `assert x is not None` before use |
| `assets/test_async_template.py` | `gather_profiles` took `Awaitable`, but `TaskGroup.create_task` needs a coroutine | Typed as `Coroutine[Any, Any, dict]` |
| `assets/fuzz_target_template.py`, `examples/wallet/fuzz/fuzz_parse_amount.py`, `examples/wallet/demo/fuzz_v0.py` | `atheris` is not installed here and its bundled stub lacks `instrument_func` and `instrument_imports` | Targeted `# pyright: ignore` with reason; `fuzz_parse_amount.py` imports `atheris` again in its second `__main__` block instead of relying on a possibly unbound name |
| `assets/test_async_template.py` (`_sleep.delays`), `examples/wallet/tests/unit/test_money.py` (`parse_amount(12.5)`) | A function attribute set on purpose; a wrong type passed on purpose | Targeted `# pyright: ignore` with reason |

## 2026-11-15: evals run on the corrected version

All 17 cases in `evals/evals.json`, one run each, on macOS arm64 with Claude Opus 5.5 subagents in Claude Code. With the skill: fresh subagents told to load it, kept away from `evals/`. Without: fresh subagents told not to load any skill or read the skill files. A separate fresh subagent graded each pair against the expectations without being told which reply was which (replies that name the skill's own files partly give that away).

| | With the skill | Without |
|---|---|---|
| All 61 expectations | 58 | 32 |
| The 47 that do not name a file or script of this skill | 45 | 28 |

One run per case, so this shows direction, not rates. The with-skill misses: `green-but-bugs` did not say that a green run over a suite the audit flags is not a pass; `github-actions-ci` did not offer `security.yml`; `fuzz-config-parser` replayed the crash folder, not the corpus.

The grader questioned four expectations; they are unchanged until reviewed. In `async-retry-backoff`, injecting the sleep function and using `@pytest.mark.asyncio` are required, though patching `asyncio.sleep` and auto mode also work. In `asyncio-internalerror`, setting `asyncio_default_test_loop_scope` is required, though it is not needed for the fix. In `fuzz-config-parser`, "replays the corpus" is ambiguous. In `parse-duration-tests`, "not from running the function" cannot be seen in a reply.

The trigger evals in `evals/trigger_evals.json` have not been run.

## 2026-11-10: skill-builder conformance

`check_skill.py --strict` gave 0 errors and 5 warnings after the 2026-11-09 fixes; it now gives 0 and 0.

| Warning | Fix |
|---|---|
| SB006: description had no negative scope | Adds "Not for" other languages, bats tests (bash-scripting-specialist), and a skill's own claims (audit-skill) |
| SB007: `tests/conftest.py` read as a bundled path | It meant the user's project; reworded to "into the project's `tests/`" |
| SB010: no trigger evals | `evals/trigger_evals.json`: 10 should trigger, 10 near misses that belong to sibling skills or other languages |
| SB022: no `metadata.test-command` | Added; it installs pytest-randomly because `test_flake_hunt.py` skips without it. skill-builder's packager now runs the selftests in the unpacked copy |
| SB043: no "Extend this skill" section | Added, naming the structure test that catches each missed step. The Sources section was removed to stay under the 260-line cap: each of its lines repeated a row of the Reference files table |

Added `test_test_command_runs_the_selftests` and `test_trigger_evals_have_both_kinds`, each watched failing first; `test_frontmatter` now accepts a `metadata` block. The trigger evals have not been run.

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-10 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0, pytest-randomly (the `metadata.test-command` environment, without Hypothesis) | 260 passed, 0 skipped |

## 2026-11-09: corrections to the mutation-testing reference

Found by an `audit-skill` eval run (2026-11-07) on `references/mutation-testing.md`, then rechecked by hand: each paper claim against the paper's own text, each number by rerunning it. Details in `AUDIT.md` under 2026-11-09.

### Fixed

| Problem | Evidence | Fix |
|---|---|---|
| Chen et al. (ASE 2020) reported backwards: the reference and its digest said mutation-based selection beat coverage-based at equal size | §6.4, Fig. 5: coverage-selected sets "are as strong or stronger ... (at least until coverage is satisfied)"; §8: mutation "is most effective when employed after coverage has exhausted its usefulness" | Reference and digest rewritten; the reduction result (Fig. 7), which was right, kept |
| mutmut result on the wallet example stated as 183 killed, 11 survived | rerun twice on 2026-11-09, macOS arm64, Python 3.13.2, mutmut 3.8.1, pytest 9.2.0: 181 killed, 13 survived | New figures with their stamp in the reference and the example README; the two extra survivors (encoding mutants) classified as equivalent, with the reason |
| "A survivor is a behavior change no test detects" | the same file shows equivalent survivors | Reworded |
| mutmut on a whole package takes "minutes"; mutation runs are "too slow per push" | 6 s and 9 s for the example | Cost stated as mutants times suite time, with the example figure; nightly for whole packages, `mutate_check.py` before pushing |
| "workflow step 6" for proving tests can fail | SKILL.md numbers it step 5 | Corrected, and guarded by a new test |
| Zhao et al. green-suite requirement cited as §3.3 | the passage is footnote 1, attached to RQ2 in §3.1 (printed on the §3.3 page) | Locator corrected in the reference and its digest |

### Added

- `test_workflow_step_references_match_skill_md` and `test_mutmut_result_is_stated_the_same_everywhere` in `selftest/test_skill_structure.py`.

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-09 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0, Hypothesis, pytest-randomly | 258 passed, 0 skipped |

## 2026-11-02: re-verification, test-gaming defenses, research digests

Re-verified on Python 3.13 with current releases, then expanded. Gaps, corrections,
how each source was checked, and what was left undone are in `AUDIT.md`.

### Fixed

| Problem | Evidence | Fix |
|---|---|---|
| pytest-asyncio 1.4.1 plus `filterwarnings = ["error"]`: `INTERNALERROR` at startup, in the config asset and in the wallet example | reproduced; the wallet example still crashed after the first fix whenever the plugin was installed | a targeted `filterwarnings` ignore that works with and without the plugin (asset and wallet verified both ways), plus commented asyncio keys for projects that depend on the plugin |
| sqlite3 connections leaked in the demo and the security template; Python 3.13 turns that into a `ResourceWarning` that fails a later test or crashes at exit | 3.13 warns, 3.12 does not; templates `1 failed, 67 passed` under strict config | closing fixtures |
| Audit `NO_ASSERT` false positive on `with pytest.RaisesGroup(...)` | async template | `RaisesGroup`/`RaisesExc` count as checks |
| Flaky rerun guidance gave only the non-order-dependent figure | Gruber et al. §IV-C | both figures, plus reruns needed to classify one failure |
| Dead Just et al. 2014 link; Inozemtseva and Holmes cited without Chen et al. 2020 | 404; digest | new link; Chen 2020 added |
| Versions | pytest 9.1.1, Hypothesis 6.168.3, CrossHair 0.0.111, mutmut 3.8.0, ruff 0.16.10, zizmor 1.30.1, setup-python v7.0.0, upload-artifact v7.0.1 | pytest 9.2.0, Hypothesis 6.170.14, CrossHair 0.0.112, mutmut 3.8.1, ruff 0.17.3, zizmor 1.31.1, setup-python v7.1.0, upload-artifact v7.0.2 (SHAs from `git ls-remote`) |
| Em dashes in references | 92 | removed |

### Added

- `scripts/diff_guard.py`: test-weakening edits since a git base (deleted tests or files,
  dropped parametrize rows, fewer assertions, trivial asserts, new skips, pragmas,
  collection hooks, config deselection, lowered coverage floor) and source-side
  warnings (constant `__eq__`, special-casing a test's literal), with snapshot rewrites
  reported too.
- Hooks: SessionStart base commit, diff guard and escalation exit in the Stop gate,
  `SubagentStop` entry, stricter block message.
- pytest 9.1/9.2: `max_warnings` ratchet (exit code 6), `-m` marker validation,
  `YIELD_FIXTURE` audit code, `RaisesGroup` rule, `register_fixture` section.
- References: `async-testing.md`, `llm-written-tests.md`, `sources.md`, 16 paper
  digests; snapshot testing; generator checks for Hypothesis.
- `assets/test_async_template.py`; `examples/diagnoses/` (five worked diagnoses).
- Selftests for every script and for the skill's structure; evals 6 to 17 in the
  expectations format.

### Changed

- Workflow step 4 runs the diff guard; step 5 accepts "unproductive, with a reason";
  a short policy for when tests and code disagree.
- Solver search on the wallet demo now about 0.4 s (was 1.7 s on CrossHair 0.0.111).
- Scripts are executable (`chmod +x`) and pass ruff 0.17's expanded default rule set; `ci-cd.md` explains that default change.

### Verification at release (machine not recorded)

| Check | Result |
|---|---|
| `pytest selftest` | 255 passed, machine not recorded (audit 56, diff guard 46, hooks 17, mutate_check 5, flake_hunt 3, structure 128) |
| `examples/wallet`: `pytest` | 87 passed, 96.95% branch coverage |
| `examples/wallet`: `-m integration --no-cov`, `-m symbolic --no-cov` | 2 passed, 1 passed |
| `examples/wallet/demo` | 6 failed, 1 passed (as designed) |
| `mutate_check.py` on `transfer_fee_cents` | 32 of 34 killed (2 equivalent, as before) |
| Templates under the strict config with pytest-asyncio keys | 68 passed, 1 skipped, 1 xfailed, in file order and random order |
| Audit on `assets/`, `examples/wallet/tests`, `selftest/` | clean |
| `ruff check --config assets/ruff-pytest.toml` and `ruff check --isolated` (ruff 0.17's expanded defaults) on scripts and selftests | clean under both |
| Independent fact-check (separate agent re-read the papers, reran the scripts and diagnoses, checked pins with `git ls-remote`) | 2 errors and 12 imprecisions, all fixed: `CASES_REDUCED` missing from the guard's table, a digest contradicting Chen 2020 §6.4, `RaisesExc` deletions not detected by the guard (now tested), `-m` validation undocumented, SessionStart missing `fork`, Gruber 13% row mislabeled, a Petrovic correlation removed as unverifiable, stale pins; about 60 claims confirmed |
| zizmor 1.31.1 (auditor persona), actionlint 1.7.12, pre-commit validate-config | no findings, clean, valid |
| `conftest_ml.py` fixtures on PyTorch 2.15 CPU | 4 passed (shape, overfit, determinism, checkpoint) |

Machine for every row above: not recorded. Rerun 2026-11-03 on macOS arm64 (Apple Silicon), Python 3.13: `pytest selftest` 253 passed, 3 skipped; `examples/wallet` 87 passed, 1 skipped, 2 deselected. The other rows were not rerun.

## 2026-10-04: audit and expansion

Machine for the results in this release: not recorded.

The previous version (August 2026) was audited by running its own tooling against
deliberately broken tests and its own reference module. Every fix below was
reproduced before it was made and re-checked after.

### Fixed

| Problem | Evidence | Fix |
|---|---|---|
| `audit_tests.py` missed 7 defect classes | reported `clean: no findings` on 7 defective tests; pytest passed 6 of them | negative and left-side float literals, `approx` without `assert`, tuple asserts, `patch.object` without autospec, swallowed `AssertionError`, existence-only asserts; one regression test each in `selftest/` |
| `XFAIL_NOT_STRICT` fired even when the shipped config sets `strict = true` | every xfail flagged under the skill's own config | the audit reads the nearest pytest config |
| `SHADOWED_TEST_NAME` flagged `TestA.test_x` vs `TestB.test_x` | one name table per file | per-scope tables |
| `PYTHONHASHSEED` in a conftest counted as seeding | it only affects child processes and seeds string hashing | removed from seed detection; docs corrected |
| Flaky-test guide blamed dict iteration order | dicts are insertion-ordered | only `set`/`frozenset` of `str`/`bytes` depend on the hash seed |
| Reference `Wallet` accepted `withdraw(nan)` with 19 passing tests and a clean audit | balance became `nan` | property example rewritten with `@example(math.nan)`; the old one used `allow_nan=False` and hid the bug |
| Network guard blocked loopback and Unix sockets | multiprocessing and local test servers fail | loopback and AF_UNIX allowed; gaps documented |
| `frozen_clock` limits undocumented; ruff PT022 | `from time import time` and `datetime.now()` unaffected | documented; `return` instead of `yield` |
| Config passed `--strict-config` and `--strict-markers` on top of `strict = true` | redundant | removed |
| Ruff per-file-ignore globs never matched when run with `--config` | 306 findings, mostly S101 in tests | globs start with `**/`; `mutants/` excluded |
| Ruff SIM105 rewrites a swallowed assertion into `contextlib.suppress` | makes the hidden failure permanent | SIM105 off for tests; audit reports `SWALLOWED_FAILURE` |
| Stale versions | ruff 0.15.11, ruff hook id `ruff`, pre-commit-hooks v5.0.0, checkout v4, setup-python v5 | ruff 0.16.10, `ruff-check`, v6.0.0, checkout v7.0.1, setup-python v7.0.0 |
| CI workflow security | no `permissions`, tag-pinned actions, credentials persisted | least privilege, SHA pins, `persist-credentials: false`, concurrency; actionlint clean, zizmor auditor persona no findings |

### Added

- Audit: `SWALLOWED_FAILURE`, `BARE_COMPARISON`, `NAN_COMPARE`, `WEAK_ASSERT`,
  `SUPPRESSION_NO_REASON`; inline suppression with required reason; `github` and
  `sarif` output; severities; integration-directory exemption for `NETWORK`;
  `asyncio.sleep(0)` allowed; fixtures checked for `SLEEP`.
- `scripts/mutate_check.py`: stdlib targeted mutation testing (workflow step 5).
- `scripts/flake_hunt.py`: multi-seed reruns, order-dependent vs nondeterministic.
- Claude Code hooks: post-edit audit and a Stop gate with fingerprint skip and loop guard.
- Workflows: `tests.yml`, `nightly.yml`, `security.yml`, `dependabot.yml`.
- Templates: property, security, fuzz target, integration conftest.
- Hypothesis profiles (`dev`, `ci`, `nightly`, `crosshair`) and directory markers in `conftest.py`.
- References: property/fuzz/symbolic, security, mutation, integration, flaky tests,
  CI/CD, hooks.
- `examples/wallet` worked example with a `demo/` of planted bugs.
- `selftest/` (60 tests) and `evals/evals.json`.

### Changed

- `references/ci-and-coverage.md` split into `ci-cd.md` and `flaky-tests.md`.
- `assets/ci.yml` replaced by `assets/github/workflows/tests.yml`.
- Copying this version over an older install leaves `references/ci-and-coverage.md`
  and `assets/ci.yml` behind; delete them, or replace the folder (see INSTALL.md).
- Package no longer includes empty `.benchmarks/` directories created by
  pytest-benchmark during verification.
- `SKILL.md` workflow: every step has a completion check; mutation checking is now
  step 5; new escalation step and testing ladder. Rules 28 to 31 added.
- Shell snippets in docs contain no `#` comment lines, so they paste cleanly into zsh.

### Verification at release (machine not recorded)

| Check | Result |
|---|---|
| `pytest selftest` | 60 passed |
| `examples/wallet`: `pytest` | 87 passed, 97% branch coverage |
| `examples/wallet`: `-m integration`, `-m symbolic` | 2 passed, 1 passed |
| Templates | 39 passed, 1 skipped (Docker unavailable, skip path verified) |
| Audit on every test file in the skill | clean |
| `ruff check` across the skill (demo excluded on purpose) | clean |
| actionlint, zizmor (auditor), pre-commit validate-config | clean, no findings, valid |
| mutmut on the example | 183 of 194 mutants killed |
