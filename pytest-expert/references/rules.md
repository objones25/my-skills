# The rule set, with rationale and sources

Every rule in SKILL.md, expanded: what it means, why it exists, what enforces it, and where it
comes from. Read this when you need to justify a rule, resolve a conflict between two of them,
or decide whether a specific case earns an exception.

## Contents

- [How enforcement works](#how-enforcement-works)
- [Structure](#structure): rules 1–5
- [Assertions](#assertions): rules 6–10
- [Isolation and determinism](#isolation-and-determinism): rules 11–18
- [Test doubles](#test-doubles): rules 19–22
- [Markers and skips](#markers-and-skips): rules 23–25
- [Coverage](#coverage): rules 26–27
- [Rules added in the October 2026 audit](#rules-added-in-the-october-2026-audit) (rules 28 to 31)
- [Audit code index](#audit-code-index)
- [When to break a rule](#when-to-break-a-rule)
- [Source list](#source-list)

## How enforcement works

Three layers, deliberately overlapping. A rule that only exists in prose gets violated within
a week.

| Layer | Tool | Catches |
|---|---|---|
| **Config** | pytest `strict = true`, `filterwarnings = ["error"]` | Unregistered markers, bad config keys, duplicate parametrize IDs, xfail that passes, deprecation warnings |
| **Lint** | `ruff` PT001–PT031 | Fixture and parametrize misuse, over-broad `raises`/`warns`, unittest assertions in pytest code |
| **Audit** | `scripts/audit_tests.py` | The semantic traps lint does not model: assertion-free tests, fake mock assertions, unseeded randomness, network, sleep, module-state leaks |
| **Mutation** | `scripts/mutate_check.py`, mutmut | Tests that run, assert, and still cannot fail when the code is wrong |

Run all of them. They do not overlap as much as they look like they do. In the
October 2026 audit, the old audit script reported `clean: no findings` on a file of
seven defective tests while pytest passed six of them; ruff caught two of the seven.

---

## Structure

### Rule 1: One behavior per test

A test proves one claim. If the name needs "and", split it.

The reason is diagnostic, not aesthetic. When `test_user_creation_and_email_and_billing` fails,
the report tells you three subsystems might be broken. When `test_signup_sends_welcome_email`
fails, you know. The Hitchhiker's Guide states it as: *"A testing unit should focus on one tiny
bit of functionality and prove it correct."*

The corollary matters more: a test that fails should point at one cause. Multiple assertions in
one test are fine when they describe one behavior from different angles (the return value *and*
the resulting state), because they fail together.

Source: python-guide.org; Real Python ("Write small, focused tests").

### Rule 2: Long, declarative names

`test_withdraw_rejects_amount_over_balance`, not `test_wd2` or `test_withdraw_2`.

The name is the failure message. It appears in CI output, in the `-v` listing, in the summary
line from `-ra`, and in your inbox at 3am. A reader who has never opened the file should be able
to tell from the name alone what broke.

The Hitchhiker's Guide is blunt about it: *"Use long and descriptive names for testing functions
... when a test fails, you know what failed."* And: *"testing code may be read more often than
production code"*: an unclear test's value drops sharply.

Source: python-guide.org (rules 9 and 10).

### Rule 3: Arrange / Act / Assert

Three visually separated phases: build the world, do the one thing, check what happened. Brian
Okken calls this the best template for most tests, and notes the alternative name Given/When/Then.

The value is that it makes the "Act" step obvious. When a test has three interleaved actions,
you cannot tell which one the assertion is really about, and neither can the next person.

Repeated Arrange blocks are the signal to extract a fixture. Repeated Act+Assert blocks with
different data are the signal to parametrize.

Workflow tests with interleaved assertions are not forbidden: Okken's phrasing is that they are
"not evil, just trickier to deal with." Use them for genuine state machines, not as an excuse.

Source: pythontest.com (Brian Okken).

### Rule 4: No `if`/`for`/`while` in a test body

Audit code: `BRANCHING`.

Branching in a test means the test has cases. Loops mean the test has cases. `parametrize` is
what handles cases, and it is strictly better in three ways:

- **Isolation.** Each case is a separate test. One failure does not stop the others.
- **Reporting.** The failing case appears in the report by ID. A loop reports `test_x` failed at
  iteration 2, if you are lucky enough that the assertion message says so.
- **Selection.** `pytest -k withdraw-minimum` runs one case.

Worse, a branch in a test can silently skip the assertion. `if platform == "linux": assert ...`
passes trivially everywhere else, and nobody notices for a year.

The genuine exception is a loop that *builds* data rather than driving cases:
`items = [make(i) for i in range(3)]` in the Arrange block. The audit only flags `for` loops
whose iterable is a literal collection, which is the case-driving shape.

### Rule 5: No duplicate test names in a module

Audit code: `SHADOWED_TEST_NAME`.

Python has no overloading. Two `def test_foo` in one file means the second silently replaces the
first, and the first test stops running the moment the copy-paste happens. Nothing warns you:
not pytest, not the type checker. Coverage even drops, which people then explain away.

The check is per scope: `TestA.test_x` and `TestB.test_x` are different tests and are not flagged.


---

## Assertions

### Rule 6: Every test asserts something

Audit code: `NO_ASSERT`.

A test with no assertion checks exactly one thing: that the code did not raise. That is
occasionally a legitimate smoke test, and when it is, it should say so explicitly rather than
appearing to check something it does not. The audit counts `assert`, `pytest.raises`,
`pytest.warns`, `pytest.fail`, and the real `Mock.assert_*` methods.

This one is common in AI-generated and hurried tests, and it is the reason coverage numbers lie:
an assertion-free suite can reach 100% line coverage while proving nothing.

A bare `x == pytest.approx(y)` without `assert` does not count as an assertion; the old
audit counted it. For a deliberate smoke test, suppress on the `def` line with a reason:
`def test_x():  # audit: ignore[NO_ASSERT] smoke test: must not raise`.

`print()` in a test (`PRINT`, warning) is the same failure in another shape: it shows a
value instead of checking it. If output is the behavior, capture it with `capsys`.

### Rule 7: No assertion that cannot fail

Audit code: `ALWAYS_TRUE`.

`assert True`, `assert 1`, `assert "the balance should update"`: the last one is the sneaky
version, because it reads like documentation and passes because non-empty strings are truthy.

The tuple form is the subtle one: `assert (result == 3, "msg")` asserts a non-empty
tuple, which is always truthy, so the comparison inside never matters.

### Rule 8: Floats never compare with `==`

Audit code: `FLOAT_EQ`.

`0.1 + 0.2 == 0.3` is `False` in IEEE 754 binary floating point, on every machine. Use
`pytest.approx`:

```python
assert result == pytest.approx(0.3)                  # default: rel=1e-6
assert result == pytest.approx(0.3, abs=1e-9)        # absolute tolerance
assert result == pytest.approx(expected, rel=1e-3)   # relative, for larger magnitudes
assert results == pytest.approx([0.1, 0.2])          # works on sequences and dicts
```

Verified signature (pytest 9.2.0):

```
pytest.approx(expected, rel=None, abs=None, nan_ok=False)
```

`rel` is a *fraction*, not a percentage: `rel=1e-3` means "within 0.1%". When both are given, the
comparison passes if either tolerance is satisfied.

The rule is stricter than "only when it matters" on purpose. A comparison to an exact float
literal that happens to work today breaks the moment the arithmetic changes upstream, and the
resulting failure looks like a real regression.

For tensors, use `torch.testing.assert_close` instead: it checks dtype, shape, and device too.
See `ml-testing.md`.

The audit checks both sides of the comparison and negative literals (`-0.5` is a unary
minus applied to a constant, which the old audit missed). Comparing against NaN is a
separate code, `NAN_COMPARE` (rule 30).

### Rule 9: `pytest.raises` is specific and matched

Audit code: `BROAD_RAISES`. Lint: PT011 (`raises-too-broad`), PT010, PT012.

```python
# passes when your code fails for entirely the wrong reason
with pytest.raises(Exception):
    withdraw(500)

# passes only for the failure you meant
with pytest.raises(InsufficientFunds, match=r"cannot withdraw 500\.0"):
    withdraw(500)
```

`match=` takes a regex applied with `re.search`, so escape regex metacharacters in literal text:
`\.` for a decimal point, or wrap in `re.escape()`.

Keep the `with` block to the single statement expected to raise (PT012). A three-line block gives
you no way to know which line raised, and an early line raising the right exception type for the
wrong reason produces a green test.

Configure `raises-require-match-for` in ruff (see `assets/ruff-pytest.toml`) to extend this to
`ValueError`, `OSError`, and other over-broad built-ins.

### Rule 10: Exact expected values, not vague properties

Real Python's example is the canonical one:

```python
# the test passes for a broken add()
def test_add():
    a = random.randint(50, 150)
    b = random.randint(50, 150)
    assert 50 <= add(a, b) <= 300

# the test fails when add() is broken
@pytest.mark.parametrize(("a", "b", "expected"), [(100, 10, 110), (200, -50, 150)])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

Two defects in the first version, and they compound: random inputs make failures irreproducible,
and the range assertion is so loose that `a + b + 1` still passes.

Range assertions have real uses: a timing bound, a probabilistic property, a tolerance on a
learned metric. When you write one, write it because the exact value genuinely is not knowable,
not because you did not want to compute it.

Source: Real Python.

---

## Isolation and determinism

### Rule 11: Tests pass alone, in any order, in parallel

*"Each test unit must be fully independent. Each test must be able to run alone, and also within
the test suite, regardless of the order that they are called."*: python-guide.org, rule 2.

Verify it rather than assuming it:

| Check | Command |
|---|---|
| alone | `pytest tests/test_wallet.py::test_withdraw_reduces_balance` |
| shuffled order (pytest-randomly) | `pytest -p randomly` |
| parallel (pytest-xdist) | `pytest -n auto` |
| many orders | `python scripts/flake_hunt.py --runs 30 -- tests/` |

Order dependence is always a defect, and always in the test, never in the runner. The usual
causes: module-level mutable state (rule 13), a too-broad fixture scope (see `fixtures.md`), a
real file at a fixed path, or a monkeypatch applied without teardown.

### Rule 12: No unseeded randomness

Audit code: `UNSEEDED_RANDOM`.

An unseeded random input means a failure you cannot reproduce, which means a failure you cannot
fix, which means a test everyone learns to re-run until it goes green. Seed centrally in
`conftest.py` (see `assets/conftest.py`) so this is handled once for the whole suite.

The audit is context-aware. It stays quiet if a `conftest.py` calls `random.seed(...)`,
`np.random.seed(...)`, `torch.manual_seed(...)`, or `default_rng(<seed>)`, and it always
allows an explicitly seeded local generator such as `random.Random(7)`. Setting
`PYTHONHASHSEED` does not count: inside a running test it only affects child processes,
and it seeds string hashing, not any RNG.

Property-based testing with Hypothesis is the principled exception: it generates inputs
deliberately, records the failing case, and shrinks it to a minimal reproducer. That is the
opposite of accidental randomness. Hypothesis is listed as a recommended tool by python-guide.org.

### Rule 13: No mutated module-level state

Audit code: `MUTABLE_MODULE_STATE`.

```python
SEEN = []                       # module level

def test_first():
    SEEN.append("a")
    assert len(SEEN) == 1       # passes alone, fails after test_zero runs
```

A module-level list or dict used as read-only parametrize data is fine, and the audit allows it.
It only reports when a test actually mutates the object.

### Rule 14: No network in a unit test

Audit code: `NETWORK`.

Real Python: *"Avoid randomness, real network calls, or calls to external services in your
tests."* A unit test that hits the network is slow, fails when the wifi does, and fails
differently when a rate limit kicks in.

`assets/conftest.py` blocks this at the socket layer, which also catches the calls made by
dependencies you did not know were making them. Tests that genuinely need a real service carry
`@pytest.mark.integration` and are deselected by default.

Loopback and Unix-domain sockets stay allowed by the conftest guard. The audit skips tests
marked `integration` or `e2e` and tests under an `integration/` or `e2e/` directory.

### Rule 15: No `time.sleep`

Audit code: `SLEEP`.

A sleep is either too short (flaky on a loaded CI runner) or too long (the suite gets slower
every time someone adds one). Inject a clock or use `monkeypatch`: `assets/conftest.py` provides
a `frozen_clock` fixture that lets a test advance time by an hour instantly.

`frozen_clock` patches the `time` module only. Code that did `from time import time`, and
`datetime.now()`, still see real time; inject a clock or use the time-machine package.
`await asyncio.sleep(0)` (yielding to the event loop) is allowed, and sleeps inside
fixtures are reported too.

### Rule 16: Filesystem writes go to `tmp_path`

`tmp_path` is a `pathlib.Path` unique to each test function, created and cleaned up
automatically. `tmp_path_factory` is the session-scoped version.

Use these, not `tmpdir`/`tmpdir_factory`: those return legacy `py.path.local` objects rather
than `pathlib.Path`, and are kept only for backward compatibility.

Failed tests keep their directories by default (`tmp_path_retention_policy = "failed"`), which is
often exactly what you want when debugging.

Source: pythontest.com.

### Rule 17: Environment changes go through `monkeypatch`

Audit code: `ENV_MUTATION`.

`os.environ["MODE"] = "test"` persists for the rest of the process. The test that breaks is some
unrelated one, later, and only when the suite runs in a particular order.

Verified `MonkeyPatch` API (pytest 9.2.0): all of it undone at teardown:

```
setenv(name: str, value: str, prepend: str | None = None) -> None
delenv(name: str, raising: bool = True) -> None
setattr(target, name, value, raising: bool = True) -> None
delattr(target, name, raising: bool = True) -> None
setitem(dic, name, value) -> None
delitem(dic, name, raising: bool = True) -> None
chdir(path) -> None
syspath_prepend(path) -> None
context() -> Generator[MonkeyPatch]     # for use outside a test function
```

### Rule 18: No `sys.path` manipulation

Audit code: `SYS_PATH`.

If your tests cannot import your package, the packaging is wrong. Fix that instead:

- Install the package editable: `pip install -e .`
- Use a `src/` layout so tests import the *installed* copy, not the working tree
- Set `pythonpath = ["src"]` in pytest config if you deliberately do not install
- Use `--import-mode=importlib`, which the pytest project recommends for new projects because it
  does not modify `sys.path` at all

A `sys.path.insert` in a test file makes the suite pass locally and fail in CI, or worse, test
the working tree while CI tests the built wheel.

Source: pytest docs, *Good Integration Practices*.

---

## Test doubles

### Rule 19: `mock.patch` always passes `autospec=True`

Audit code: `PATCH_NO_SPEC`.

A bare `Mock` accepts any call: any name, any arity, any keywords. The test passes. Then someone
renames the real method, the production call site breaks, and the test (which is mocking that
exact method) stays green.

```python
with mock.patch("myapp.service.fetch_rate", autospec=True) as fake:
    fake.return_value = 1.5
    ...
fake.assert_called_once_with("EUR")   # arity now checked against the real signature
```

`autospec=True` builds the double from the real object's signature, so a wrong-arity call raises
`TypeError` inside the test, where it is cheap to fix. `spec=` and `spec_set=` are narrower
variants; `spec_set` additionally forbids setting attributes the real object does not have.

### Rule 20: Patch where it is used, not where it is defined

```python
# myapp/service.py
from myapp.rates import fetch_rate

def convert(wallet, to): return wallet.balance * fetch_rate(to)
```

```python
mock.patch("myapp.service.fetch_rate", autospec=True)   # works
mock.patch("myapp.rates.fetch_rate", autospec=True)     # no effect
```

`from X import y` binds `y` into the importing module's namespace at import time. Patching the
original module rebinds a name nothing is looking at any more. This is the single most common
"my mock isn't working" cause.

### Rule 21: Never call a fake Mock assertion

Audit code: `FAKE_MOCK_ASSERT`.

`Mock` auto-creates any attribute you touch. So:

```python
m.called_once_with(3)          # returns a Mock. Truthy. Asserts nothing. Passes forever.
assert m.called_once           # same trap, in assert form
m.assert_called_once_wtih(3)   # typo. Auto-created. Passes forever.
```

The real names, all of which raise `AssertionError` when unmet:

`assert_called`, `assert_called_once`, `assert_called_with`, `assert_called_once_with`,
`assert_any_call`, `assert_has_calls`, `assert_not_called`, and the `assert_await*` equivalents.

`unittest.mock` mitigates the typo case for `assert_`-prefixed names on newer Pythons, but not
for the non-prefixed ones, and not at all on `MagicMock` attributes you construct yourself. The
audit checks all of them.

### Rule 22: Prefer real objects and `monkeypatch` over mocks

A mock asserts that your code called something in a particular way. It does not assert that the
call *worked*. A suite made mostly of mocks tests your assumptions about the dependency rather
than the dependency, and it breaks on every refactor while missing every real bug.

The order of preference:

1. **The real object**, if it is fast, pure, and in-process. Testing `Wallet` against a real
   `Wallet` is not a compromise.
2. **A fake**: a working in-memory implementation (a dict-backed repository, an in-memory
   SQLite). Real behavior, no I/O.
3. **`monkeypatch`** for a single function or attribute at the boundary you own.
4. **`mock.patch(autospec=True)`** when the *interaction* is the thing under test: "did the retry
   logic call the API exactly three times?"

Mock the boundary you own, not the library. Real Python's framing: use mocks for "databases,
APIs, or other slow or unpredictable systems", and complement unit tests with a smaller number
of integration and end-to-end tests that exercise the real thing.

---

## Markers and skips

### Rule 23: Register every marker

Config: `strict_markers = true` (or `strict = true`). Since pytest 9.2 this also validates
marker names in `-m` expressions: `pytest -m slwo` fails with "Unknown marker(s) in '-m'
expression" instead of silently selecting nothing (pytest 9.2.0 changelog, #2781).

Without it, `@pytest.mark.slwo` produces a warning that scrolls past, and that test is now
unmarked, uncategorized, and running in every default selection: or worse, never selected by
`-m slow` and effectively dead. With strict markers it is a hard error at collection.

Register in config (`markers = [...]`) with a description after the colon; `pytest --markers`
prints them. To mark every test in a module: `pytestmark = pytest.mark.slow` at module level, or
a list for several.

Source: pythontest.com; pytest docs.

### Rule 24: `skip`/`skipif`/`xfail` carry `reason=`

Audit code: `SKIP_NO_REASON`.

Run with `-ra` (in the recommended `addopts`) and the summary prints every non-passing test with
its reason. Without `reason=`, the report shows an `s` and nothing else, and a test that has been
skipped for two years looks identical to one skipped this morning. Put the issue number in the
reason.

### Rule 25: `xfail` is strict

Audit code: `XFAIL_NOT_STRICT`. Config: `strict_xfail = true` (alias `xfail_strict`).

Non-strict `xfail` reports `XPASS` when a known-broken test starts passing, and `XPASS` is not a
failure. So the bug gets fixed, nobody is told, the marker stays, and the test's real assertions
are never enforced again.

Strict mode turns that unexpected pass into a `FAILED` that means "the bug is fixed, delete this
marker." That is the notification you actually want.

`--runxfail` ignores `xfail` marks entirely; Okken suggests it for final pre-production runs, to
see what the suite would say with the excuses removed.
The audit reads the nearest `pyproject.toml`, `pytest.toml`, `pytest.ini`, `tox.ini`, or
`setup.cfg`. When `strict`, `strict_xfail`, or `xfail_strict` is true there, it does not
ask for `strict=True` on each marker.


---

## Coverage

### Rule 26: Branch coverage with an enforced floor

```toml
addopts = ["--cov", "--cov-branch", "--cov-fail-under=80"]
```

Line coverage counts a line as covered if it executed. Branch coverage counts *decision
outcomes*: `if x:` needs both the taken and not-taken path. Since the untested branch is where
bugs live, line coverage systematically overstates how tested you are.

Set the floor at or just below where the suite is now, and ratchet it up as you go. A floor you
have to lower to merge is worse than no floor: it teaches everyone that the number is
negotiable.

### Rule 27: Do not chase 100%

Real Python: *"Don't chase 100% coverage by adding unnecessary or low-value tests. Focus on tests
that protect important behavior."*

Coverage is a *negative* signal. It tells you what is definitely not tested. It says nothing
about whether what is tested is tested well: an assertion-free suite (rule 6) reaches 100% while
proving nothing. Use the report to find untested critical paths, then decide by hand whether each
one is worth a test.

Use `exclude_also` in coverage config for code that genuinely cannot run under test:
`if TYPE_CHECKING:`, `raise NotImplementedError`, abstract methods, `if __name__ == "__main__":`.

---

## Rules added in the October 2026 audit

### Rule 28: Never swallow a failure

Audit code: `SWALLOWED_FAILURE`.

```python
def test_total():
    try:
        assert total() == 3
    except AssertionError:
        pass
```

This test cannot fail. Neither can one that wraps the call in `except Exception:` or a
bare `except:` without re-raising or calling `pytest.fail`. Ruff's SIM105 suggests
rewriting the block as `contextlib.suppress(AssertionError)`, which makes the hidden
failure permanent, so `assets/ruff-pytest.toml` turns SIM105 off for tests. Catching a
specific, documented exception is fine and common in property tests:
`except AmountError: return`.

### Rule 29: A comparison must be asserted

Audit code: `BARE_COMPARISON`. Ruff: B015.

`result == 3` on its own line computes a boolean and throws it away. It is the most
common way an assertion silently disappears during an edit.

### Rule 30: Never compare with NaN using `==`

Audit code: `NAN_COMPARE`.

NaN is unequal to everything, itself included. `assert x != math.nan` always passes and
`assert x == float("nan")` always fails. Use `math.isnan(x)`, or
`pytest.approx(expected, nan_ok=True)`. The v0 wallet accepted `withdraw(nan)` because
NaN fails both `amount <= 0` and `amount > balance`; see `property-fuzz-symbolic.md`.

### Rule 31: At least one assertion checks a value

Audit code: `WEAK_ASSERT` (warning; fails the gate only with `--warnings-as-errors`).

A test whose only checks are `is not None`, `isinstance(...)`, `hasattr(...)`, or
`len(x) > 0` passes for almost any wrong answer. Keep the existence check if it helps
the failure message, and add one that pins the value.

## Rules added in the November 2026 audit

Verified against pytest 9.2.0 on Python 3.13 by running each snippet.

### Rule 32: No `yield_fixture`

Audit code: `YIELD_FIXTURE`.

Plain `@pytest.fixture` has accepted `yield` teardown since pytest 3.0 (pytest docs,
"yield_fixture" page). `yield_fixture` is an alias, deprecated in 6.2 (#7988), and pytest
9.2.0 schedules it for removal in pytest 10 or 11 (#14755). Under `filterwarnings =
["error"]` collection already fails with `PytestRemovedIn10Warning: @pytest.yield_fixture
is deprecated.`; the audit names it before anything runs.

### Rule 33: Exception groups are asserted with `pytest.RaisesGroup`

`asyncio.TaskGroup`, anyio task groups, and `except*` code raise `ExceptionGroup`, which
`pytest.raises(ValueError)` never matches. Two verified shapes:

```python
with pytest.RaisesGroup(pytest.RaisesExc(ValueError, match="^bad amount$"), KeyError):
    asyncio.run(run_both())

with pytest.raises(ExceptionGroup) as info:
    asyncio.run(run_both())
assert info.group_contains(ValueError, match="bad amount")
```

`RaisesGroup` requires every member to match and no extras: with only `ValueError`
expected, a group holding `ValueError` and `KeyError` fails with `Failed: Raised
exception group did not match: 1 matched exception. Unexpected exception(s):
[KeyError('missing')]`. Member order does not matter. `allow_unwrapped=True` also
accepts a bare exception; `flatten_subgroups=True` ignores nesting. A mismatch raises
`pytest.fail.Exception`, not `AssertionError`, so `pytest.raises(AssertionError)` around
it does not catch it. Prefer `RaisesGroup` when the full set of errors is the contract,
`group_contains` when only one member matters.

### Rule 34: A green run after the agent edited the tests is not evidence

A suite turned green by deleting a test, dropping assertions, adding a skip, or
deselecting in config proves nothing. `scripts/diff_guard.py` compares the test
tree with a git base and reports those edits; the Stop hook runs it. See
`llm-written-tests.md` for the codes and when an edit is legitimate.

## Audit code index

| Code | Severity | Rule |
|---|---|---|
| `NO_ASSERT` | error | 6 |
| `PRINT` | warning | 6 |
| `ALWAYS_TRUE` | error | 7 |
| `FLOAT_EQ` | error | 8 |
| `BROAD_RAISES` | error | 9 |
| `BRANCHING` | error | 4 |
| `SHADOWED_TEST_NAME` | error | 5 |
| `UNITTEST_CLASS` | error | 3 (plain functions get fixtures and assertion rewriting) |
| `UNSEEDED_RANDOM` | error | 12 |
| `MUTABLE_MODULE_STATE` | error | 13 |
| `NETWORK` | error | 14 |
| `SLEEP` | error | 15 |
| `ENV_MUTATION` | error | 17 |
| `SYS_PATH` | error | 18 |
| `PATCH_NO_SPEC` | error | 19 |
| `FAKE_MOCK_ASSERT` | error | 21 |
| `SKIP_NO_REASON` | error | 24 |
| `XFAIL_NOT_STRICT` | error | 25 |
| `SWALLOWED_FAILURE` | error | 28 |
| `BARE_COMPARISON` | error | 29 |
| `NAN_COMPARE` | error | 30 |
| `WEAK_ASSERT` | warning | 31 |
| `YIELD_FIXTURE` | error | 32 |
| `SUPPRESSION_NO_REASON` | error | see below |

`python scripts/audit_tests.py --list-codes` prints the same list.

## When to break a rule

The rules are defaults with reasons, and a reason that does not apply is not a reason. Break one
when you can state which rule, why here, and what you did instead: in a comment next to the code.

Legitimate exceptions that come up often:

| Rule | Exception |
|---|---|
| 12 (randomness) | Hypothesis property-based tests generate inputs on purpose and shrink failures to a minimal reproducer. |
| 14 (network) | Contract tests against a real staging service, marked `integration`, run on a schedule rather than per-commit. |
| 10 (exact values) | A learned metric, a timing bound, or a genuinely probabilistic property: assert a range and say why. |
| 4 (no loops) | A state-machine test that must apply a sequence of transitions. Parametrize the sequences, loop inside. |
| 8 (float `==`) | Comparing to a value that is exactly representable *and* produced by exact integer arithmetic. Rare enough that approx costs nothing. |

Suppress at the narrowest scope that covers the decision, and always give the reason
after the bracket. A pragma without a reason is itself a finding, `SUPPRESSION_NO_REASON`.

| Scope | Syntax |
|---|---|
| one line | `assert x == 0.5  # audit: ignore[FLOAT_EQ] exact binary fraction` |
| one test (on the `def` line) | `def test_x():  # audit: ignore[NO_ASSERT] smoke test: must not raise` |
| one file | `# audit: ignore-file[PRINT] demo module` |
| whole run | `python scripts/audit_tests.py tests/ --ignore BRANCHING` |

---

## Source list

- **pytest documentation**, *Good Integration Practices*: test layout, `src` layout, import
  modes, strict mode. https://docs.pytest.org/en/stable/explanation/goodpractices.html
- **pytest documentation**, *API Reference*: configuration options.
  https://docs.pytest.org/en/stable/reference/reference.html
- **The Hitchhiker's Guide to Python**, *Testing Your Code*: the eleven general rules on focus,
  independence, speed, naming, and bug-driven tests. https://docs.python-guide.org/writing/tests/
- **Real Python**, *code testing* best practices: small focused tests, fixtures, parametrize,
  determinism, mocking boundaries, reasonable coverage.
  https://realpython.com/ref/best-practices/code-testing/
- **Brian Okken** (author, *Python Testing with pytest*), *pytest tips and tricks*: AAA
  structure, fixture scopes, `--strict-markers`/`--strict-config`/`-ra`, `xfail_strict`, built-in
  fixtures, debugging flags. https://pythontest.com/pytest-tips-tricks/
- **Ruff**, `flake8-pytest-style` (PT) rule reference.
  https://docs.astral.sh/ruff/rules/#flake8-pytest-style-pt
- **Gruber et al.**, *An Empirical Study of Flaky Tests in Python*, ICST 2021: order
  dependence causes 59% of flaky tests. https://arxiv.org/pdf/2101.09077
- **Inozemtseva and Holmes** (ICSE 2014) and **Just et al.** (FSE 2014): coverage versus
  mutation score as effectiveness signals. See `mutation-testing.md`.
- **PyTorch**, *Reproducibility*: seeding, `use_deterministic_algorithms`, cuDNN flags,
  DataLoader worker seeding. https://docs.pytorch.org/docs/stable/notes/randomness.html
