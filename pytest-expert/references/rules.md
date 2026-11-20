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
- [Diff guard code index](#diff-guard-code-index-rule-34)
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
`items = [make(i) for i in range(3)]` in the Arrange block. The audit flags `if`, `while`,
`match`, and `for` or `async for` loops whose iterable is not a call (a literal or a name:
the case-driving shapes). `for i in range(3)` and comprehensions are not flagged.

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
`pytest.warns`, `pytest.fail`, and the real `Mock.assert_*` methods, but only where they can
run: a bare `pytest.raises(ValueError)` with no `with` block and no callable checks nothing, and
an `assert` after a `return` or `raise`, or inside a nested function the test never calls,
never executes. Each of those tests is reported as `NO_ASSERT`.

This one is common in AI-generated and hurried tests, and it is the reason coverage numbers lie:
an assertion-free suite can reach 100% line coverage while proving nothing.

A bare `x == pytest.approx(y)` without `assert` does not count as an assertion; the old
audit counted it. For a deliberate smoke test, suppress on the `def` line with a reason:
`def test_x():  # audit: ignore[NO_ASSERT] smoke test: must not raise`.

When the suite asserts through its own helper (`check_total(result)`), name the helper instead of
suppressing every test. `--assert-helper PATTERN` (repeatable) and `assert_helpers = [...]` under
`[tool.pytest-expert]` in `pyproject.toml` (read on Python 3.11 or later) make a live call to a
matching function count as an assertion for `NO_ASSERT` and for nothing else: `WEAK_ASSERT` and
every other rule see the test as before. Patterns are case-sensitive `fnmatch` globs. One without a
dot matches the final name of the callee (`check_*` matches `check_x(r)`, `self.check_x(r)` and
`helpers.check_x(r)`); one with a dot matches the whole dotted callee, as written or with the import
alias resolved (`helpers.expect_*` matches `h.expect_ok(r)` after `import helpers as h`). The call
must be able to run, by the same rules as `assert`: not after an unconditional `return` or `raise`,
not in a nested function nothing calls, not in a lambda. The command line adds to the configured
list. A pattern that would match every call (`*`, `?`, `*.*`, `[a-z]`) or an empty one is rejected
with exit 2, as is an unknown key or a non-list value under `[tool.pytest-expert]`. The patterns in
effect are echoed in the text summary (`NO_ASSERT relaxed by assert helpers: check_*`), the SARIF
run properties (`assertHelpers`) and a GitHub `::notice`; `--format json` prints
`{"findings": [...], "assert_helpers": [...]}` when helpers are in effect and a plain list of
findings otherwise.

`print()` in a test (`PRINT`, warning) is the same failure in another shape: it shows a
value instead of checking it. If output is the behavior, capture it with `capsys`.

### Rule 7: No assertion that cannot fail

Audit code: `ALWAYS_TRUE`.

`assert True`, `assert 1`, `assert "the balance should update"`: the last one is the sneaky
version, because it reads like documentation and passes because non-empty strings are truthy.

The tuple form is the subtle one: `assert (result == 3, "msg")` asserts a non-empty
tuple, which is always truthy, so the comparison inside never matters. The audit also reports
`assert x or True`, `assert 1 == 1`, and any other condition whose value is fixed by its literal
parts.

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
`\.` for a decimal point, or wrap in `re.escape()`. The audit reports `Exception` and
`BaseException` with no `match=`, or with a `match=` that matches any message (`""`, `".*"`,
`"^.*$"`, `".*?"`, `"(?s).*"`, or `None`).

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
| many orders | `python scripts/flake_hunt.py --runs 31 -- tests/` |

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
allows an explicitly seeded local generator such as `random.Random(7)` or
`random.Random(seed=7)`. The conftest lookup walks up from the test file and stops at the
project root, the nearest directory holding a pytest config file or `.git`. Setting
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

Loopback and Unix-domain sockets stay allowed by the conftest guard. The audit reports a call
that opens a connection or resolves a host (`requests.get`, `httpx.Client`,
`urllib.request.urlopen`, `socket.socket`, `socket.create_connection`,
`aiohttp.ClientSession`, `http.client.HTTPConnection`, and similar), not any other name from
those libraries. It skips tests marked `integration` or `e2e` (on the function, its class, or a
module-level `pytestmark`) and tests under an `integration/` or `e2e/` directory inside the
audited project.

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
unrelated one, later, and only when the suite runs in a particular order. The audit reports
assignment and deletion on `os.environ` and calls such as `os.environ.update`, `.setdefault`,
`.pop`, `.clear`, `os.putenv` and `os.unsetenv`; code inside `with mock.patch.dict(os.environ, ...)`
is exempt, because that restores the environment on exit.

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
the working tree while CI tests the built wheel. The audit reports `insert`, `append`, `extend`,
`remove`, `pop` and `clear` on `sys.path`, and assignment to it.

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

A mock assertion that is not called asserts nothing, and some misspelt ones are silent too:

```python
m.assert_called_once           # a bound method, never called: truthy, asserts nothing, on every Python
m.called_once_with(3)          # Python < 3.12: returns a truthy Mock, asserts nothing
assert m.called_once           # same, in assert form
m.assert_called_once_wtih(3)   # typo: raises AttributeError on a default Mock, passes on Mock(unsafe=True)
```

The first form is the live trap: no Python version warns about it. For the others, Python 3.12
added `AttributeError` for `called_once`, `called_once_with`, `called_with`, `not_called` and
`has_calls` on a default `Mock`, and `assert_`-prefixed misspellings have raised since 3.5. Checked
2026-11-20 with `unittest.mock.Mock()` on Python 3.11.14, 3.12.16 and 3.13.2: the `called_once*`
names return a Mock on 3.11 and raise on 3.12 and 3.13; `assert_called_once_wtih` raises on all
three. The audit still reports them, because the suite may run on 3.11 or older, and because
`Mock(unsafe=True)` and attributes of other objects accept any name.

The real names, all of which raise `AssertionError` when unmet:

`assert_called`, `assert_called_once`, `assert_called_with`, `assert_called_once_with`,
`assert_any_call`, `assert_has_calls`, `assert_not_called`, and the `assert_await*` equivalents.
Call them with parentheses.

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

Audit code: `SKIP_NO_REASON`. It covers marks on a function, a class, and a module-level
`pytestmark`, and an empty `reason=""`.

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
The audit reads the pytest config the way pytest 9.2.0 picks it: from the audited directory up
to the project root, the first of `pytest.toml`, `.pytest.toml`, `pytest.ini`, `.pytest.ini`,
`pyproject.toml` (only if it has a `[tool.pytest]` table), `tox.ini` or `setup.cfg` that carries
pytest settings. `strict_xfail` (alias `xfail_strict`) decides if it is set, otherwise `strict`.
When that is true, the audit does not ask for `strict=True` on each marker.


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
bare `except:` without re-raising or calling `pytest.fail`, and neither can a
`with contextlib.suppress(AssertionError):` block. Ruff's SIM105 suggests
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

26 codes. The text in the last column is what `python scripts/audit_tests.py --list-codes` prints.

| Code | Severity | Rule | What it reports |
|---|---|---|---|
| `NO_ASSERT` | error | 6 | test contains no reachable assert, pytest.raises/warns/fail that runs, or real mock assertion |
| `PRINT` | warning | 6 | print() in a test; assert instead, or use capsys/caplog |
| `ALWAYS_TRUE` | error | 7 | assertion can never fail (truthy literal, `x or True`, 1 == 1, non-empty tuple/list/dict/set) |
| `FLOAT_EQ` | error | 8 | float value compared with == or != (or `in` a float list); use pytest.approx |
| `BROAD_RAISES` | error | 9 | pytest.raises on Exception/BaseException with no match= (or a match that matches anything); passes on the wrong error |
| `BRANCHING` | error | 4 | if/while or for-over-cases in a test body; use parametrize |
| `SHADOWED_TEST_NAME` | error | 5 | two tests with the same name in one scope; the first never runs |
| `UNITTEST_CLASS` | error | 3 (plain functions get fixtures and assertion rewriting) | unittest.TestCase subclass; plain functions get fixtures and assert rewriting |
| `UNSEEDED_RANDOM` | error | 12 | unseeded randomness and no seeding conftest; failures become unreproducible |
| `MUTABLE_MODULE_STATE` | error | 13 | module-level mutable is mutated by tests; state leaks between tests |
| `NETWORK` | error | 14 | network call (requests/httpx/urllib/socket/aiohttp/http.client) in a test without an integration or e2e marker or directory |
| `SLEEP` | error | 15 | time.sleep in a test or fixture; slow and flaky, inject a clock |
| `ENV_MUTATION` | error | 17 | os.environ mutated directly; use monkeypatch.setenv/delenv |
| `SYS_PATH` | error | 18 | sys.path mutated in a test; fix packaging or use monkeypatch.syspath_prepend |
| `PATCH_NO_SPEC` | error | 19 | mock.patch / patch.object without autospec/spec/new; the double drifts from the API |
| `FAKE_MOCK_ASSERT` | error | 21 | mock assertion that cannot fail: an assert_* method named without parentheses, or a misspelt assert_* / called_once / not_called name (those only pass silently on Mock(unsafe=True), non-Mock objects, and called_once-style names on Python < 3.12; a default Mock raises AttributeError) |
| `SKIP_NO_REASON` | error | 24 | skip/skipif/xfail without a reason (or an empty one); the report will not say why |
| `XFAIL_NOT_STRICT` | error | 25 | xfail with no strict= (or strict=False) while the pytest config does not enable strict_xfail |
| `SWALLOWED_FAILURE` | error | 28 | except AssertionError/Exception/bare, or contextlib.suppress of them, without re-raise; failures vanish |
| `BARE_COMPARISON` | error | 29 | comparison used as a statement; it is computed and thrown away, add assert |
| `NAN_COMPARE` | error | 30 | == / != against NaN is always False / True; use math.isnan or nan_ok |
| `WEAK_ASSERT` | warning | 31 | every assertion is existence-only (is not None, isinstance, len > 0) |
| `YIELD_FIXTURE` | error | 32 | pytest.yield_fixture is deprecated; pytest 9.2 schedules removal for pytest 10 or 11; use @pytest.fixture |
| `SUPPRESSION_NO_REASON` | error | see below | audit: ignore[...] pragma without a reason after the bracket |
| `SUPPRESSION_INERT` | warning | see below | audit: ignore[...] pragma that suppresses nothing: unknown, misspelt, lower-case or empty code, ALL, or no matching finding in its line, scope or file (stale) |
| `SYNTAX` | error | none: a file that does not parse is reported, not skipped | file does not parse |

`audit_tests.py` exits 0 when it checked and found nothing, 1 when an error-severity finding
remains (or any warning, with `--warnings-as-errors`), and 2 when it could not check: bad
arguments, an unknown `--ignore` code, no test files under the path, an unreadable file or
config, or an internal error. Exit 2 is never a pass. A clean run prints `clean: no findings
(N file(s) checked)`.

## Diff guard code index (rule 34)

19 codes from `python scripts/diff_guard.py --list-codes`; `llm-written-tests.md` says when
an edit is legitimate. Exit 0 = nothing found, 1 = an error-severity finding (or any warning
with `--warnings-as-errors`), 2 = could not check (nothing to compare, an unreadable or
unparsable config, git missing, an unknown base, an internal error).

| Code | Severity | What it reports |
|---|---|---|
| `TEST_FILE_DELETED` | error | a test file present in the base is gone (a pure rename or move is followed, not reported) |
| `TEST_DELETED` | error | a test function present in the base is gone from every test file |
| `TEST_EMPTIED` | error | test body reduced to pass, ..., or an early return |
| `ASSERTS_REDUCED` | error | test has fewer live checks (assert, pytest.raises/warns, assert*/self.assert* calls) than in the base; dead code does not count |
| `ASSERT_WEAKENED` | error | test kept its checks but lost exact comparisons (== loosened to truthiness, is not None, in, <), or pytest.raises widened to Exception / lost match= |
| `CASES_REDUCED` | error | a parametrize list lost rows (a failing case can be dropped this way) |
| `ASSERT_TRIVIALIZED` | error | test gained an assertion that cannot fail (assert True, assert x == x, x or True, approx with a huge tolerance) |
| `SKIP_ADDED` | error | skip, skipif, xfail, pytest.skip(), importorskip or a unittest skip added to a test, class, module or conftest |
| `PRAGMA_ADDED` | error | new '# audit: ignore' pragma silences the test audit |
| `COLLECTION_NARROWED` | error | conftest gained collect_ignore entries or a collection hook, or changed the body of one |
| `CONFIG_NARROWED` | error | pytest config gained --deselect/--ignore/-k/-m/--collect-only/--no-cov/..., or lost testpaths, python_files/classes/functions patterns |
| `COVERAGE_LOWERED` | error | effective coverage floor (coverage fail_under or --cov-fail-under, in any config file) lowered or removed (a floor the base never had is not checked), or coverage omit gained entries |
| `SYNTAX` | error | a .py file in the test tree (or a changed source file) cannot be decoded or parsed, so it is not compared; a collected test file or conftest then fails to collect, a helper module fails when imported |
| `EXPECTATION_CHANGED` | warning | a literal on the expected side of an assert, or a parametrize row, changed; confirm the spec changed |
| `CONSTANT_EQ` | warning | source gained __eq__/__ne__/__hash__ returning a constant: makes any == assertion pass |
| `SPECIAL_CASE` | warning | source gained a comparison with a literal that also appears in the tests: possible special-casing |
| `SNAPSHOT_CHANGED` | warning | a stored snapshot (syrupy __snapshots__/*.ambr, or any file under __snapshots__) was rewritten or deleted; review it like code |
| `PARAM_HOOK_CHANGED` | warning | a pytest_generate_tests hook was added or its body changed (it decides which parameter sets run); confirm no cases were dropped |
| `SKIP_VIA_HELPER` | warning | a test gained a call to (or a request for) a helper or fixture that can reach pytest.skip/xfail/importorskip, or such a helper gained the skip; confirm the skip condition is real |

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
A pragma that suppresses nothing is `SUPPRESSION_INERT` (warning): an unknown or misspelt code (the
nearest valid one is suggested), empty brackets, a lower-case code, `ALL`, a code pragmas cannot
suppress, or a well-formed pragma with no matching finding in its line, scope or file. The last
kind is a stale pragma: the finding went away and the pragma now hides the next one in its scope.
A pragma cannot suppress `SUPPRESSION_INERT` or `SUPPRESSION_NO_REASON`; `--ignore` removes either.

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
