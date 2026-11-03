# Parametrize and test doubles

Two topics that account for most of the leverage (and most of the silent failures) in a pytest
suite. Verified against pytest 9.2.0.

## Contents

- [Parametrize](#parametrize)
  - [Readable IDs](#readable-ids)
  - [Marking individual cases](#marking-individual-cases)
  - [Stacking and indirect](#stacking-and-indirect)
  - [When not to parametrize](#when-not-to-parametrize)
- [Test doubles](#test-doubles)
  - [The decision tree](#the-decision-tree)
  - [monkeypatch](#monkeypatch)
  - [mock.patch and autospec](#mockpatch-and-autospec)
  - [Patch where it is used](#patch-where-it-is-used)
  - [The fake-assertion trap](#the-fake-assertion-trap)
  - [Fakes](#fakes)
- [Property-based testing](#property-based-testing)
- [Snapshot testing](#snapshot-testing)

---

# Parametrize

`@pytest.mark.parametrize` runs one test function once per input tuple, reporting each as a
separate test. It is the answer to rule 4 (no loops in test bodies) and to the copy-paste-modify
urge: when you find yourself duplicating a test to change one value, parametrize it.

```python
@pytest.mark.parametrize(
    ("starting", "withdrawal", "expected"),
    [
        (100.0, 100.0, 0.0),
        (100.0, 0.01, 99.99),
        (100.0, 99.99, 0.01),
    ],
)
def test_withdraw_boundaries(starting, withdrawal, expected):
    wallet = Wallet(balance=starting)
    assert wallet.withdraw(withdrawal) == pytest.approx(expected)
```

Three separate tests. One failing does not stop the others, each appears in the report by name,
and `pytest -k` can select one.

The argnames form matters for consistency. A tuple `("a", "b")` is what
`assets/ruff-pytest.toml` requires (`parametrize-names-type = "tuple"`, ruff PT006); the
comma-string form `"a, b"` also works but mixing the two across a codebase makes grep useless.
A single argument takes a bare string: `@pytest.mark.parametrize("amount", [1, 2, 3])`.

## Readable IDs

Without `ids`, pytest builds them from the values:

```
test_withdraw_boundaries[100.0-0.01-99.99]
```

That tells you the inputs but not the intent, and it is useless for non-primitive values, which
render as `wallet0`, `wallet1`. Three ways to fix it:

```python
# 1. explicit list, one per case
@pytest.mark.parametrize(
    ("starting", "withdrawal", "expected"),
    [(100.0, 100.0, 0.0), (100.0, 0.01, 99.99)],
    ids=["withdraw-all", "withdraw-minimum"],
)

# 2. per-case, with pytest.param
@pytest.mark.parametrize(
    ("starting", "withdrawal", "expected"),
    [
        pytest.param(100.0, 100.0, 0.0, id="withdraw-all"),
        pytest.param(100.0, 0.01, 99.99, id="withdraw-minimum"),
    ],
)

# 3. a function, when the values can name themselves
@pytest.mark.parametrize("wallet", [Wallet(100.0), Wallet(0.0)], ids=lambda w: f"balance-{w.balance}")
```

Under `strict = true`, `strict_parametrization_ids` makes duplicate IDs an error instead of
silently appending `0`, `1`. That is what you want: two cases with the same ID usually means one
is an accidental duplicate (ruff PT014 catches the exact-duplicate case too).

## Marking individual cases

```python
@pytest.mark.parametrize(
    "amount",
    [
        1.0,
        pytest.param(1e12, marks=pytest.mark.slow),
        pytest.param(float("nan"), marks=pytest.mark.xfail(strict=True, reason="NaN handling, #88")),
    ],
)
def test_deposit(amount):
    ...
```

One known-broken case does not require excluding the whole test. Note `strict=True` on the xfail
so the marker gets removed when the bug is fixed (rule 25).

## Stacking and indirect

Stacked decorators produce the Cartesian product:

```python
@pytest.mark.parametrize("currency", ["USD", "EUR"])
@pytest.mark.parametrize("amount", [1.0, 100.0])
def test_convert(currency, amount):    # 4 tests
    ...
```

Useful, and easy to detonate: three stacked decorators of four values each is 64 tests. If most
combinations are meaningless, write the meaningful ones as an explicit list of tuples instead.

`indirect=True` routes the parameter through a fixture rather than into the test, so the fixture
does the per-case setup:

```python
@pytest.fixture
def wallet(request):
    return Wallet(balance=request.param)      # request.param is the value

@pytest.mark.parametrize("wallet", [0.0, 100.0], indirect=True)
def test_balance(wallet):
    assert wallet.balance >= 0
```

Reach for this when a case needs *construction* work, not just a different literal.

## When not to parametrize

- **The cases need different assertions.** Parametrizing the expected exception type alongside
  the expected value produces a test with `if` in it, which is rule 4 again. Write two tests.
- **One case is conceptually different.** `test_withdraw_succeeds` and `test_withdraw_rejects_overdraft`
  are different behaviors (rule 1), not two rows of one table.
- **The parameter list encodes the implementation.** If adding a branch to the code requires
  adding a row, the test is tracking structure rather than behavior.

---

# Test doubles

A **test double** is any stand-in for a real dependency. Three kinds, and the distinction is
worth keeping straight:

- **Stub**: returns canned values. No assertions about how it was called.
- **Fake**: a working implementation with a shortcut. An in-memory dict standing in for a
  repository; SQLite standing in for Postgres.
- **Mock**: records how it was called so the test can assert on the interaction.

## The decision tree

```
Is the dependency fast, pure, and in-process?
├── yes → use the real thing. Not a compromise; it is the strongest test available.
└── no → does the test care HOW it was called, or only what it returned?
    ├── only the return value
    │   ├── a working in-memory version exists or is cheap to write → fake
    │   └── otherwise → monkeypatch.setattr with a stub function
    └── the CALL is the behavior under test
        ("does the retry logic call the API exactly three times?")
        └── mock.patch(..., autospec=True) + assert_called_once_with / assert_has_calls
```

The bias runs left: every step toward a mock trades real coverage for coupling to your
assumptions. A suite that is mostly mocks tests your beliefs about the dependency rather than
the dependency, breaks on every refactor, and misses every real bug. Real Python's framing is
narrow on purpose: mocks are for "databases, APIs, or other slow or unpredictable systems",
paired with a smaller number of integration tests that exercise the real thing.

## monkeypatch

First choice for replacing a single function, attribute, dict entry, or env var. No mock library,
no spec to get wrong, and everything is undone at teardown.

```python
def test_convert_uses_the_fetched_rate(funded_wallet, monkeypatch):
    monkeypatch.setattr("myapp.service.fetch_rate", lambda _currency: 1.5)
    assert convert(funded_wallet, "EUR") == pytest.approx(150.0)
```

Full verified API (pytest 9.2.0):

```
setattr(target, name, value, raising=True)     # also accepts "module.path.attr" as a single string
delattr(target, name, raising=True)
setitem(dic, name, value)
delitem(dic, name, raising=True)
setenv(name, value, prepend=None)
delenv(name, raising=True)
chdir(path)
syspath_prepend(path)
context()                                      # context manager, for use outside a test function
```

`raising=True` means patching something that does not exist is an error. Leave it on: it catches
the typo'd target that would otherwise make the patch a silent no-op.

`monkeypatch.context()` is for narrowing the window:

```python
with monkeypatch.context() as m:
    m.setenv("MODE", "test")
    run_thing()
# undone here, not at test teardown
```

## mock.patch and autospec

When the interaction is what you are testing:

```python
from unittest import mock

def test_retries_three_times():
    with mock.patch("myapp.service.fetch_rate", autospec=True) as fake_rate:
        fake_rate.side_effect = [TimeoutError, TimeoutError, 1.5]
        assert convert_with_retry(wallet, "EUR") == pytest.approx(150.0)
    assert fake_rate.call_count == 3
    fake_rate.assert_has_calls([mock.call("EUR")] * 3)
```

**`autospec=True` is not optional** (rule 19, audit code `PATCH_NO_SPEC`). A bare `Mock` accepts
any call (any name, any arity, any keywords) and returns another Mock. So a test that mocks
`fetch_rate(currency)` keeps passing after someone renames the parameter, changes the arity, or
deletes the function entirely. `autospec` builds the double from the real signature, so the
wrong-arity call raises `TypeError` inside the test where it is cheap to fix.

```python
with mock.patch("myapp.service.fetch_rate", autospec=True) as fake:
    fake("USD", "extra")     # TypeError: too many positional arguments
```

Related knobs: `spec=SomeClass` restricts attribute *access* to what the class has;
`spec_set=SomeClass` additionally forbids *setting* attributes the class does not have.
`mock.patch.object(obj, "method", autospec=True)` when you have the object rather than its path.
`mock.patch.dict(d, {...})` for dictionaries: that one needs no spec.

Useful configuration:

| Attribute | Effect |
|---|---|
| `return_value` | What every call returns |
| `side_effect = [a, b, c]` | Successive calls return each in turn |
| `side_effect = SomeError` | Every call raises |
| `side_effect = fn` | Every call delegates to `fn` |

`pytest-mock` provides the same thing as a `mocker` fixture with automatic teardown:
`mocker.patch("...", autospec=True)`. Worth adding; the rules are identical.

## Patch where it is used

```python
# myapp/service.py
from myapp.rates import fetch_rate

def convert(wallet, to):
    return wallet.balance * fetch_rate(to)
```

```python
mock.patch("myapp.service.fetch_rate", autospec=True)   # works
mock.patch("myapp.rates.fetch_rate", autospec=True)     # no effect
```

`from X import y` copies the reference into the importing module's namespace at import time.
Patching the original module rebinds a name that `convert` is no longer looking at. This is the
single most common cause of "my mock isn't working": and note the failure mode: the test does
not error, it silently uses the real function and hits the network.

The `assets/conftest.py` network blocker turns that silent failure into a loud one.

## The fake-assertion trap

Audit code: `FAKE_MOCK_ASSERT`. This is the highest-severity trap in the whole skill because the
test looks correct and passes forever.

`Mock` auto-creates any attribute you touch. So all three of these assert nothing:

```python
m.called_once_with(3)          # not a real method. Returns a Mock. Truthy.
assert m.called_once           # not a real attribute. A Mock. Truthy.
m.assert_called_once_wtih(3)   # typo. Auto-created. Called. Returns a Mock. Passes.
```

The real methods, every one of which raises `AssertionError` when unmet:

```
assert_called            assert_awaited
assert_called_once       assert_awaited_once
assert_called_with       assert_awaited_with
assert_called_once_with  assert_awaited_once_with
assert_any_call          assert_any_await
assert_has_calls         assert_has_awaits
assert_not_called        assert_not_awaited
```

Modern `unittest.mock` raises `AttributeError` for unknown *`assert_`-prefixed* names, which
catches `assert_called_onc`. It does not catch `called_once_with`, and it does not help on a
`MagicMock` you built by hand with those attributes set. `scripts/audit_tests.py` checks all of
them, in both call and attribute form.

Two real attributes that are *not* assertions, and read like they are: `m.called` (bool) and
`m.call_count` (int). `assert m.called` is a valid assertion; `assert m.called_once` is not.

## Fakes

Often the best option and the most overlooked. Real behavior, no I/O, no coupling to call
sequences:

```python
class InMemoryWalletRepo:
    """Same interface as the real repo, dict instead of Postgres."""
    def __init__(self):
        self._rows: dict[str, Wallet] = {}

    def save(self, key: str, wallet: Wallet) -> None:
        self._rows[key] = wallet

    def load(self, key: str) -> Wallet:
        if key not in self._rows:
            raise KeyError(key)
        return self._rows[key]


@pytest.fixture
def repo():
    return InMemoryWalletRepo()
```

A fake survives refactors that break every mock, because it implements the contract rather than
recording the calls. Define the interface as a `typing.Protocol` and the type checker will tell
you when the fake drifts from the real thing.

---

# Property-based testing

Hypothesis generates inputs, finds a failing one, shrinks it to the smallest
reproducer, and replays it on later runs. It is the principled exception to rule 12:
the randomness is deliberate, recorded, and reproducible.

```python
@given(amount=st.floats(min_value=0.01, max_value=1e9))
@example(amount=math.nan)
@example(amount=math.inf)
def test_withdraw_never_leaves_a_non_finite_balance(amount):
    wallet = Wallet(balance=100.0)
    try:
        wallet.withdraw(amount)
    except (ValueError, InsufficientFunds):
        return
    assert math.isfinite(wallet.balance)
```

The earlier version of this example generated floats with `allow_nan=False`. The
reference `Wallet` in `assets/test_example.py` accepts `withdraw(nan)` and corrupts
its balance, so excluding NaN hid exactly the bug the property should catch. Known
special values belong in `@example`, which runs every time; in measured runs
`st.floats()` produced NaN only about 1% of the time.

Patterns, CI profiles, solver-backed search (CrossHair), and fuzzing (Atheris) are in
`property-fuzz-symbolic.md`.

---

# Snapshot testing

A **snapshot test** compares output with a stored copy of a previous output instead
of a hand-written expected value. It suits large, stable structures (rendered
reports, serialized API responses, CLI output) where writing the expected value by
hand would be error-prone. Its weakness is the mirror image: the first stored value is
whatever the code produced, right or wrong, and one flag rewrites every stored value
to match the current output.

Two tools, both verified on pytest 9.2.0 (syrupy 6.1.1, inline-snapshot 0.36.1):

| | syrupy | inline-snapshot |
|---|---|---|
| Stored where | `tests/__snapshots__/<module>.ambr`, a separate file | in the test source: `assert x == snapshot({...})` |
| Use | `def test_x(snapshot): assert render() == snapshot` (a fixture) | `from inline_snapshot import snapshot` |
| First run, nothing stored | test fails (`AssertionError`) | test errors and prints the proposed source diff |
| Create | `pytest --snapshot-update` | `pytest --inline-snapshot=create` |
| Rewrite after a change | `pytest --snapshot-update` | `pytest --inline-snapshot=fix` |

The measured hazard: after changing `render_invoice` to add 1 to the total, both tests
failed; one run with `--snapshot-update --inline-snapshot=fix` rewrote the `.ambr`
file and the test source to `'total_cents': 251`, and the next run was `2 passed`.
Nothing else noticed. Rules that follow:

1. Review a snapshot diff the way you review code. A rewritten snapshot is a changed
   expectation; the commit message should say which behavior changed and why.
2. Never run the update flags to turn a red suite green. An agent doing this is the
   snapshot form of editing the expected value. `scripts/diff_guard.py` reports a
   rewritten `.ambr` as `SNAPSHOT_CHANGED` and a changed `snapshot(<literal>)` as
   `EXPECTATION_CHANGED` (both warnings, shown to the reviewer).
3. Keep snapshots small and specific: snapshot the one structure the test is about,
   not a whole page, so a diff is readable and an unrelated change does not force a
   blanket update.
4. Prefer an explicit assertion when the expected value is short. `assert total == 250`
   states intent; a snapshot only states "same as last time".

