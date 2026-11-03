# Fixtures

A **fixture** is a function decorated `@pytest.fixture` whose return value is injected into any
test that names it as a parameter. It replaces setup/teardown boilerplate, and it is the main
mechanism by which a pytest suite stays DRY without becoming order-dependent.

Verified against pytest 9.2.0.

## Contents

- [The mechanism](#the-mechanism)
- [Scope: the most consequential choice](#scope-the-most-consequential-choice)
- [Teardown](#teardown)
- [conftest.py layering](#conftestpy-layering)
- [Built-in fixtures](#built-in-fixtures)
- [Factory fixtures](#factory-fixtures)
- [Parametrized fixtures](#parametrized-fixtures)
- [autouse](#autouse)
- [The request object](#the-request-object)
- [Registering fixtures from code (pytest 9.1+)](#registering-fixtures-from-code-pytest-91)
- [Fixture anti-patterns](#fixture-anti-patterns)

## The mechanism

```python
import pytest

@pytest.fixture
def wallet():
    return Wallet(balance=100.0)

def test_withdraw(wallet):        # name match -> pytest calls wallet() and injects the result
    assert wallet.withdraw(30) == 70.0
```

The parameter name is the lookup key. Nothing is imported; pytest resolves the name by searching
the test module, then each `conftest.py` from the test's directory upward, then installed plugins.

Verified signature:

```
pytest.fixture(
    fixture_function=None, *,
    scope='function',      # 'function' | 'class' | 'module' | 'package' | 'session' | callable
    params=None,           # iterable -> one fixture instance per value
    autouse=False,         # run for every test in range without being requested
    ids=None,              # names for the params in the report
    name=None,             # rename the fixture, decoupling it from the function name
)
```

Fixtures compose by requesting each other:

```python
@pytest.fixture
def wallet():
    return Wallet()

@pytest.fixture
def funded_wallet(wallet):        # requests the one above
    wallet.deposit(100.0)
    return wallet
```

Build the derived case through the production API (`deposit`) rather than setting state by hand.
A bug in `deposit` then surfaces in the fixture instead of being papered over.

Discover what is available with `pytest --fixtures`. It prints every fixture visible at a given
path, its scope, and the file that defines it, which is how you find the one you are accidentally
shadowing.

## Scope: the most consequential choice

Scope controls how often the fixture runs and how long its value lives.

| Scope | Created once per | Use for |
|---|---|---|
| `function` (default) | test | Anything mutable. This is almost always right. |
| `class` | test class | Shared read-only setup for one class of tests. |
| `module` | test file | An expensive read-only resource used by one file. |
| `package` | directory | Rare. A per-package resource. |
| `session` | whole run | Expensive immutable things: a started container, a loaded model, a compiled binary. |

**The rule that matters: a widely-scoped fixture must be immutable, or it becomes shared mutable
state and breaks test independence (rule 11).** A session-scoped `Wallet` means test A's deposit
changes what test B sees, and the suite passes or fails depending on collection order.

Okken's pattern for expensive-but-mutable resources: two fixtures, one per scope:

```python
@pytest.fixture(scope="session")
def db_engine():
    """Expensive. Created once. Never mutated."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()

@pytest.fixture
def db(db_engine):
    """Cheap. Per test. Rolls back, so each test starts from the same state."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()
```

Session scope for the setup cost, function scope for resetting to a known state. Ruff PT003
flags the reverse mistake: explicitly writing `scope="function"`, which is already the default.

A narrower-scoped fixture cannot be requested by a wider-scoped one. A `session` fixture asking
for `tmp_path` (function-scoped) is a `ScopeMismatch` error at collection. Use
`tmp_path_factory` instead.

## Teardown

`yield` splits the fixture into setup and teardown. Everything after the `yield` runs when the
fixture's scope ends, including when the test fails.

```python
@pytest.fixture
def server():
    proc = start_server()          # setup
    yield proc                     # the test runs here
    proc.terminate()               # teardown, even if the test raised
```

Either half can be empty. If teardown must run even when *setup* fails partway, wrap in
`try/finally`.

`request.addfinalizer()` is the older API. It still works, and Okken's advice is to avoid it:
`yield` reads better and nobody has to remember the ordering rules.

Teardown runs in reverse dependency order: if `db` depends on `db_engine`, `db` tears down
first.

## conftest.py layering

`conftest.py` is imported automatically. Fixtures defined there are visible to every test in
that directory and below, with no import statement. Fixtures do not need to be exported and
should not be imported by hand.

```
tests/
├── conftest.py                 # seeding, network blocking, universal fixtures
├── unit/
│   ├── conftest.py             # unit-only fixtures
│   └── test_wallet.py
└── integration/
    ├── conftest.py             # real DB, real HTTP client
    └── test_checkout.py
```

Resolution walks upward from the test file, so a nearer `conftest.py` shadows a farther one by
name. That is a useful override mechanism and a nasty debugging surprise in equal measure: when
a fixture is not what you expect, `pytest --fixtures <path>` shows which file won.

The root `conftest.py` also determines `rootdir` behavior and is where `pytest_configure`,
`pytest_collection_modifyitems`, and other hooks live. See `assets/conftest.py`.

## Built-in fixtures

Verified present in pytest 9.2.0. These are the ones worth knowing by heart.

| Fixture | Scope | What it gives you |
|---|---|---|
| `tmp_path` | function | A `pathlib.Path` to an empty directory unique to this test. |
| `tmp_path_factory` | session | `.mktemp("name")` for wider-scoped temp directories. |
| `monkeypatch` | function | Temporary attribute/dict/env/cwd changes, undone at teardown. |
| `capsys` | function | `.readouterr()` returns captured `(out, err)` at the Python level. |
| `capteesys` | function | New in pytest 9: captures **and** still prints to the terminal. |
| `capfd` | function | Same, at the file-descriptor level: catches output from C extensions and subprocesses. |
| `capsysbinary` / `capfdbinary` | function | Bytes rather than str. |
| `caplog` | function | Captured log records: `.text`, `.records`, `.at_level(...)`. |
| `recwarn` | function | Every warning raised, as a list. Prefer `pytest.warns` when you know what you expect. |
| `request` | function | Metadata about the running test. See below. |
| `pytestconfig` | session | Config values and command-line options. |
| `subtests` | function | New in pytest 9: multiple independently-reported checks inside one test. |
| `cache` | function | Cross-run key/value store (what `--lf` and `--ff` are built on). |

`tmpdir` and `tmpdir_factory` still exist but return legacy `py.path.local` objects. Use the
`tmp_path` versions.

Installed packages can add fixtures through plugin entry points without any import in your
code: in the verification venv, `platformdirs` 4.x registered `platformdirs_isolated`.
`pytest --fixtures -v` prints the file each fixture comes from;
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` turns entry-point loading off when you need a clean run.

### capsys vs caplog

```python
def test_receipt_is_printed(capsys):
    print_receipt(wallet)                     # the CODE prints
    captured = capsys.readouterr()
    assert captured.out == "receipt: 100.0 USD\n"
    assert captured.err == ""

def test_withdrawal_is_logged(caplog):
    with caplog.at_level(logging.INFO):
        wallet.withdraw(30.0)
    assert "withdraw 30.0" in caplog.text
    assert caplog.records[0].levelno == logging.INFO
```

Both make output *assertable*. `print()` written by the test itself asserts nothing, which is why
the audit flags it (`PRINT`). `print()` written by the code under test and captured here is the
correct pattern.

`readouterr()` clears the buffer, so call it once and hold the result.

## Factory fixtures

When a test needs several instances, or needs to vary one, return a function instead of an object.

```python
@pytest.fixture
def make_wallet():
    def _make(balance=0.0, currency="USD"):
        return Wallet(balance=balance, currency=currency)
    return _make

def test_two_wallets_are_independent(make_wallet):
    a = make_wallet(balance=50.0)
    b = make_wallet(balance=10.0, currency="EUR")
    a.withdraw(50.0)
    assert a.balance == pytest.approx(0.0)
    assert b.balance == pytest.approx(10.0)
```

With cleanup, track what you made and tear it all down:

```python
@pytest.fixture
def make_tempfile(tmp_path):
    created = []
    def _make(name, content=""):
        path = tmp_path / name
        path.write_text(content)
        created.append(path)
        return path
    yield _make
    for path in created:
        path.unlink(missing_ok=True)
```

## Parametrized fixtures

`params=` runs every test that requests the fixture once per value. Use it when *setup* differs
per case; use `@pytest.mark.parametrize` when only the *inputs* differ.

```python
@pytest.fixture(params=["sqlite", "postgres"])
def db(request):
    if request.param == "sqlite":
        yield sqlite_session()
    else:
        yield postgres_session()
```

Every test requesting `db` now runs twice. That is powerful and easy to overdo: the multiplier
compounds across fixtures, and two three-param fixtures in one test means nine runs.

Attach marks to individual params with `pytest.param`:

```python
@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    ...
```

Verified signature: `pytest.param(*values, marks=(), id=None)`.

Okken's ordering: prefer function parametrization; reach for fixture parametrization when work
is needed per parameter; use the `pytest_generate_tests` hook only when you are certain nothing
else will do.

## autouse

`autouse=True` applies the fixture to every test in its scope without being requested.

Right for cross-cutting invariants that every test needs and no test should have to remember:
RNG seeding, network blocking, environment cleaning. `assets/conftest.py` uses it for exactly
these three.

Wrong for anything a test's behavior depends on, because the dependency becomes invisible. A
reader of the test cannot tell where the state came from. If the test needs it, let the test ask
for it by name.

## The request object

```python
@pytest.fixture
def resource(request):
    request.node.name                              # the test's name
    request.node.get_closest_marker("integration") # the marker, or None
    request.param                                  # the current value, in a parametrized fixture
    request.config.getoption("--my-flag")          # a command-line option
    request.getfixturevalue("other")               # resolve a fixture by name at runtime
```

`get_closest_marker` is how a fixture changes behavior per marker: the network blocker in
`assets/conftest.py` uses it to exempt integration tests.

`getfixturevalue` resolves a fixture dynamically. It defeats static analysis of the dependency
graph, so use it only when the name genuinely is not known until runtime.

## Registering fixtures from code (pytest 9.1+)

`pytest.register_fixture` is for plugins that must create fixtures whose names are only
known at run time (one fixture per configured backend, say). Its signature is
keyword-only, read with `inspect` on 9.2.0:

```
register_fixture(*, name, func, node, scope="function", params=None, ids=None, autouse=False)
```

`node` sets visibility: only items under that collection node see the fixture. Calling it
from a plain `pytest_sessionstart` fails with `AttributeError: 'Session' object has no
attribute '_fixturemanager'`, because the fixture manager is attached by another
`pytest_sessionstart` implementation. Run yours after it:

```python
@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    for name in ("sqlite", "memory"):
        pytest.register_fixture(name=f"{name}_backend", func=_make_backend(name), node=session)
```

That ran green (2 passed). In a project, a parametrized fixture is almost always the
better answer; reach for this only inside a plugin.

## Fixture anti-patterns

| Pattern | Problem | Fix |
|---|---|---|
| Session-scoped mutable object | Shared state; order dependence | Function scope, or session setup + function reset |
| One `setup_everything` fixture | Every test pays for what one test needs | Small composable fixtures |
| `autouse` fixture the test depends on | Invisible dependency | Request it by name |
| Assertions inside a fixture | A setup failure reports as ERROR, not FAILURE, and the message points at the wrong place | Assert in the test |
| Fixture returning a tuple of unrelated things | Callers unpack junk they do not need | Separate fixtures |
| `@pytest.fixture` on a function named `test_*` | pytest collects it as a test *and* a fixture | Rename it (ruff PT004/PT005) |
| Importing a fixture from another test module | Breaks under `--import-mode=importlib`; bypasses the resolution order | Move it to `conftest.py` |
