# Integration testing

An **integration test** checks that your code works with a real dependency across a
boundary: a database engine, another process, a container, a network service. Unit
tests prove your logic; integration tests prove the seams.

## Contents

- [Choosing the double](#choosing-the-double)
- [Markers and layout](#markers-and-layout)
- [In-process real dependencies](#in-process-real-dependencies)
- [Containers with testcontainers](#containers-with-testcontainers)
- [API contract testing](#api-contract-testing)
- [Pitfalls](#pitfalls)

## Choosing the double

Move down this list only when the one above cannot exercise the behavior you need.

| Level | Example | Speed | Catches |
|---|---|---|---|
| Fake (in-memory implementation of your interface) | dict-backed repository | microseconds | your logic |
| Real, in-process | SQLite in `tmp_path` | milliseconds | SQL, schema, transactions |
| Real, in a container | PostgreSQL via testcontainers | seconds per session | engine-specific SQL, extensions, drivers |
| Real service in staging | contract tests on a schedule | minutes | deployment and config drift |

## Markers and layout

```
tests/
├── unit/          fast, default run
├── property/      fast, default run
├── security/      fast, default run, marked security
├── integration/   auto-marked integration, deselected by default
└── symbolic/      auto-marked symbolic, deselected by default
```

`assets/conftest.py` marks tests by directory in `pytest_collection_modifyitems`,
so nobody has to remember the decorator. The default `addopts` deselects
`integration`; the network blocker exempts it.

| Command | Runs |
|---|---|
| `pytest` | everything except integration, e2e, gpu, symbolic, slow |
| `pytest -m integration --no-cov` | integration only |
| `pytest -m ""` | everything |

## In-process real dependencies

From `examples/wallet/tests/integration/test_sqlite_file_store.py`:

```python
def test_wallet_survives_reopening_the_database(tmp_path):
    path = tmp_path / "wallets.db"
    first = SqliteWalletStore(path)
    first.put(Wallet(owner="alice", balance=Decimal("12.50")))
    first.close()
    second = SqliteWalletStore(path)
    assert second.get("alice") == Wallet(owner="alice", balance=Decimal("12.50"))
    second.close()
```

This catches what an in-memory database cannot: data that was never committed, or a
schema that only works on one connection.

## Containers with testcontainers

`assets/conftest_integration.py` starts PostgreSQL once per session and gives each
test a transaction that always rolls back:

```python
@pytest.fixture(scope="session")
def postgres_url():
    ...skip with a reason if testcontainers or Docker is missing...
    with PostgresContainer(POSTGRES_IMAGE, driver=None) as container:
        yield container.get_connection_url()

@pytest.fixture
def pg(postgres_url):
    with psycopg.connect(postgres_url) as conn:
        conn.autocommit = False
        try:
            yield conn
        finally:
            conn.rollback()
```

Session scope pays the startup cost once; function scope plus rollback keeps tests
independent (rule 11). Verification status: the skip path was run and reported
`SKIPPED ... Docker is not available`. The live-container path needs Docker and has
not been run against this template; run it once before relying on it.

GitHub's `ubuntu-latest` runners have Docker, so the `integration` job in
`assets/github/workflows/tests.yml` can run container tests as-is. Pin images by
digest (`postgres:17-alpine@sha256:...`) so a retagged image cannot change your
test environment.

## API contract testing

For HTTP services with an OpenAPI schema, Schemathesis generates requests from the
schema (property-based testing for APIs) and checks responses against it:

```
schemathesis run http://localhost:8000/openapi.json
```

Run it against a locally started service in the integration job. It finds 500
errors, responses that violate the schema, and inputs the schema allows but the
service rejects. Consumer-driven contract tools (Pact) solve a different problem:
proving two services you own agree on an interface.

## Pitfalls

| Pitfall | Fix |
|---|---|
| Tests share rows in a session-scoped database | per-test transaction rollback, or truncate in teardown |
| Fixed host ports collide under `pytest -n auto` | let the container pick a port; read it from the container object |
| Hidden dependency on test order | run the integration job with pytest-randomly enabled |
| Slow startup on every test | widen the fixture scope, keep the reset per test |
| A flaky external service | it belongs in a scheduled contract job, not the merge gate |
