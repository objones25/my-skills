# 03: "One test fails in CI maybe a third of the time and never on my laptop"

*Illustrative system.* A currency module with a module-level rate table
(`src/rates/cache.py`); CI runs with pytest-randomly, laptops run in file order.

## Verdict

Order dependence, not CI. A test that sets a rate leaks it into a test that assumes no
rate is set. Isolate the shared state with a fixture; do not add a retry.

## Candidate causes

1. Order dependence: shared state between tests (Gruber et al. found order dependence
   behind 59% of flaky Python tests, `papers/gruber-flaky-python.md`).
2. Nondeterminism inside one test (time, randomness, network).
3. Environment differences (CPU count, locale, timezone).

`flake_hunt.py` separates 1 from 2: it reruns under many random orders and repeats each
seed. A failure that follows the seed is order-dependent; one that varies within a seed
is nondeterministic.

## The evidence

```
$ pytest -q -p no:randomly
3 passed

$ python scripts/flake_hunt.py --runs 12 --repeat-seed 2 -- tests/
3 tests x 12 seeds x 2 repeat(s)
FLAKY tests.test_convert::test_unknown_currency_converts_at_par
      order-dependent; fails with --randomly-seed in [1, 2, 3, 5, 6, 8, 11, 12]

1 flaky test(s)
```

Reproducing one seed shows the leak:

```
$ pytest -q -p randomly --randomly-seed=1
E       AssertionError: assert 500 == 1000
E        +  where 500 = convert(1000, 'EUR')
FAILED tests/test_convert.py::test_unknown_currency_converts_at_par
```

`test_set_rate_changes_conversion` set EUR to 0.5 in a module-level dict and never
undid it.

## The fix

An autouse fixture gives each test an empty table and restores the real one:

```python
@pytest.fixture(autouse=True)
def isolated_rates(monkeypatch):
    monkeypatch.setattr(cache, "_RATES", {})
```

```
$ python scripts/flake_hunt.py --runs 12 --repeat-seed 2 -- tests/
3 tests x 12 seeds x 2 repeat(s)
no flaky tests found
```

Twelve seeds clear this case; to call a test free of order-dependent flakiness at 95%
confidence, Gruber et al. estimate about 31 random-order runs (`flaky-tests.md`).
Better still, make the rate table an object passed in, so there is no module state.
