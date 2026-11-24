# Flaky tests

A **flaky test** passes or fails without the code changing. It is always a defect,
usually in the test, and sometimes a real race in the code.

## What the data says about Python

Gruber et al. (ICST 2021) analyzed 876,186 tests from 22,352 PyPI projects and found
7,571 flaky tests:

| Cause | Share |
|---|---|
| Order dependency (a test depends on what ran before it) | 59% |
| Test infrastructure problems | 28% |
| Not order-dependent, any cause (network and randomness were 79 of the 100 sampled) | 13% |

Their rerun numbers depend on the kind of flakiness and the question
(`papers/gruber-flaky-python.md`, [Gruber21 §IV-C]):

| Question | Not order-dependent | Order-dependent |
|---|---|---|
| Is this passing test free of flakiness (95% confidence)? | about 170 reruns | about 31 random-order runs |
| Was this one observed failure flaky (95% confidence, most tests)? | 1 rerun | 3 reruns |

So two runs prove almost nothing about a test being clean, but they are often enough
to classify a single failure. `flake_hunt.py` defaults to `--runs 31`, which matches the
order-dependent figure, the common case; it says little about rare non-order-dependent
flakiness.

## Hunting: `scripts/flake_hunt.py`

Reruns a selection under many pytest-randomly seeds and reports tests whose outcome
changes. `--repeat-seed` reruns each order to separate the two kinds:

| Kind | Signature | Usual cause |
|---|---|---|
| order-dependent | fails for specific seeds, every time | shared state between tests |
| nondeterministic | fails at random, even with the same order | randomness, time, threads, network |

Without `--repeat-seed`, pytest-randomly reseeds `random` per test from the run's seed, so a
test that draws from `random` fails for particular seeds every time and is labelled
`order-dependent or nondeterministic (use --repeat-seed 3 to tell apart)`. With
`--repeat-seed N` (N > 1) the repeats of each seed reseed the random number generators from
OS entropy, so the draw changes between repeats while the order stays fixed, and the test is
labelled `nondeterministic`. Without pytest-randomly nothing is shuffled, so only
nondeterminism can show: the header says `pytest-randomly not installed; order fixed`, and a
flake is `nondeterministic; fails in runs [20, 23, 25, 36, 52] of 60` (the coin flip above,
`--runs 60`, 2026-11-20).

Real output on a planted suite (a test that relies on a module-level list being
empty, and a 1-in-10 coin flip), 2026-11-20, macOS arm64, Python 3.13, pytest 9.2.0,
pytest-randomly 5.1.0. The coin flip's seeds differ from run to run; the order-dependent
test's do not:

```
3 tests x 31 seeds x 2 repeat(s)
FLAKY tests.test_cache::test_coin_flip_that_fails_one_time_in_ten
      nondeterministic; fails with --randomly-seed in [2, 6, 12, 21, 25, 28, 30]
FLAKY tests.test_cache::test_first_remembered_item_has_count_one
      order-dependent; fails with --randomly-seed in [1, 5, 8, 12, 13, 15, 16, 18]

2 flaky test(s)
```

Command (the default is `--runs 31`). An `xfail` test that fails counts as a failure and one that passes counts as a pass, so an `xfail` test that flips between the two is reported:

```
python scripts/flake_hunt.py --runs 31 --repeat-seed 2 -- tests/
```

Exit status: 0 = every test behaved consistently, 1 = at least one flaky test, 2 = could not
hunt (bad `--runs` or `--repeat-seed`, a path or `-k` that selects no test, a pytest usage,
internal or collection error or crash, nothing executed). Exit 2 is never a clean result. Do
not pass pytest-xdist's `-n`: tests in different worker processes cannot interfere, so an
order-dependent pair would look clean, and the script refuses it with exit 2.

Reproduce one failing order:

```
pytest -p randomly --randomly-seed=5
```

The nightly workflow runs a 31-seed hunt; treat a clean result as evidence, not proof.

## Causes, in the order to check them

| Cause | How to confirm | Audit code |
|---|---|---|
| Module-level mutable state | fails only after a specific other test | `MUTABLE_MODULE_STATE` |
| Wide fixture scope holding something mutable | `pytest --fixtures` shows a session or module scope | none |
| Unseeded randomness | different values per run | `UNSEEDED_RANDOM` |
| Real time: `datetime.now()`, timeouts, sleeps | fails near midnight, under load, or across timezones | `SLEEP`, ruff `DTZ` |
| A real file at a fixed path | parallel runs collide | none; use `tmp_path` |
| Environment variable set without monkeypatch | leaks into later tests | `ENV_MUTATION` |
| Network | fails offline or under rate limits | `NETWORK` |
| Set iteration order | differs between interpreter processes | none |

On set order: dicts keep insertion order in every supported Python version, so
dict iteration is not hash-dependent. Iterating a `set` or `frozenset` of `str` or
`bytes` is, because string hashes are randomized per process. `PYTHONHASHSEED` pins
that only if it is set before the interpreter starts (in the CI workflow `env`, or
the shell); setting it inside a running test affects child processes only. This
matters for pytest-xdist workers and for parametrizing over a set.

## Fixing

Fix the cause. Never add a retry decorator or `pytest-rerunfailures` to a merge
gate: a retry converts "this test sometimes detects a real race" into "this test
detects nothing", and the race ships.

If you must quarantine while investigating, skip honestly with the issue number:

```python
@pytest.mark.skip(reason="flaky under parallel runs, investigating in #123")
```

`pytest -n auto` (pytest-xdist) and pytest-randomly in every CI run surface order
dependence early, which is why `assets/github/workflows/tests.yml` uses both.

## Sources

- Gruber, Lukasczyk, Kroiss, Fraser, *An Empirical Study of Flaky Tests in Python*, ICST 2021: https://arxiv.org/pdf/2101.09077
- pytest-randomly: https://github.com/pytest-dev/pytest-randomly
