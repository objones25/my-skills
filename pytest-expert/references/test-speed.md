# Keeping the suite fast

A test should run only the computation it is testing. Tests of a plot, a report, a
serializer or a metric often first train a model, run a simulation, or solve a large
system just to get their input. Each one costs seconds or minutes, and the full suite
grows to 30 minutes. Then nobody runs it before committing. Meszaros names the symptom:
"The tests take long enough to run that developers don't run them every time they make
a change to the system under test (SUT)." Software Engineering at Google says the same
of growth: "Tests often start fast enough but slow down as the system grows."

Sources were checked on 2026-11-27 against pytest 9.2.0, pytest-xdist 3.8.0, Hypothesis 6.170,
pytest-timeout 2.4.0, pytest-fail-slow 0.6.0, pytest-testmon 2.4.1 and pyinstrument 5.1.4.
Each source is listed at the end of this file.

## Contents

- [The rule](#the-rule)
- [Techniques, cheapest first](#techniques-cheapest-first)
- [Compute once: scope as a cache](#compute-once-scope-as-a-cache)
- [Hypothesis](#hypothesis)
- [ML and numerical code](#ml-and-numerical-code)
- [Finding slow tests](#finding-slow-tests)
- [Enforcement: the duration budget](#enforcement-the-duration-budget)
- [Dev loop](#dev-loop)
- [What not to do](#what-not-to-do)
- [Sources](#sources)

## The rule

Before writing the arrange step, ask what the test is about. If the expensive function
is not the subject, it must not run. Build its output directly, or a small version of it.

```python
# Bad: tests the plotting function, pays for a full training run.
def test_loss_curve_has_one_line_per_run():
    history = train(model_config, epochs=50)       # 4 minutes
    fig = plot_loss_curves([history])
    assert len(fig.axes[0].lines) == 1

# Good: the plot only needs a History; construct one.
def test_loss_curve_has_one_line_per_run():
    history = History(loss=[0.9, 0.5, 0.3])
    fig = plot_loss_curves([history])
    assert len(fig.axes[0].lines) == 1
```

If the code cannot take a hand-built input, because it calls `train()` internally, that is
a design finding. Add a seam: pass the result, or the function that computes it, as an
argument. See `parametrize-and-doubles.md` for choosing the double. The expensive function
gets its own tests, small or `slow`. One end-to-end `slow` test still proves the pieces
connect.

## Techniques, cheapest first

| Technique | When | Source |
|---|---|---|
| Construct the intermediate result directly | The test is downstream of the expensive step | Meszaros, *Minimal Fixture* |
| Shrink the problem: tiny config, small grid, few steps, small n | The test exercises the expensive code itself, but not its scale | transformers tests |
| Fake or stub the slow collaborator | The code calls it and cannot take a precomputed result | Meszaros, Google ch. 11 |
| Compute once per module or session, then copy per test | Many tests need the same costly, read-only artifact | pytest fixtures docs |
| Precomputed file checked into the repo | The artifact is costly and stable | Commit the script that regenerates it, and name it in the fixture's docstring |
| Mark `slow`, run nightly | The cost is the point (convergence, accuracy, the full pipeline) | Meszaros, *Subset Suite*; transformers `@slow` |

Meszaros, on general fixtures: "Because a General Fixture contains many more objects
than a Minimal Fixture (page X), it naturally takes longer to construct." And on slow
components: "We can make our tests run much faster by replacing the slow components
with a Test Double (page X) that provides near-instantaneous responses."

Google's chapter on testing says to "always write the smallest possible test for a given
piece of functionality". It also notes that "A test that runs on a single process and never
makes blocking calls can effectively run as fast as the CPU can handle."

## Compute once: scope as a cache

A `scope="module"` or `scope="session"` fixture runs once and is shared. pytest's docs
say tests "will thus each receive the same smtp_connection fixture instance, thus saving
time". Sharing is only safe for read-only data, because a shared mutable object breaks
rule 11 (see `fixtures.md`). When tests mutate it, hand each one a copy:

```python
@pytest.fixture(scope="session")
def _fitted_small_model():
    return fit(SMALL_DATASET, epochs=2)        # once per session, and still small

@pytest.fixture
def fitted_model(_fitted_small_model):
    return copy.deepcopy(_fitted_small_model)  # each test gets a fresh copy
```

The deepcopy pattern is this skill's recommendation, not something the pytest docs state.

Pitfalls:
- **Parametrized fixtures can rerun.** "Pytest only caches one instance of a fixture at a
  time, which means that when using a parametrized fixture, pytest may invoke a fixture more
  than once in the given scope."
- **xdist runs a session fixture once per worker**, not once per run: "tests in different
  processes requesting a high-level scoped fixture (for example session) will execute the
  fixture code more than once". The xdist docs' fix is a `FileLock` around a JSON file.
  `--dist loadscope` helps "if you have expensive module-level or class-level fixtures".
- **Large generated files** go through `tmp_path_factory`. The docs' example says: "Instead
  of computing the same image for each test that uses it into its own tmp_path, you can
  generate it once per-session to save time".
- **`request.config.cache` across runs** stores JSON. The API reference says it "uses the
  simple dumps/loads API of the json stdlib module", and `cache.mkdir(name)` gives a directory. The docs say nothing
  about invalidation. If you use it, key the entry on a hash of the code, config and seed
  that produced it; otherwise it goes stale silently. `pytest --cache-clear` removes it.

## Hypothesis

A `@given` test calls its body `max_examples` times; "The default max examples is
`100`". An expensive call inside a property runs a hundred times, so never train or
simulate inside `@given`. Build the expensive input once in a fixture instead, and draw
only the cheap part.

The per-example budget is Hypothesis's `deadline`: "The default deadline is 200
milliseconds. If running on CI, the default is `None` instead." Keep the deadline on in the
`dev` profile (`assets/conftest.py` does) so a costly example shows up locally. Control the
total with profiles. The docs recommend "running quick deterministic tests on every commit,
and a longer non-deterministic nightly testing run".

## ML and numerical code

Hugging Face transformers' testing guide states the pattern. Quoted from the guide:

- "The ModelTester class builds tiny configs and dummy inputs. Keep dimensions small so tests
  finish in seconds on CPU."
- "Mark any test with @slow if it downloads weights, loads a large dataset, or takes more
  than a few seconds. The pull request CI skips slow tests, but the nightly schedule runs
  them."

`RUN_SLOW=1` turns the slow ones on. The same idea applies to numerical code: test a solver
on a 4x4 system or an 8-point grid with a known answer, and keep the convergence study at
production size behind `slow`. `ml-testing.md` covers tiny models and the overfit-one-batch
test, which is already `slow`.

## Finding slow tests

```
pytest --durations=10 --durations-min=1.0
```

`--durations` (`--durations=0` for every test) reports setup, call and teardown separately, for example
`1.21s call test_a.py::test_slow`. A slow fixture shows up as `setup`, not as `call`, and a
module or session fixture is charged to the first test that uses it.

| Cost | How to see it |
|---|---|
| A test or fixture | `pytest --durations=10 --durations-min=1.0` |
| Imports at collection (heavy modules imported at the top of tests or conftest) | `python -X importtime -m pytest --collect-only -q 2> importtime.log` |
| Where the time goes inside the suite | `pyinstrument -m pytest [args]` (one consolidated report) |

Work at module level in a test file or `conftest.py` runs at collection, before any test.
It cannot be deselected and is billed to every run, even `pytest -k one_test`. Move it into
a fixture.

## Enforcement: the duration budget

`assets/conftest.py` fails any setup, call or teardown phase over `BUDGET_S` (default 2 s).
`PYTEST_EXPERT_BUDGET_S` changes it, and `0` turns it off. An empty value means the default.
Anything that is not a finite number >= 0 stops the run. The check skips tests marked `slow`,
`integration`, `e2e`, `gpu` or `symbolic`, xfail tests, and the call phase of Hypothesis tests.
The failure names the test and the phase:

```
______________________________ test_summary_table ______________________________
tests/test_report.py::test_summary_table call took 2.50s (budget 2s). If the test is not
about that computation, shrink its inputs, build the intermediate result directly, or use
a fake; otherwise mark it @pytest.mark.slow. See pytest --durations=10 and
references/test-speed.md in pytest-expert.
```

(Measured 2026-11-27, macOS arm64, Python 3.13.2, pytest 9.2.0, with a test whose `train()`
busy-waits 2.5 s. Line breaks added.)

It aims at the slow marker's "~1s" but stops at 2 s, so a 0.9 s test on a loaded CI runner
does not become flaky. It measures wall-clock time for what actually ran. It does not guess
from function names, but the time includes coverage tracing and contention between xdist
workers. If a per-push CI job flakes on it, give that job a wider margin
(`PYTEST_EXPERT_BUDGET_S: "4"` in the workflow `env`). Do not mark the test `slow` for that.
When you triage, measure with `--no-cov`.

Limits to know:

- **Hypothesis tests in CI are not checked.** Their call phase is exempt, because the total
  scales with `max_examples`. The `dev` profile's 200 ms `deadline` catches a costly example
  locally, but the `ci` and `nightly` profiles set `deadline=None`, so nothing checks them
  there. Run the property tests locally before pushing.
- **A shared fixture is charged to whichever test sets it up first.** Under pytest-randomly
  or xdist, that is a different test each run. If the failure points at a module- or
  session-scoped fixture, the fixture is the thing to shrink. If it is legitimately
  expensive, mark the tests that use it `slow`.
- **It only reports after the test finishes.** Pair it with a hard ceiling for runaways.
  pytest-timeout says it "is not designed for precise timings or performance regressions".
  It is the ceiling, not the budget:

```toml
timeout = 120   # pytest-timeout installed; matches assets/pytest-config.toml. Covers fixtures too ("The timeout applies to the entire test including any fixtures")
```

`faulthandler_timeout` only dumps tracebacks. It does not stop the test.

Turn the budget off (`PYTEST_EXPERT_BUDGET_S=0`) wherever tests run under a tool or run
deliberately long: mutation testing, nightly jobs, a debugger. Otherwise a test that is only
slower under the tool fails on time, and a mutant can count as killed for the wrong reason.
That is an inference: the slowdown under mutmut was not measured here.
`assets/github/workflows/nightly.yml` sets it for the whole workflow.

**Adopting it in an existing slow suite.** Turning it on fails every offender at once, and
the tempting response is to mark them all `slow`. Instead:

1. Run `PYTEST_EXPERT_BUDGET_S=0 pytest --no-cov --durations=0 --durations-min=2` for the
   full list, with setup, call and teardown separate.
2. Group the list by file and by fixture. A few shared fixtures usually account for most of
   the time.
3. Fix the biggest ones with the techniques above, mark only the tests whose cost is the
   point, and then turn the budget on.

pytest-fail-slow (`--fail-slow 2s`) does the same job as a plugin. It does not read the
`slow` marker; it uses its own `fail_slow` marker. Its last release was 2024-06-01.

## Dev loop

The full suite runs in CI and before a push. While editing, run less:

| Flag | Runs |
|---|---|
| `-x` | until the first failure |
| `--lf` / `--ff` | last failures only / failures first |
| `--sw` | stepwise: stop at a failure, resume from it next run |
| `-n auto` | across physical cores (pytest-xdist; logical cores if psutil is missing) |
| `--testmon` | only tests affected by changed code (pytest-testmon; the first run must use `--testmon` to build `.testmondata`) |

pytest-testmon's compatibility with xdist and coverage was not checked here.

## What not to do

- **A static check for calls named `train`, `fit` or `solve`.** `fit` is standard
  scikit-learn API and is cheap on tiny data. A name says nothing about input size, so a
  failing gate on names would teach agents to add ignore pragmas. The runtime budget
  measures the real cost instead.
- **Raising the budget to make a test pass.** That is test gaming. Shrink the input, or mark
  the test `slow` if its cost is the point. `scripts/diff_guard.py` does not yet flag an added
  `slow` marker or a raised budget, so check for those by hand when reviewing an agent's diff.
- **Marking everything `slow`.** A deselected test that should be fast is coverage you lost.
  `slow` is for tests where the cost is the point.
- **Session-scoped mutable state for speed.** It trades minutes for order-dependent failures.

## Sources

All checked 2026-11-27.

- pytest, *How to use fixtures* (scope, parametrized fixture caching): https://docs.pytest.org/en/stable/how-to/fixtures.html
- pytest, *How to use temporary directories*: https://docs.pytest.org/en/stable/how-to/tmp_path.html
- pytest, *How to re-run failed tests and maintain state between test runs* (cache, `--lf`, `--sw`): https://docs.pytest.org/en/stable/how-to/cache.html
- pytest, *API Reference* (`TestReport.duration`, `pytest_runtest_makereport`, `--durations-min`): https://docs.pytest.org/en/stable/reference/reference.html
- pytest-xdist 3.8.0, *How-tos* and *Distribution*: https://pytest-xdist.readthedocs.io/en/stable/how-to.html, https://pytest-xdist.readthedocs.io/en/stable/distribution.html
- Hypothesis, *API reference* (`max_examples`, `deadline`, profiles): https://hypothesis.readthedocs.io/en/latest/reference/api.html; `is_hypothesis_test` docstring read from Hypothesis 6.170.16
- pytest-timeout 2.4.0: https://pypi.org/project/pytest-timeout/
- pytest-fail-slow 0.6.0: https://pypi.org/project/pytest-fail-slow/
- pytest-testmon 2.4.1: https://pypi.org/project/pytest-testmon/
- pyinstrument 5.1.4, *User guide*: https://pyinstrument.readthedocs.io/en/latest/guide.html
- Python, *Command line and environment* (`-X importtime`): https://docs.python.org/3/using/cmdline.html
- Meszaros, *Slow Tests* (web draft of *xUnit Test Patterns*, generated 2011; the page says the published chapter "has likely changed substanstially" [sic]): http://xunitpatterns.com/Slow%20Tests.html
- Winters, Manshreck, Wright (eds.), *Software Engineering at Google*, ch. 11 *Testing Overview*: https://abseil.io/resources/swe-book/html/ch11.html
- Hugging Face transformers, *Writing model tests*: https://huggingface.co/docs/transformers/testing
