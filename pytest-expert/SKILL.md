---
name: pytest-expert
description: Build, audit, and harden pytest suites for Python codebases. Use whenever the user writes, reviews, or fixes tests; configures pytest or coverage; hunts flaky tests; tests async (asyncio, anyio) or ML/PyTorch code; or asks for property-based testing (Hypothesis), fuzzing (Atheris), symbolic execution (CrossHair), mutation testing, snapshot tests, security tests, integration tests, CI test pipelines, pre-commit hooks, or Claude Code hooks that run tests. Also use when an AI agent wrote or changed tests, says "all tests pass now", or a suite went green suspiciously, and proactively when writing Python code that will need tests. Ships a stdlib audit that fails the build on tests that cannot fail, a diff guard that catches tests weakened to get green, a mutation checker, a flaky-test hunter, hooks, hardened workflows, templates, and worked examples.
---

# pytest Expert

Test suites that catch real bugs, run fast, and fail loudly when a test is lying. A
green suite proves less than it looks like: tests that assert nothing, failures
swallowed by `except`, mocks that accept any call, NaN slipping past comparisons. The
tooling here makes those states hard to reach quietly, then measures whether the
tests can fail at all.

Verified November 2026 against pytest 9.2.0, pytest-asyncio 1.4.1, Hypothesis 6.170.14,
CrossHair 0.0.112, Atheris 3.1.0, mutmut 3.8.1, ruff 0.17.3, zizmor 1.31.1, actionlint
1.7.12, PyTorch 2.15, on Python 3.13. Every command and output quoted in `references/`
was run; research claims cite a digest in `references/papers/`.

## Terms

| Term | Meaning |
|---|---|
| **Fixture** | A `@pytest.fixture` function whose return value is injected into any test that names it as a parameter. |
| **Parametrize** | `@pytest.mark.parametrize`: one test function, run once per input row, each reported separately. |
| **Marker** | A label (`@pytest.mark.integration`) used to select or deselect tests. |
| **conftest.py** | A file pytest imports automatically; its fixtures and hooks apply to its directory and below. |
| **Test double** | A stand-in for a real dependency: a fake, a stub, or a **mock** (records how it was called). |
| **Flaky test** | Passes or fails without the code changing. Always a defect. |
| **Branch coverage** | Share of decision outcomes executed; `if x:` needs both paths. |
| **Property-based test** | Checks a rule on many generated inputs; Hypothesis shrinks a failure to the smallest case. |
| **Fuzzing** | Coverage-guided input mutation (Atheris): keep inputs that reach new code, report crashes. |
| **Symbolic execution** | A constraint solver finds inputs that reach each code path (CrossHair). |
| **Mutant** | A small deliberate change to the code. Tests should fail on it ("killed"); if not, it "survived". |
| **Hook** | A command run automatically on an event: a git commit, or Claude editing a file. |
| **Test gaming** | Editing tests or test config so the suite passes without the behavior being correct (deleting a failing case, adding a skip, lowering a floor). |

## Workflow

Each step ends on a check you can run. Do not move on until it passes.

**1. Read the code before writing tests.** Find entry points, pure functions, and I/O
boundaries, and look at the existing layout (`tests/`, `conftest.py`,
`pyproject.toml`). Never invent an API; when behavior is ambiguous, read the callers.
*Done when* you can list the functions to test and the boundary each one crosses.

**2. Install the gates.** Merge `assets/pytest-config.toml` into `pyproject.toml`,
copy `assets/conftest.py` to `tests/conftest.py` (add `assets/conftest_ml.py` for
PyTorch), and adopt `assets/ruff-pytest.toml`. Copy `scripts/audit_tests.py` into the
project's `scripts/`. *Done when* `pytest --collect-only -q` succeeds under strict
config.

**3. Write tests at the right layer.** Use the ladder below and the hard gates. Thread
one realistic example through related tests. *Done when* every behavior you were
asked to cover has a test and `python scripts/audit_tests.py tests/` prints
`clean: no findings`.

**4. Run the gates.**

```
python scripts/audit_tests.py tests/
ruff check .
pytest
pytest -p randomly -n auto
python scripts/diff_guard.py
```

*Done when* all five exit 0. Report audit findings and how you resolved each. A green
pytest run over a suite the audit flags is not a pass, and neither is one the diff
guard flags: it compares the tests with a git base (default `HEAD`; `--base <ref>` for
the commit before your work) and reports deleted tests or parametrize rows, dropped
assertions, new skips, deselection, a lowered coverage floor, and source-side cheats.

**5. Prove the tests can fail.** Run the mutation checker on each function you tested:

```
python scripts/mutate_check.py src/pkg/module.py --function name -- python -m pytest -x -q -p no:randomly --no-cov tests/
```

*Done when* every surviving mutant is killed by a new test, or recorded as equivalent
(no input can distinguish it) or unproductive (killable, but the test would pin an
arbitrary detail), each with a one-line reason. Report the score. This is the
highest-value step and the easiest to skip. Coverage does not substitute: 49% of the
mutant-killing tests Meta kept added no line coverage (`references/papers/meta-ach.md`).

**6. Escalate where the risk is.** For each code shape in the ladder below, add the
layer it calls for. *Done when* each parser has an exception-contract property, each
security boundary has a payload table plus a property, and the user has said which
slow layers (fuzzing, solver, mutmut) to schedule.

**7. Automate, when the user wants CI or hooks.** Copy the workflows from
`assets/github/` into `.github/`, `assets/.pre-commit-config.yaml` to the repo root,
and the Claude Code hooks per `references/hooks.md`. *Done when* `actionlint` and
`zizmor` report no findings and `pre-commit run --all-files` passes.

## When the tests and the code disagree

Never edit, skip, or delete a failing test to make it pass, and never special-case the
code for a test's inputs. If the test is wrong (it contradicts the spec), say so: with
the hooks installed, write `.claude/pytest-expert-escalation.md` (Issue type, Test,
Problem, Tried, Recommendation) and stop. Evidence and details:
`references/llm-written-tests.md`.

## The testing ladder

| Code shape | Layer | Reference |
|---|---|---|
| Pure logic with branches | example tests, parametrized | `rules.md`, `parametrize-and-doubles.md` |
| Parsers, serializers, validators | property tests (round trip, exception contract) and a fuzz target | `property-fuzz-symbolic.md` |
| Logic gated by exact values (checksums, magic constants, narrow ranges) | solver search: Hypothesis `backend="crosshair"` | `property-fuzz-symbolic.md` |
| State across operations | Hypothesis stateful machine against a model | `property-fuzz-symbolic.md` |
| Untrusted input reaching files, SQL, deserialization, auth | security payload tables plus properties | `security-testing.md` |
| Real database, process, or service | integration tests, marked and deselected by default | `integration-testing.md` |
| Models, tensors, training loops | determinism, shapes, gradients, overfit one batch | `ml-testing.md` |
| Coroutines, task groups, timeouts | injected sleep, deadlines in tests, `RaisesGroup` | `async-testing.md` |
| Large stable rendered output | snapshot, reviewed on every update | `parametrize-and-doubles.md` |
| Tests an AI agent wrote or changed | spec-derived expected values, diff guard, mutation check | `llm-written-tests.md` |
| The suite as a whole | mutmut nightly, flake hunt nightly | `mutation-testing.md`, `flaky-tests.md` |

Measured on the worked example: random generation found malformed-input bugs in
seconds but missed a checksum-gated bug in 3 of 3 runs at 1,000 examples; the solver
backend found it in about 0.4 seconds (CrossHair 0.0.112). Use both.

## The hard gates

Enforced by `scripts/audit_tests.py`, ruff, or pytest strict mode. Codes in parentheses
are audit codes; `references/rules.md` explains each rule with its source.

**Structure**
1. One behavior per test. If the name needs "and", it is two tests.
2. Long declarative names: `test_withdraw_rejects_amount_over_balance`.
3. Arrange, Act, Assert, in that order. Plain functions, not `unittest.TestCase` (`UNITTEST_CLASS`).
4. No `if`, `while`, or for-over-cases in a test body (`BRANCHING`). Parametrize instead.
5. No duplicate test names in one scope (`SHADOWED_TEST_NAME`).

**Assertions**
6. Every test asserts something (`NO_ASSERT`); `print` is not a check (`PRINT`, warning).
7. No assertion that cannot fail (`ALWAYS_TRUE`), including `assert (x, "msg")`.
8. Floats never compare with `==` (`FLOAT_EQ`). Use `pytest.approx`.
9. `pytest.raises` names a specific exception and passes `match=` (`BROAD_RAISES`). Anchor the pattern when the message is the contract.
10. Assert exact expected values, not loose ranges; check the type when it matters (`5.0 == 5` is true).
28. Never swallow a failure with `except AssertionError`, `except Exception`, or bare `except` (`SWALLOWED_FAILURE`).
29. A comparison must be asserted (`BARE_COMPARISON`).
30. Never compare with NaN using `==` or `!=` (`NAN_COMPARE`).
31. At least one assertion pins a value, not just existence (`WEAK_ASSERT`, warning).
33. Exception groups are asserted with `pytest.RaisesGroup` or `group_contains`.

**Isolation and determinism**
11. Tests pass alone, in any order, and in parallel.
12. No unseeded randomness (`UNSEEDED_RANDOM`). Seed centrally in `conftest.py`.
13. No module-level mutable state that tests mutate (`MUTABLE_MODULE_STATE`).
14. No network in unit tests (`NETWORK`); integration tests are marked and deselected by default.
15. No `time.sleep` in tests or fixtures (`SLEEP`). Inject a clock.
16. File writes go to `tmp_path`.
17. Environment changes go through `monkeypatch` (`ENV_MUTATION`).
18. No `sys.path` manipulation (`SYS_PATH`). Fix the packaging.

**Test doubles**
19. `mock.patch` and `patch.object` pass `autospec=True` (`PATCH_NO_SPEC`).
20. Patch where the name is used, not where it is defined.
21. Never call a Mock method that only looks like an assertion (`FAKE_MOCK_ASSERT`).
22. Prefer real objects, then fakes, then `monkeypatch`, then mocks.

**Markers and skips**
23. Every marker is registered; strict mode rejects typos.
24. `skip`, `skipif`, and `xfail` carry a reason (`SKIP_NO_REASON`).
25. `xfail` is strict (`XFAIL_NOT_STRICT`, silent when the config sets `strict_xfail`).
32. No `yield_fixture` (`YIELD_FIXTURE`); pytest 9.2 schedules it for removal.

**Coverage**
26. Branch coverage with an enforced floor; ratchet it up, never down.
27. Do not chase 100%. Coverage shows what is untested; mutation testing shows whether tested code is tested well.

**Integrity**
34. A suite made green by weakening tests is not green (`scripts/diff_guard.py`).

Deliberate exceptions get an inline pragma with a reason:
`# audit: ignore[CODE] reason`. A pragma without a reason is itself a finding.

## Choosing what to test

1. Pure functions with branching logic: cheapest, densest in real bugs.
2. Boundaries you own: parsers, serializers, validators. Both directions, plus malformed input.
3. Every bug you fix: write the failing test first.
4. Error paths: the `except` branch that never ran is the one that crashes.
5. Integration seams: a few tests proving components connect, run separately.

Not worth testing: getters, third-party behavior, `__repr__`, pure configuration.

## Reference files

| File | Read when |
|---|---|
| `references/rules.md` | justifying a rule, resolving a conflict, the audit code index, suppression syntax |
| `references/fixtures.md` | designing fixtures: scopes, conftest layering, factories, teardown |
| `references/parametrize-and-doubles.md` | parametrize patterns and the test-double decision tree |
| `references/property-fuzz-symbolic.md` | Hypothesis patterns and gotchas, CrossHair, Atheris, which finds what |
| `references/security-testing.md` | traversal, injection, deserialization, limits, authz matrices, SAST, secrets |
| `references/mutation-testing.md` | `mutate_check.py`, mutmut, reading survivors, equivalent mutants |
| `references/integration-testing.md` | fakes vs real dependencies, testcontainers, API contract tests |
| `references/flaky-tests.md` | order dependence, nondeterminism, `flake_hunt.py`, fixing without retries |
| `references/ci-cd.md` | workflow hardening, coverage, speed, supply chain, debugging flags |
| `references/hooks.md` | pre-commit stages, Claude Code PostToolUse and Stop hooks |
| `references/ml-testing.md` | models, tensors, data pipelines, training loops, GPU markers |
| `references/async-testing.md` | pytest-asyncio 1.x config and loop scope, anyio, hangs, cancellation, injected time |
| `references/llm-written-tests.md` | an agent writing or changing tests: oracles, gaming codes, mutation as acceptance, escalation |
| `references/sources.md` | every research source, with venue, how it was read, and what it supports |
| `references/papers/README.md` | 16 paper digests (mutation testing, LLM test generation, ImpossibleBench, PBT, flaky tests, smells) |

## Assets

| File | Use |
|---|---|
| `assets/pytest-config.toml` | strict pytest, coverage, and mutmut config for `pyproject.toml` |
| `assets/conftest.py` | seeding, network guard, cwd and env isolation, Hypothesis profiles, directory markers |
| `assets/conftest_ml.py` | torch determinism, device fixtures, tiny model and batch |
| `assets/conftest_integration.py` | session PostgreSQL container with per-test rollback |
| `assets/ruff-pytest.toml` | PT, bandit (S), bugbear, and friends |
| `assets/test_example.py` | every core rule in working code |
| `assets/test_properties_template.py` | eight property patterns, runnable |
| `assets/test_security_template.py` | six security patterns, runnable |
| `assets/fuzz_target_template.py` | Atheris harness skeleton |
| `assets/test_async_template.py` | seven async patterns, runnable (needs pytest-asyncio) |
| `assets/github/workflows/tests.yml` | merge gate: audit, lint, matrix tests, integration |
| `assets/github/workflows/nightly.yml` | Hypothesis, CrossHair, Atheris, mutmut, flake hunt |
| `assets/github/workflows/security.yml` | pip-audit, ruff S, zizmor, gitleaks, dependency review |
| `assets/github/dependabot.yml` | keeps SHA pins and dependencies current |
| `assets/.pre-commit-config.yaml` | commit and push hooks |
| `assets/claude/` | Claude Code hooks: SessionStart base commit, post-edit audit, Stop/SubagentStop gate with escalation exit |

## Scripts

| Script | Does |
|---|---|
| `scripts/audit_tests.py` | stdlib AST audit; 25 codes; text, json, github, and sarif output; exit 1 on any error |
| `scripts/diff_guard.py` | stdlib git + AST; test-weakening edits since a base ref (16 codes), source-side cheats as warnings |
| `scripts/mutate_check.py` | stdlib targeted mutation check for one file or function |
| `scripts/flake_hunt.py` | reruns under many random orders; labels order-dependent vs nondeterministic |

Each prints usage with `--help`; the audit and the guard take `--list-codes` and
`--format json|github`. `selftest/` tests the scripts, the hooks, and the skill's own structure (`pytest selftest -q`).

## Worked example

`examples/wallet/` is a small package tested at every layer: unit, property
(including a stateful model), security, integration, symbolic, and a fuzz target with
corpus replay. `examples/wallet/demo/` holds v0 versions with planted bugs and the
tests that find them. Start with `examples/wallet/README.md`.

`examples/diagnoses/` holds five worked diagnoses with real script output (production
bug behind a green suite, a gamed suite, a CI-only flake, a hanging async suite, 100%
coverage at 38% mutation score). Index: `examples/diagnoses/README.md`.

## Sources

Research (mutation testing, coverage, LLM-written tests, ImpossibleBench, PBT, flaky
tests, smells): `references/sources.md`, with digests in `references/papers/`.
Per-rule sources: `references/rules.md`. Tool docs (pytest, ruff, Hypothesis,
CrossHair, Atheris, GitHub Actions, Claude Code hooks) are cited in the reference that
uses them. What changed and how it was verified: `CHANGELOG.md`, `AUDIT.md`.
