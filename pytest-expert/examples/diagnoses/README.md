# Worked diagnoses

Each example is an invented system, used to show a diagnosis end to end: the symptom,
the candidate causes, the cheapest test that separates them, real output from this
skill's scripts on that system, and the fix. Outputs were captured on 2026-11-02 with
pytest 9.2.0 on Python 3.13. The systems are illustrations, not findings about any real
project.

| File | Symptom | Main scripts | References |
|---|---|---|---|
| `01-green-suite-bugs-in-prod.md` | green, 100% coverage, bug in production | `audit_tests.py`, `mutate_check.py` | `rules.md`, `mutation-testing.md` |
| `02-agent-gamed-the-suite.md` | an agent says the suite passes now | `diff_guard.py` | `llm-written-tests.md`, `hooks.md` |
| `03-ci-only-flake.md` | fails in CI one run in three | `flake_hunt.py` | `flaky-tests.md` |
| `04-async-test-hangs.md` | async suite hangs until the CI job dies | pytest-timeout | `async-testing.md` |
| `05-coverage-high-mutation-low.md` | coverage 100%, audit clean | `mutate_check.py` | `mutation-testing.md` |

The shape every answer keeps: verdict first, then candidate causes, the separating
test, the evidence, and the fix.
