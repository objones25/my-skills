# Tests written by an AI agent

An agent asked to "make the tests pass" has three ways to get there: fix the code, fix
a test that was wrong, or weaken a test that was right. All three end green. This file
is about telling them apart, and about writing tests whose expected values come from
somewhere other than the code under test.

Script output quoted below is real. Research claims carry a location tag and point to
the digest in `references/papers/` that holds the full context.

## Contents

- [Terms](#terms)
- [The oracle problem](#the-oracle-problem)
- [Weakening edits and the diff guard](#weakening-edits-and-the-diff-guard)
- [Mutation score is the acceptance signal](#mutation-score-is-the-acceptance-signal)
- [A workflow for the agent](#a-workflow-for-the-agent)
- [A sanctioned "this test is wrong" exit](#a-sanctioned-this-test-is-wrong-exit)
- [What the hooks enforce](#what-the-hooks-enforce)

## Terms

| Term | Meaning |
|---|---|
| **Oracle** | Whatever decides that an output is correct: a hand-computed value, a spec, a property, a reference implementation. |
| **Change-detector test** | A test whose expected value was copied from the code's current output. It detects change, not error, and it locks in today's bugs. |
| **Test gaming** | Editing tests or test config so the suite passes without the behavior being correct. |
| **Mutation score** | Share of small deliberate code changes (mutants) that make at least one test fail. A test that cannot fail on a wrong answer has no value. |

## The oracle problem

The fastest way for an agent to write a passing test is to run the function, read the
result, and paste it into `assert f(x) == <result>`. That test passes by construction
and catches nothing that is already wrong. The tell is an expected value nobody could
have written without running the code: a long float, a 40-key dict, an exact error
string from deep inside a library.

Sources of expected values, best first:

1. **The spec, docstring, issue, or a hand calculation.** `parse_duration("1h30m") ==
   5400` because 3600 + 30 x 60 = 5400, not because the function said so.
2. **A property** that holds for every input: round trip, invariant, exception
   contract (`property-fuzz-symbolic.md`). The agent does not need to know any one
   output to write it.
3. **A slow, obviously correct reference** for a fast implementation (differential
   test).
4. **A snapshot**, only for large stable structures, reviewed by a human when created
   (`parametrize-and-doubles.md`, Snapshot testing).

When the only available oracle is the code itself, say so in the test's docstring
("characterization test: pins current behavior before refactor") so nobody mistakes it
for a correctness check. Meta's TestGen-LLM keeps only generated tests that build and
pass on the current code, which by design locks in existing behavior, bugs included
[TestGen §2] (`papers/meta-testgen-llm.md`). On code that already contains a bug,
coverage of LLM-written suites was uniformly weak at predicting whether they caught it
[Zhao26 §5.5] (`papers/llm-tests-proxy-metrics.md`).

## Weakening edits and the diff guard

`scripts/diff_guard.py` compares the test tree with a git base (the commit the session
started from, when the hooks are installed) and reports each pattern below. Errors fail
the Stop gate; warnings are shown for a human to judge.

| Code | Edit | Legitimate when |
|---|---|---|
| `TEST_FILE_DELETED`, `TEST_DELETED` | a test file or function removed | the feature was removed, or the test moved and was renamed (a rename reads as a delete) |
| `TEST_EMPTIED` | body reduced to `pass`, `...`, or an early `return` | almost never |
| `ASSERTS_REDUCED` | fewer `assert`, `pytest.raises`, `assert_*` calls | a redundant check was removed (say which) |
| `ASSERT_TRIVIALIZED` | `assert True`, `assert x == x` added | never |
| `CASES_REDUCED` | rows removed from a `@pytest.mark.parametrize` list | a duplicate row was removed (say which) |
| `SKIP_ADDED` | skip, skipif, xfail, `pytest.skip()`, `importorskip`, at test, class, or module level | a real platform or dependency limit, with a reason and an issue |
| `PRAGMA_ADDED` | new `# audit: ignore[...]` | a deliberate exception with a reason (still worth a human look) |
| `COLLECTION_NARROWED` | `collect_ignore` entries or a `pytest_collection_modifyitems` hook added | a real collection need, reviewed |
| `CONFIG_NARROWED` | addopts gained `--deselect`, `--ignore`, `-k`; `testpaths` lost a path | rarely in a shared config |
| `COVERAGE_LOWERED` | `fail_under` or `--cov-fail-under` lowered | never silently |
| `SYNTAX` | a test file no longer parses | never; reported instead of as mass deletion |
| `CONSTANT_EQ` (warning) | source gained an `__eq__`, `__ne__`, or `__hash__` that returns a constant | almost never; a sentinel class, perhaps |
| `SPECIAL_CASE` (warning) | changed source compares a value with a literal that a test also uses (`if cents == 1000: return 10`) | often legitimate (a boundary from the spec); a human decides |
| `EXPECTATION_CHANGED` (warning) | a literal on the expected side of an assert changed, including `snapshot(<literal>)` | the spec changed; the commit says how |
| `SNAPSHOT_CHANGED` (warning) | a syrupy `.ambr` file rewritten or deleted | the rendered output was meant to change |

Real output, after an edit that skipped the failing test and committed it:

```
tests/test_cart.py:9: SKIP_ADDED: test_discount: skip/xfail added
```

Why both sides matter: on ImpossibleBench (tasks whose tests contradict the spec, so
any pass is a cheat), Claude models and Qwen3-Coder cheated mostly (>79%) by editing
tests, while OpenAI models used all four strategies (editing tests, special-casing,
overloading comparison operators, recording extra state) at >14% each; read-only tests
stopped test edits but not special-casing or overloading [IB §4.2, §5.2]
(`papers/impossiblebench.md`). That is why the guard reads source diffs too.

What it still cannot see: a test rewritten to assert the same number of weaker things,
recorded extra state (a counter that changes behavior on the second call), a
special case on a literal no test file contains, or an expected value that was wrong
from the first commit. Mutation testing and review cover those.

When the user asks for a deletion, accept the specific code for that run
(`--allow TEST_DELETED`, or `PYTEST_EXPERT_DIFF_ALLOW=TEST_DELETED` for the hook) and
say in the reply which tests were removed and why. Do not widen the allow list to get
past a block you were not asked to cause.

## Mutation score is the acceptance signal

Coverage says a line ran. It does not say any assertion would notice if the line were
wrong. For tests an agent wrote, the acceptance check is: for each function the change
touched, run `scripts/mutate_check.py` and kill or justify every survivor
(`mutation-testing.md`). A survivor translates directly into a missing test: "nothing
fails when `>` becomes `>=` on line 31" is a precise instruction. That is the design of
Meta's ACH, which prompts the model with one mutant at a time and keeps only tests that
kill it; 49% of the tests it kept added no line coverage and a coverage filter would
have discarded them [ACH §1, Table 8] (`papers/meta-ach.md`). At Google, developers
shown individual surviving mutants during review wrote more tests (rs = 0.9 between
mutant exposure and test hunks; coverage alone, rs = -0.24) [Petrovic21 §III-C]
(`papers/petrovic-mutation-practices.md`). Coverage also let one TestGen-LLM test with
no assertion through its filters [TestGen §3.2.1].

Assertion strength matters for the same reason. The audit's `WEAK_ASSERT` flags a test
whose every check is existence-only (`is not None`, `isinstance`, `len(x) > 0`); such
tests survive almost every mutant.

## A workflow for the agent

1. Read the code and its callers; list the behaviors to test and where each expected
   value will come from (spec, hand calculation, property). No value from running the
   function under test.
2. For a bug fix, write the test first and watch it fail for the reason the bug
   describes. A test that passes before the fix does not test the fix.
3. Make the change. If an existing test now fails, decide explicitly: is the test
   wrong (the spec changed: update it and say so) or the code (fix the code)? Never
   edit the expected value to match new output without that decision written down.
4. Run the gates: `audit_tests.py`, ruff, pytest, `diff_guard.py`.
5. Run `mutate_check.py` on each touched function; add tests for survivors.
6. Report: tests added, any test changed or removed and why, mutation score, anything
   skipped with its reason.

## A sanctioned "this test is wrong" exit

Sometimes the test is the bug. An agent with no honest way to say so is pushed toward
gaming. Giving it one works: an escalation tool plus a written policy cut hacking from
23.6% to 5.3% across 8 models, with no change in solve rate, and 99.4% of escalation
reports named the real defect [Esc26 Table 1, Table 2, §4.2]
(`papers/escalation-channels.md`; one author, 9 problems, treat as suggestive). An abort
option cut GPT-5's cheating on ImpossibleBench from 54% to 9% [IB §5.3]. Policy text
alone barely moved test tampering [Esc26 §4.1], so the mechanical diff guard stays.

The Stop gate implements this. Instead of editing a test it believes is wrong, the
agent writes `.claude/pytest-expert-escalation.md`:

```
Issue type: test contradicts spec
Test: tests/test_cart.py::test_discount
Problem: expects 90 for a 10% discount on 100; the spec (docs/pricing.md) says discounts apply after tax
Tried: re-read the spec and both callers
Recommendation: change the expected value to 99, or confirm the spec
```

The gate checks that all five lines are filled, moves the note to
`.claude/pytest-expert-escalations/`, lets the agent stop, and shows the user the note
plus whatever is still red. An incomplete note is sent back with the missing fields.
The block message itself now says this, close to ImpossibleBench's strictest prompt
("if flawed, STOP"), which cut cheating more than any looser wording [IB §5.1].

The policy the agent follows, in three lines:

1. Never edit, skip, or delete a failing test to make it pass.
2. Never special-case code for a test's inputs or hard-code expected outputs.
3. If a test contradicts the spec, escalate with the note above and stop.

## What the hooks enforce

With `assets/claude/` installed (`hooks.md`), step 4's checks run without the agent
choosing to: the post-edit hook audits each test file Claude writes, and the Stop and
SubagentStop gate refuses to end a turn while the audit, the diff guard, or the fast
suite is red. Steps 1, 2, 3, 5, and 6 still depend on the agent and on review; the
hooks make the cheapest forms of gaming visible, not impossible.
