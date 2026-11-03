# Automated Unit Test Improvement using Large Language Models at Meta (TestGen-LLM)

Nadia Alshahwan, Jubin Chheda, Anastasia Finegenova, Beliz Gokkaya, Mark Harman, Inna Harper, Alexandru Marginean, Shubho Sengupta, Eddy Wang (Meta). FSE 2024 industry track (per the PDF's reference line; DOI placeholder "XXXXXXX" in the arXiv version). arXiv:2402.09171 (v1, Feb 2024). Read in full from the arXiv PDF. Industrial self-report; the two LLMs are anonymized as LLM1 and LLM2 [§1].

## The idea

Let an LLM extend an existing human-written test class, then throw away every generated test that does not build, does not pass five times in a row, or does not raise line coverage. What survives is offered to engineers as a diff with its coverage gain. Because every survivor provably builds, passes and adds coverage, hallucinated tests never reach a reviewer [TestGen Abstract, §2].

## Terms

| Term | Meaning |
|---|---|
| Test class improvement | Adding test cases to an existing Kotlin test class, never rewriting existing ones [§1]. |
| Filters | Builds; passes (on every one of five runs); improves coverage. Tests failing any filter are discarded [§2]. |
| Flaky test | Passes on some runs and fails on others in an identical environment [§2]. |
| Implicit oracle | The weakest check: the code should not raise [§5]. |
| Ensemble | Running several LLMs, prompts and temperatures and pooling what passes the filters [§2.1.3]. |
| Diff time / post-land | Recommending tests while the engineer's change is in review vs after it lands [§3]. |

## The method

1. **Pre-process** the existing test class (and optionally the class under test) into a prompt; four prompt templates (extend_test, extend_coverage, corner_cases, statement_to_complete) [§2, Table 2].
2. **Generate** with LLM1 or LLM2 at several temperatures [§3.3].
3. **Builds filter** [§2].
4. **Passes filter.** A test that fails on first run is dropped, because without an oracle the tool cannot tell a found bug from a wrong assertion. A test that does not pass all five runs is flaky and dropped [§2].
5. **Coverage filter.** Keep only tests that raise line coverage over all test classes sharing the build target [§2, §3.3].
6. **Post-process** into a diff with per-test coverage details for review [§3.2].

## Key results

- Reels and Stories evaluation, 86 components: 75% of test classes got at least one new test that builds; 57% one that builds and passes reliably; 25% one that also raises coverage [TestGen §3.3]. The abstract states these as shares of "test cases"; §3.3 says "test classes". Prefer the §3.3 wording.
- First Instagram test-a-thon (Nov 2023): of 17 TestGen-LLM diffs, 16 landed; one was abandoned because the test had no assertion (a "TODO" in its place) even though it raised coverage [§3.2.1].
- Second Instagram deployment, fully automated: 42 diffs, 36 accepted, 4 rejected, 2 withdrawn. Rejections: a test of a getter, two tests that each checked several things, one test with no assertion [§3.4].
- Facebook test-a-thon (Dec 2023): 280 diffs; 144 accepted, 64 rejected or abandoned, 61 unreviewed, 11 withdrawn; over 50% accepted overall, almost 70% of reviewed [§3.4.1].
- Overall the tool improved about 10% of classes it ran on (11.5% in the abstract) with 73% of its recommendations accepted [Abstract, §1].

## What it means for a pytest suite

- Build, pass, and coverage are not enough. Every rejection reason Meta engineers gave maps to an existing audit rule: no assertion (`NO_ASSERT`, hard gate 6), testing a getter ("Not worth testing" list), several behaviors in one test (hard gate 1). Coverage-raising tests with no assertion passed all three of Meta's filters [§3.2.1, §3.4]. Run `audit_tests.py` as a fourth filter on any agent-written test.
- Rerun new tests several times before accepting them. Meta uses five runs [§2]; the skill's PostToolUse hook runs an edited test once. A repeat count of 3-5 for newly added tests (`pytest --count` via pytest-repeat, or `flake_hunt.py --repeat-seed`) is cheap.
- "Discard tests that fail on first run" means the tool can only write regression tests that lock in current behavior, including bugs [§2]. For a bug fix the skill should require the opposite: a test that fails before the fix.
- Prefer extending an existing test file over generating a new one. Meta frames the task as extending a test class, which works even when the prompt holds only the existing test class; adding the class under test helps [§2.1.5].

## Threats to validity / caveats

- Kotlin/Android at Meta; anonymized models; acceptance measured at test-a-thons where engineers were primed to accept tests [§3.2].
- Coverage gain is the only quality filter; the paper reports no mutation or fault-detection numbers for the accepted tests.
- Abstract and body differ on whether 75/57/25% are per test case or per test class [Abstract vs §3.3].

## Disagreements

- ACH (`meta-ach.md`), by an overlapping team a year later, replaces the coverage filter with "kills a mutant" and finds 49% of good tests add no coverage, which implies TestGen-LLM's coverage filter discarded useful tests.
- Zhao et al. 2026 (`llm-tests-proxy-metrics.md`): coverage says little about whether LLM tests expose a bug already in the code, which is exactly the case TestGen-LLM's passes-first filter cannot handle.
