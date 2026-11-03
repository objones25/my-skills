# CoverUp: Effective High Coverage Test Generation for Python

Juan Altmayer Pizzorno (UMass Amherst), Emery D. Berger (UMass Amherst and AWS). Proc. ACM Softw. Eng. 2, FSE, Article FSE128, July 2025. DOI 10.1145/3729398. arXiv:2403.16218 (v4, May 2025). Read from the arXiv PDF: approach (§3) in full, evaluation summaries and ablations (§4), threats (§5).

## The idea

Measure coverage of the existing pytest suite, cut the module into small segments with missing lines or branches, and prompt an LLM with each segment's code, the exact lines still uncovered, and a tool to look up any name it needs. Run the returned tests; if they error, fail, or add no coverage, say so in the same chat and ask again, up to two more times. Keep a test only if it passes and raises coverage. This beats both a search-plus-LLM tool (CodaMosa) and a mutation-plus-LLM tool (MuTAP) on line and branch coverage [CoverUp Abstract, §3].

## Terms

| Term | Meaning |
|---|---|
| Segment | A function, method, or class excerpt (target about 50 lines) that lacks coverage, shown with enough surrounding class context [§3.1, §4.1]. |
| Line+branch coverage | A combined measure counting both executed lines and executed branch outcomes. |
| Continued chat | Feeding errors, failures, or "no new coverage" back to the LLM in the same conversation [§3.3]. |
| Tool function | A function the LLM can call to request the source of a name it does not know [§3.4]. |
| State pollution | A test that passes alone but changes global state (module constants, `sys.modules`) so later tests fail [§3.5, Fig. 10]. |
| SlipCover | The authors' low-overhead Python coverage tool, used to measure each candidate quickly [§3.3]. |
| Ablated CoverUp | Same model, but the prompt gives no coverage detail, no tool function, and no continued chat [§4.1, RQ2]. |

## The method

1. **Measure** coverage of any existing suite [§3].
2. **Segment** code lacking coverage; keep comments, which the authors found help comprehension [§3.1].
3. **Prompt** with the excerpt, the uncovered lines, and instructions such as "include assertions" and "avoid state pollution" [§3.2].
4. **Run and keep or continue.** Save tests that pass and raise coverage; otherwise continue the chat with the error or the missing coverage, at most two more prompts [§3.3, §4.4].
5. **Integration check.** New tests that pass alone may break others. Default fix: run each test in isolation via a pytest plugin. Alternatives: find and disable polluting tests, or disable failing ones [§3.5].
6. **Flaky handling.** Run each new test several times with pytest-repeat; if any run fails, continue the chat so the LLM fixes it [§3.6].

## Key results

- CM suite (425 modules from 35 projects where Pynguin struggles), against CodaMosa: per-module median line+branch coverage 80% vs 47%; overall line+branch 60% vs 49% and 45% for two CodaMosa variants [CoverUp Abstract, §4.2].
- MuTAP's HumanEval-derived suite (163 functions, about 2,000 lines): overall line+branch 89% vs 77% [Abstract, §4.2].
- Against an LLM-only ablation with the same model: overall line coverage 64% vs 39%, line+branch 60% vs 34% [§4.3].
- 60.3% of successes came from the first prompt, 27.2% from the second, 12.5% from the third; continuing the chat gives about 40% of successes [§4.4].
- Removing error-fixing feedback cost 14-37% across coverage metrics; removing code context cost 7-16% [§4.6].
- CoverUp used 48% more tokens than CodaMosa [§4.5].
- Example failure modes shown: a generated test that overwrote module constants and deleted them on exit, breaking other tests [Fig. 10]; a test that assumed a random generator's blood-type labels and failed intermittently [Fig. 11].

## What it means for a pytest suite

- Give the agent the uncovered lines rather than a coverage percentage. CoverUp's prompt names exact missing lines and branches; the ablation without this did much worse [§4.3, §4.6]. `pytest --cov-report=term-missing` provides the list.
- Feed failures back in the same turn. The skill's PostToolUse hook already returns failing-test output to the agent as `additionalContext`; CoverUp's data (40% of successes after the first try) supports keeping that loop [§4.4].
- Run new tests in random order and in isolation, and repeat them. State pollution and randomness in LLM-written tests are common enough that CoverUp built isolation and repetition into its pipeline [§3.5, §3.6]. This matches hard gates 11-13 and `flake_hunt.py`; the post-edit hook could add `-p randomly` and a small repeat count for the edited file.
- Coverage-guided generation is a good first pass, then switch to mutation. CoverUp evaluates only coverage; see Disagreements.

## Threats to validity / caveats

- No mutation score or fault-detection measurement for CoverUp's own tests in this version; "effective" means high coverage. The authors assume the code under test is correct (regression tests) [§1].
- Benchmarks were chosen to be hard for Pynguin; the authors note a different selection could give different results [§5].
- Some benchmark dependencies failed to install, as in CodaMosa's original setup; the authors ignored these failures [§5].
- Main results use gpt-4o-2024-05-13 [§4.1].

## Disagreements

- ACH (`meta-ach.md`): 49% of tests that killed a real-looking mutant added no line coverage, so a coverage-only keep rule like CoverUp's step 4 would discard them.
- Zhao et al. 2026 (`llm-tests-proxy-metrics.md`): coverage tracks bug detection for LLM tests on bug-free code when comparing models, but not on buggy code. CoverUp's regression-test assumption puts it in the first case.
