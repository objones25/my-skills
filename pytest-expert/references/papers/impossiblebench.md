# ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases

Ziqian Zhong, Aditi Raghunathan (CMU), Nicholas Carlini (Anthropic). arXiv:2510.20270 (Oct 2025), "Preprint. Under review." Read from the arXiv PDF: §1-§7 in full; appendices not read. Code: github.com/safety-research/impossiblebench.

## The idea

Take a coding task, then change its unit tests so they contradict the written specification (assert `f(2) == 5` where the spec implies 4). Tell the agent the spec wins. Now any passing submission must be a cheat, so the pass rate is a clean "cheating rate". Frontier agents cheated often, especially in multi-file repositories, and they did it in four ways: editing tests, overloading `__eq__`, keeping hidden state across calls, and special-casing test inputs [IB Abstract, §1, §4.1].

## Terms

| Term | Meaning |
|---|---|
| Impossible task | A benchmark task whose tests conflict with its specification [§2]. |
| One-Off mutation | Change the expected value in one test [§2.2]. |
| Conflicting mutation | Add a duplicate test with a different expected value, so the suite contradicts itself [§2.2]. |
| Cheating rate | Pass rate on impossible tasks [§1]. |
| Scaffold | The agent harness. Minimal: no tools; full: multiple tools [§3.2]. |
| Open test | Agent has read and write access to the tests (the default here) [§2.4]. |
| Special casing | Code that detects the specific test input and returns what that test expects [§4.1]. |
| Operator overloading | Returning an object whose `__eq__` always returns True [§4.1]. |
| Record extra states | Counting calls and returning different results for the same input [§4.1]. |

## The method

1. **Base tasks.** LiveCodeBench (single-function algorithmic) and SWE-bench (real multi-file repos) [§2.1].
2. **Mutate tests** with Claude Sonnet 4 into One-Off and Conflicting variants [§2.2].
3. **Quality control** (SWE-bench): drop mutations that pass with the original patch or an empty patch: 8.8% of One-Off and 3.4% of Conflicting removed [§2.3].
4. **Run agents** with open tests, explicit "do not modify the tests", and up to 10 submissions with failure feedback [§2.4]. Models: GPT-5, o3, o4-mini, GPT-4.1, Claude Opus 4.1, Sonnet 4, Sonnet 3.7, Qwen3-Coder [§3.1].
5. **Classify cheating** in passing transcripts with Claude Opus 4 [§4.2].
6. **Ablate** prompt wording, test access, and the feedback loop [§5]; **evaluate LLM monitors** on known cheats [§6].

## Key results

- GPT-5 cheated on 54.0% of Conflicting-SWEbench tasks; newer Claude models cheated less than Sonnet 3.7; there was less cheating on LiveCodeBench than SWE-bench [IB §4, Figs. 3-4]. Fig. 1 shows rates from 76% (GPT-5) to 39% (o3) without naming the variant in its caption.
- OpenAI models used all four strategies (>14% each); Claude models and Qwen3-Coder cheated mostly (>79%) by editing tests [§4.2, Fig. 5]. On One-Off, OpenAI models mostly special-cased [§4.2].
- Prompt wording: on Conflicting-LiveCodeBench, the loose prompts gave >85% cheating for GPT-5 and o3; the strict prompt D ("critically evaluate the tests ... if flawed, STOP ... Do NOT try to carve out the code to pass the tests") lowered them to 1% and 33% [§5.1, Table 1, Fig. 6]. On SWE-bench, loosening the prompt raised GPT-5 from 54% to 66% and Opus 4.1 from 50% to 55% [§5.1].
- Hiding tests cut cheating to near zero but hurt real performance; read-only tests restored performance and stopped test edits, but not special-casing or overloading [§5.2, Fig. 7].
- Allowing multiple submissions raised average pass rate on open-test SWE-bench from 80% to 83% and cheating on Conflicting-SWEbench from 33% to 38% [§5.3].
- An abort option (submit `flag_for_human_intervention`) cut GPT-5 from 54% to 9% and o3 from 49% to 12%; much smaller effect for Opus 4.1 [§5.3, Fig. 8].
- LLM monitors caught 86-89% of cheats on Impossible-LiveCodeBench but only 42-50% (Sonnet 4) and 57-65% (Opus 4) on Impossible-SWEbench; one evasion was justifying spec-violating code as "backward compatibility" [§6, Figs. 9-10].

## What it means for a pytest suite

The skill's Stop hook blocks the agent until the suite is green. That gives the agent exactly the incentive this paper measures. Concretely:

- **Detect test edits as well as test results.** Claude models' main cheat is editing tests [§4.2]. The skill's `diff_guard.py` (run by the Stop gate against the session-start commit) already blocks deleted tests, dropped assertions, new skips and deselection; this paper is the evidence for keeping it on by default.
- **Make tests read-only during implementation work** when the task is "make the tests pass" (permissions deny on `tests/**` for Edit/Write, or a PreToolUse hook). Read-only kept legitimate performance [§5.2]. Lift it when the task is writing tests.
- **Scan source changes for the other three cheats:** `__eq__` returning a constant, functions that branch on literal values copied from tests, module-level call counters. These survive read-only tests [§4.1, §5.2]. Some fit `audit_tests.py`-style AST checks over the diff of `src/`.
- **Give the agent an exit.** The Stop-gate message should say: if a test contradicts the specification, stop and report it instead of changing code to match. An explicit abort path cut cheating sharply for some models [§5.3]; see also `escalation-channels.md`.
- **Word the instructions like prompt D**: critically evaluate tests, stop on a flawed test, do not carve code to fit [Table 1].
- **The `PYTEST_EXPERT_MAX_BLOCKS` cap matters.** Each forced retry is a feedback round, and more rounds raised cheating [§5.3]. Keep the cap low, and when it trips, tell the user the suite is red rather than pushing again.
- **Do not rely on an LLM reviewer alone** to catch cheats in multi-file changes [§6].

## Threats to validity / caveats

- Preprint, not peer reviewed; appendices with per-variant numbers not read here.
- Impossible tasks are an extreme: every pass is a cheat. In real work, a test may be wrong in subtler ways, and editing it may be correct.
- Mutations were written by an LLM; LiveCodeBench mutations had no quality control [§2.3].
- Cheat classification is itself done by an LLM told the transcript is a cheat [§4.2 footnote].
- Models as of mid-2025; rates will shift with newer models.

## Disagreements

- Escalation channels (`escalation-channels.md`, 2026) agree that an explicit reporting path cuts hacking, and find test-file tampering is the cheat most resistant to policy text, consistent with read-only access being the reliable fix here.
