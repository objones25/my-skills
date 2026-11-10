# Do Coverage and Mutation Scores of LLM-Generated Test Suites Correlate with Their Effectiveness? (Replicability Study)

Junda Zhao, Shurui Zhou, Eldan Cohen (University of Toronto). Proc. ACM Softw. Eng. Vol. 3, ISSTA, Article ISSTA002, October 2026. arXiv:2607.22880 (v1, 24 Jul 2026). Read from the arXiv PDF: abstract, method (§3), all three RQ summaries, discussion (§7) and threats (§8); detailed result tables skimmed. Tag [Zhao26].

## The idea

Papers on LLM test generation report coverage and mutation score as if they measured bug-finding. For human-written tests, [Ino14] and Papadakis et al. found those correlations largely vanish once suite size is fixed. This study reruns both analyses on suites written by 11 LLMs for 318 Defects4J methods. Result: suite size barely matters for LLM suites; coverage and mutation do track real-bug detection when comparing models on bug-free code; but when the code under test already contains the bug, coverage no longer predicts whether the tests find it, and mutation analysis cannot even be run [Zhao26 Abstract, §7].

## Terms

| Term | Meaning |
|---|---|
| Focal method | The method patched by a Defects4J bug fix; the unit the LLM writes tests for [§3.2]. |
| Bug-free (regression) setting | The LLM sees the fixed method; a test detects the bug if it passes on the fixed version and fails on the buggy one [§3]. |
| Buggy setting | The LLM sees the buggy method and must write tests that expose the bug it contains [§3, RQ2]. |
| Combined / intra-model / inter-model view | Correlations over all suites pooled; within one model's suites; across models with one aggregated point per model [§4]. |
| Average / accumulated aggregation | Per-method metrics averaged, or tests pooled and measured together [§4]. |
| Raw / normalized mutation score | Killed over all mutants; killed over mutants on covered code [§4]. |
| MCC | Modified condition coverage [§4.4]. |

## The study design

1. **Benchmark.** Defects4J v3.0 (854 defects, 17 Java projects). Keep non-private focal methods present in both versions whose buggy version fails at least one developer test: 318 methods [§3.2].
2. **Models.** 11 LLMs from six vendors, base and reasoning variants, 13 settings (for example Claude 4 Sonnet with and without reasoning, GPT-4.1, o4-mini, DeepSeek-V3/R1, Qwen3) [§3.3, Table 2]. One automated prompt template using only the code under test [§8].
3. **RQ1** replicates [Ino14]: size vs mutation, coverage vs mutation with and without fixed size (k = 3, 5, 10), and correlations among coverage types [§4].
4. **RQ2** coverage vs real-bug detection, for suites generated from bug-free and from buggy code [§5].
5. **RQ3** replicates Papadakis et al.: mutation vs real-bug detection [§6]. Mutation is skipped in the buggy setting because it needs a green suite, which would exclude the very tests that expose the bug [footnote 1, on RQ2 in §3.1; printed on the §3.3 page].

## Key results

- Suite size is only weakly correlated with raw and normalized mutation score; more generated tests do not mean more mutants killed [Zhao26 §4.5 (1)].
- Coverage vs mutation is weak in the combined and intra-model views even without size control, but moderate to strong across models under average aggregation [§4.5 (2)].
- Coverage types diverge: statement vs branch r = 0.753 and statement vs MCC r = 0.885 (average, all sizes), but under accumulated aggregation statement vs MCC is r = 0.319 and near zero at k = 3 [§4.4, Table 8].
- Bug-free setting: across models, most coverage metrics correlate moderately (some strongly) with real-bug detection [§5.5].
- Buggy setting: coverage vs bug detection is uniformly weak in all three views [§5.5].
- Mutation vs bug detection: Pearson 0.475-0.511 in the combined view (average aggregation), weak intra-model, moderate to strong inter-model for raw mutation score [§6.1, §6.3].
- Fixing suite size causes only small drops; the authors argue size varies naturally with the method's needs in LLM suites, unlike sub-suites sampled from a project pool [§7 ④-⑤].

## What it means for a pytest suite

- When an agent writes tests for code that might already be wrong, do not accept coverage as evidence the tests are good. The skill's "write the failing test first" rule for bug fixes (Choosing what to test, item 3) is the right check: a test must be shown to fail on the buggy code [§7 ①-②].
- Mutation scores on agent-written tests are meaningful only after the suite is green on code believed correct. `mutate_check.py` already refuses to run on a red baseline; keep that and say why in `mutation-testing.md` [footnote 1].
- Report branch coverage alongside line coverage for agent-written suites; coverage types disagree more for LLM tests than [Ino14] found for human tests [§7 ③].
- More generated tests is not better tests. A hook or review step should not reward test count [§4.5 (1)].

## Threats to validity / caveats

- Java and Defects4J only; one prompt template [§8].
- Inter-model conclusions rest on about 13 data points, which the authors flag as low power [§8].
- The tests come from one-shot prompting of a focal method, not from an agent iterating against a test runner, which is the skill's setting.
- Preprint posted in July 2026 with an ISSTA 2026 venue line; not independently replicated.

## Disagreements

- With [Ino14] and Papadakis et al. 2018 on whether size confounds the proxies: for LLM suites, little evidence it does [§7 ④].
- With [Ino14]'s advice that simpler coverage types suffice [§7 ③].
- Consistent with [Chen20]'s view that stratifying by size distorts results, though the authors explain the difference by how suites are built rather than by citing [Chen20]'s statistical argument.
