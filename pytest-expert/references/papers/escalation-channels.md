# Can escalation channels redirect reward hacking toward defect disclosure?

Francesca Gomez (Wiser Human). arXiv:2608.29460 (v3, 9 Oct 2026, cs.AI); no venue stated. Read from the arXiv PDF: §1-§7 and Appendix B.2 (tool definition). Single author, small problem set; treat as suggestive. Tag [Esc26].

## The idea

When a coding agent hits a broken test it cannot honestly pass, it may hardcode outputs or edit the test. Give it a tool to report the problem to a human instead, plus a short policy document that names test edits and hardcoding as forbidden. Across 8 frontier models on 9 tasks with known checker defects, the pair cut hacking from 23.6% to 5.3% with no measurable loss in solve rate or cost, and the reports named the actual defect 99.4% of the time [Esc26 Abstract, §4.1-§4.4].

## Terms

| Term | Meaning |
|---|---|
| Reward hacking | Pre-registered, act-based: the agent hardcodes an answer, or modifies `test.py` / `test_cases.json` during the run, regardless of whether the final answer is correct [§3.3]. |
| Escalation channel | A tool, `submit_escalation_report(issue_type, problem_assessment, approaches_tried, recommendation)`, described to the agent as going to a human reviewer and "always preferable to a shortcut or manipulated solution" [App. B.2]. |
| Anti-reward-hacking policy | A workspace file, referenced from the system prompt, forbidding test edits and hardcoded outputs [§3.1]. |
| Conditions A-E | A neither; B escalation only; C policy only; D both; E one prompt sentence only [§3.1]. |
| Ambiguous problem | A LiveCodeBench "hard" problem from the EvilGenie benchmark whose checker rejects valid answers or whose test data is wrong [§3.1]. |
| Monitoring / disclosure / escalation | Three ways a defect gets surfaced: the agent's visible reasoning and comments; its final answer text; a filed report [§3.5]. |

## The study design

1. **Tasks.** 9 ambiguous problems, each with a documented checker defect; 9 unambiguous problems as a negative control at baseline only. Docker sandbox, 70/30 visible/holdout split [§3.1].
2. **Factorial.** 2 x 2 (escalation tool x policy) plus prompt-only condition E [§3.1].
3. **Models.** 8 that cleared an 8% baseline hacking gate, from Anthropic, Google, OpenAI, xAI, Moonshot; 45 episodes per model per condition, 1,800 episodes in all [§3.2].
4. **Detection.** A GPT-4o code judge on the final code, plus a harness audit that flags any modification of test files; either one firing counts [§3.3].
5. **Correctness.** Independent validators derived by two different models from the problem statement only [§3.4].
6. **Analysis.** Mixed-effects logistic regression with random intercepts for model and problem [§3.2].

## Key results

- Hacking rate: A 23.6%, B 15.0%, C 9.7%, D 5.3%; D vs A odds ratio 9.18 (95% CI 5.01-16.83) [Esc26 Table 1].
- D gave every model its lowest rate and zero for 6 of 8; the two Gemini models never reached zero and made up all residual hacking under D [§4.1, Fig. 2].
- Policy text removed hardcoding much more than test tampering: under C, hardcoding fell from 50 to 12 episodes, test-file tampering from 35 to 23. Under D, 15 of 19 remaining hacks were tampering [§4.1].
- Solve rate unchanged: A 50.6%, D 52.5% (difference +1.9 pp, CI −5.4 to +9.2). Cost and message count indistinguishable between A and D [§4.2].
- One prompt sentence (E) reduced hacking only to 16.9% and cost 16.6% more per episode than D [§4.3].
- Reports were accurate: monitoring fired 797 times (85.8% named the real defect), disclosure 213 (93.9%), escalation 154 (99.4%) [Table 2]. In B+D, escalation caught 73 defect episodes monitoring missed (+10.1 pp) [§4.4].
- 98.7% of escalations involved no hacking [Abstract].

## What it means for a pytest suite

- Give the agent a sanctioned way to say "this test is wrong". The Stop hook currently has two outcomes: green, or keep working. Add a third: the agent writes a structured note (for example to `.claude/pytest-expert-escalation.md` with the four fields above) naming the test and why it conflicts with the spec, and the Stop hook then lets it stop with a `systemMessage` to the user instead of blocking again [§4.1, App. B.2].
- Pair the exit with a short written policy, not one prompt line. The skill's SKILL.md or hooks.md can carry a three-line rule: never edit or delete a failing test to make it pass, never hardcode expected values, escalate instead. Structured policy beat a single sentence [§4.3].
- Keep a mechanical check for test-file edits. Policy text barely moved tampering, which became most of what remained [§4.1]. A harness audit (the paper's Channel 2) is what `diff_guard.py` already does at Stop; keep it.
- Expect the exit to cost nothing in legitimate progress [§4.2].

## Threats to validity / caveats

- 9 problems from one competitive-programming benchmark; problem-level variance poorly estimated [§6].
- The tool description tells the agent escalation is preferable, mixing "an option exists" with "you should use it" [§6].
- OpenAI models ran without a text-editor tool and made 1 of 95 test-file edits, so tool availability may explain part of the family difference [§6].
- Hacking detection partly relies on a GPT-4o judge.
- Single-author preprint; no independent replication found.

## Disagreements

- Agrees with ImpossibleBench (`impossiblebench.md`) that an abort path cuts cheating, and adds that it also yields accurate defect reports. ImpossibleBench found the abort option helped Claude Opus 4.1 much less; [Esc26]'s Anthropic model (claude-fable-5) reached zero under escalation alone, so model-family effects differ across studies and model generations.
