# Evaluating agents

Source for most of this file: Anthropic, *Demystifying evals for AI agents* (Jan 2026), with measurement practice from *How we built our multi-agent research system* (Jun 2025).

## Start before it feels ready

"20-50 simple tasks drawn from real failures is a great start." Early on, effect sizes are large, so a small set detects them. Anthropic's Research team illustrates the scale with a hypothetical: early in development, "a prompt tweak might boost success rates from 30% to 80%" (*How we built our multi-agent research system*, Jun 2025). That is an illustration, not a measured result. Mature agents need bigger, harder sets to find signal. The mistake is waiting until the system is "done": without evals, every change is a vibe and every regression is a surprise.

## Sourcing tasks

- Start from the manual checks already run before each release; those are evals that haven't been written down.
- Mine the bug tracker and support queue, prioritized by user impact.
- Bar for a good task: two domain experts would independently reach the same pass/fail verdict.
- Write a reference solution proving the task is solvable. With frontier models, a 0% pass rate across many trials (0% pass@100) usually means a broken task, not a bad agent.
- Include negative cases. "If you only test whether the agent searches when it should, you might end up with an agent that searches for almost everything." In practice that means tasks where the right behavior is *not* to act: answering from knowledge instead of searching, or saying "we don't have that."

## What to grade

**Outcome over transcript.** "It's often better to grade what the agent produced, not the path it took." Check the final state: the ticket exists with the right fields, the file parses, the migration applied. Grading the exact tool sequence penalizes better solutions and produces brittle tests. Use transcript grading when efficiency, turn count, or reasoning quality is itself what's under test. For multi-turn work, define discrete checkpoints where specific state changes should have occurred.

## Choosing graders

| Grader | Good for | Cost |
|---|---|---|
| Code-based | Deterministic outcomes: tests pass, state changed, parameters correct, string/regex match | Fast, cheap, reproducible, easy to debug |
| LLM-as-judge | Tone, completeness, reasoning quality, anything rules can't reach | Non-deterministic, costlier, needs calibration |
| Human | Calibrating judges, expert consensus | Periodic, not continuous |

Judge practice: give a rubric specific enough that two runs agree, provide an escape hatch to answer "Unknown," and calibrate before trusting it: hand-grade a subset yourself, check the judge's agreement with those labels, and recalibrate against human experts regularly. An uncalibrated judge doesn't measure the agent, it measures itself. Note an unresolved disagreement in the sources: the 2025 multi-agent report found a single judge call scoring all dimensions (0.0–1.0 plus pass/fail) most consistent and best aligned with human judgment, while the 2026 evals guidance recommends an isolated judge per dimension. Test both against your rubric rather than assuming.

## Metrics beyond accuracy

Track tokens and cost per task, latency (time to first token, total), turn and tool-call counts, and error rates. For non-deterministic agents report both:

- **pass@k**: at least one of k attempts succeeds. Use where a retry is acceptable and one success counts.
- **pass^k**: all k attempts succeed. Use for customer-facing reliability claims.

Pass^k measures consistency only. Princeton's reliability work adds robustness (stability under perturbed inputs and environments), predictability (calibrated confidence), and safety (bounded severity when failures happen), measured with 12 metrics across 15 models; recent capability gains bought only small reliability gains (Rabanser et al., *Towards a Science of AI Agent Reliability*, arXiv 2602.16666 v3, Jun 2026).

Build in partial credit: identifying the problem but failing to complete the action is better than total failure, and collapsing both to 0 hides progress.

**Don't read a single-run delta as signal.** Agents are non-deterministic, so one pass over one task set gives a point estimate with no error bars, and a handful of flipped cases is as likely to be noise as improvement. Repeat each task a few times per version and compare distributions, or restrict claims to differences large enough to survive the variance you observe. Cost the plan before committing to it: tasks × repeats × versions is the number of agent runs, and it grows faster than people expect.

## Running them honestly

- **Isolate every trial** from a clean environment. Leftover files, cached data, and resource exhaustion cause correlated failures that violate independence and make aggregates lie. Shared state can also inflate scores: Anthropic saw Claude gain an unfair edge in some internal evals by reading git history left by earlier trials.
- **Read the transcripts.** "You won't know if your graders are working well unless you read the transcripts." Most grading bugs (floating-point comparisons, self-contradictory instructions, tasks requiring exact reproduction of stochastic output) surface only there.
- **Watch for saturation.** Near-100% pass rates mean the set has stopped discriminating; graduate it to a regression suite and write harder tasks.
- **Make graders resistant to shortcuts.** Passing should genuinely require solving the problem, not exploiting an unintended loophole.
- **Simulate the user for conversational agents.** A second model plays the user, as in τ-bench and τ2-bench.
- **Layer the defenses** (no single layer catches everything): automated evals on every commit, production monitoring after launch, A/B tests once traffic allows, weekly transcript sampling, periodic human studies to recalibrate judges.

## Is the benchmark itself valid?

Public agent benchmarks and home-grown suites share the same failure: a score that doesn't measure the task.

- The ABC checklist audit of 10 agent benchmarks found task-validity flaws in 7, outcome-validity flaws in 7, and reporting gaps in all 10; on τ-bench airline tasks a do-nothing agent passed 38%, and one coding benchmark could be scored 100% by overwriting its tests (Zhu et al., arXiv 2507.02825 v5).
- BenchJack reached near-perfect scores on most of 10 benchmarks without solving any task, and its patching loop cut the hackable share from near 100% to under 10% on four of them (Wang et al., arXiv 2605.12673, abstract only).
- HAL ran 21,730 rollouts (about $40k): higher reasoning effort gave equal or lower accuracy in 21 of 36 runs, scaffold choice moved cost 9x for 2 points of accuracy, and agents sometimes found gold answers online (Kapoor et al., arXiv 2510.11977).
- Rankings of fixed model+scaffold systems were reliable (0.935 to 0.994) while rankings of the underlying models were not (0.148 to 0.841), and adding many similar tasks barely helped (Hardy et al., arXiv 2610.00651, abstract only). Compare whole systems as you'll ship them; don't conclude "model A is better" from one scaffold.

Before trusting a suite: run a do-nothing agent and a trivial agent against it, check that passing requires the real outcome (a grader that reads the agent's claim is not a grader), close shortcuts like visible answers or writable tests, and record environment versions so drifting websites or APIs don't masquerade as regressions.

## Running evals with Inspect AI

Inspect AI (UK AI Security Institute; `inspect_ai` 0.3.287 as of 2026-11-02) is an open-source harness that covers most of this file's advice in code:

- `Task(dataset, solver, scorer, sandbox, epochs, message_limit, token_limit, time_limit, ...)`; the built-in `react()` agent loops until the model calls `submit()`.
- `epochs=Epochs(n, [...])` repeats each sample n times and reduces with `pass_at(k)` (at least one of k) or `pass_k(k)` (all k, the pass^k estimator), alongside `mean`, `at_least(k)`, and others.
- `sandbox="docker"` runs `bash()` / `python()` tools in a container per sample, which gives trial isolation for free.
- Custom `@scorer` functions can read the full `TaskState`, including tool calls, so you can grade outcome and evidence together; `model_graded_qa()` covers rubric grading.
- Approval policies gate tools during evals the same way production does; `agent_bridge()` and `sandbox_agent_bridge()` evaluate an agent built on another framework, or a CLI agent, without rewriting it.
- `inspect view` opens transcripts; `read_eval_log()` loads them for analysis.

`scripts/inspect_eval_template.py` is a working task with a negative case, an outcome-plus-evidence scorer with partial credit, and pass@k / pass^k reducers. Its `--smoke` mode runs offline against Inspect's `mockllm` model. One gotcha found while building it: samples and epochs run concurrently, so a mock that replays a fixed list of outputs gets them in the wrong order. Make the mock a function of the conversation instead.

## Arithmetic helpers

`scripts/reliability.py` computes pass@k and pass^k from trial counts (same estimators as Inspect), the per-attempt success rate a pass^k target implies (pass^8 of 0.9 needs about 0.987 per attempt), a paired bootstrap interval for comparing two versions across tasks, and the run count and cost of an eval plan.

## Evaluating a fix

When a user has just fixed something, close the loop: capture the failing cases as tasks, re-run before and after, and report the delta. A fix asserted without a re-measure is the same error as an agent asserting success without evidence.

## Evaluating tools specifically

Tool-level evals use realistic multi-call tasks and measure selection accuracy, call counts, and tokens; see `tools.md`.

## Evals versus tests

Evals measure a stochastic system's behavior. The harness around it (budgets, approval gates, resume, verification) is ordinary code and should have ordinary deterministic tests with a scripted model; see `testing.md`.
