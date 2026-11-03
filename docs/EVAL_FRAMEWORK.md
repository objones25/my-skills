# Evaluation framework: investigation and recommendation

Date: 2026-11-03. This is an investigation with a recommendation. Nothing beyond small pilots has been built.

## Recommendation

**Probably do not build a new eval runner.** Two official tools look able to do the hard parts. The likely shape is a thin layer inside `skill-builder` that runs them with safe settings, converts your existing eval files, and keeps a dated results history. For triggering, the pilot supports this on one skill and one machine. For output quality it is a judgment that still needs a real trial: see "Suggested order".

| Layer | Question it answers | Tool | Status |
|---|---|---|---|
| 0. Conformance | Is the skill well formed? | `check_skill.py` | Built |
| 1. Triggering | Does the description win when it should? | skill-creator's `run_eval.py` | Ran in a pilot; needs a wrapper with safe settings |
| 2. Output quality | Is the work better with the skill? | `claude plugin eval` | Runs, but quality measurement is untested; needs an adapter |
| 3. History | Is it getting better or worse? | A results log | Not built |

The main reason to build this at all: **skill quality has no repeatable, recorded measurement today.** One-off methods exist, but the audit found that no skill measures triggering and none keeps eval results. Every "successful pattern" in `SKILL_AUDIT.md` is inferred from prevalence, not outcomes. This framework is what could test them.

## What the pilots found

All runs below were on 2026-11-03 on macOS arm64 with Claude Code 2.1.317 and model `claude-sonnet-5-5`, using `skill-builder` at the time of the run. They use small samples, so treat rates as rough.

### Triggering, with skill-creator's `run_eval.py`

The same runner gave very different answers depending on how it was run:

This runner does not load the skill. It writes a stub command containing only the description, under a temporary name, and counts a hit if Claude calls that stub first. So it tests the description, not the skill's body or install path. The skill must not also be installed, or it competes with its own stub; in this pilot it was not.

| Workers | Timeout limit | Runs per query | Queries | Should-trigger runs that fired |
|---|---|---|---|---|
| 4 | 60 s | 1 | All 18 | 2 of 9 |
| 4 | 150 s | 2 | 4 | 2 of 8 |
| 1 | 60 s | 2 | 4 | 6 of 8 |
| 1 | 150 s | 3 | 4 | 12 of 12 |
| 1 | 150 s | 3 | All 18 | **24 of 27** |

In that last run, none of the 27 should-not-trigger runs fired.

- **Concurrency is strongly associated with misses.** Across these runs, 4 of 17 fired at 4 workers, and 42 of 47 at 1 worker. The runner's default is 10 workers. The likely cause, from reading `run_eval.py` and not tested in isolation: workers share one commands folder, so a worker's Claude can pick another worker's stub, which counts as a miss. Use one worker.
- **No run was limited by its timeout.** The whole 60-second run took under a minute, so the timeout comparison was not really tested, and the table's differences between the 1-worker runs are not evidence about time. The runner's 30-second default was not tried.
- **Noise is visible.** At one worker, "write me a new skill..." fired 0 of 2 in one run and 3 of 3 in others. One direct run of it loaded `superpowers:writing-skills`, so competition from other installed skills is a likely cause. That would differ on another machine.
- **The working directory matters.** The runner walks up to the nearest `.claude` folder and writes temporary commands there. Run from its own folder, that would be `~/.claude/commands`, so run it from a throwaway project with its own `.claude`.
- **Failures look like misses.** The runner discards stderr and treats errors and timeouts as not triggered, and any other first tool also counts as a miss. A total outage would "pass" every should-not-trigger case. A wrapper should detect this.
- **One apparent miss may be a bad case:** "turn the process we just went through into a reusable skill" fired 0 of 3, but a single `claude -p` has no earlier conversation for it to refer to.

### Output quality and a clean environment, with `claude plugin eval`

It needs a plugin, so a standalone skill was wrapped in a temporary one by adding `.claude-plugin/plugin.json` and copying the skill into `skills/`, with its `evals/` folder removed from the copy. The cases granted no write or shell tools, used `max_turns: 6`, and were scored by one regex for the word "trigger" in the final message. So this shows the tool runs, not that it measures quality.

| Prompt | Runs per arm | Skill tool called (with skill) | Score with skill | Score without |
|---|---|---|---|---|
| "I want to make a Claude Code skill for reviewing Terraform plans, can you help me build it?" | 3 | 3 of 3 | 0 of 3 | 2 of 3 |
| "...What should I do first, and what files will it need? Answer briefly, and do not create any files." | 3 | 0 of 3 | 3 of 3 | 3 of 3 |

The suite passed 1 of 2 cases, with a mean delta of -0.33.

- **Do not read that delta as "the skill makes things worse".** The cause of the lower score is unknown. No run recorded a turn-limit error, two of the three with-skill runs stopped under the limit, and the traces no longer exist. The case granted no shell or write tools, so the skill could not run its scripts, and its first step is scoping, which may not mention the word the regex looked for. Read the transcripts before believing a delta, and keep them.
- **A prompt's wording can stop the skill firing.** The second prompt fired 0 of 3, but it also said to answer briefly and create no files, so phrasing and that instruction are confounded. A plain question, "what is my pytest skill missing?", fired 3 of 3 under `run_eval.py`. A question-form case has been added and not yet run.
- **The grader counted any Skill call,** not this skill specifically. Use `input_match`. In a two-arm run, this grader is also excluded from the score.
- **A regex grader using `(?i)` errors out and counts as failed.** The documented way is `flags: i`.
- **Cost was low in this pilot, and is a poor predictor.** The 12 runs in the second suite were estimated at $0.55 at list prices, or about $0.03 to $0.08 per run. These were short runs with no write or shell tools and no LLM grader.

## Comparing the options

| | skill-creator `run_eval.py` | skill-creator benchmark workflow | `claude plugin eval` | Build our own |
|---|---|---|---|---|
| Triggering | Yes | No | Yes, through `tool_used: Skill` | Would have to write |
| Output quality | No | Yes | Yes | Would have to write |
| No-skill baseline | n/a | Yes, driven by the agent | Yes, by default | Would have to write |
| Runs headless, for scripts | Yes | No, it is agent-led | Yes | Yes |
| Cost ceiling | No | No | `--max-cost-usd` | Would have to write |
| JSON report | Yes, unversioned | `benchmark.json` | Versioned schema | Would have to write |
| Tests with your other skills installed | Yes | Depends | No, isolated | Would have to write |
| Works on a bare skill folder | Yes | Yes | Needs a plugin wrapper | Yes |

The two trigger measurements answer different questions, and both are useful. `run_eval.py` asks "does this description win against my installed skills?", and `plugin eval` asks "does the skill fire by itself?".

The formats are not interchangeable, so layer 2 needs an adapter. Of your 107 structured cases, 84 use `expectations` in `evals.json` and 20 use `assertions`, and all of those are prose that would map onto `llm` graders, which cost more and have their own judging error. The other 3 are already `case.yaml` with graders, in `massively-parallel-rl`. Runs also start in an empty workspace, so cases that use `evals/files/` need their fixtures passed in.

## Proposed design

Add to `skill-builder`:

1. **`run_trigger_evals.py`.** Wraps `run_eval.py`, creating a throwaway project and forcing one worker and at least 3 runs per query. It refuses if the skill is installed, exits 2 if skill-creator is missing, and checks with a known-good query that runs are really working, so an outage is not scored as a pass.
2. **`run_output_evals.py`.** Wraps the skill in a temporary plugin, converts its cases (each expectation becoming an `llm` grader, plus a `tool_used: Skill` grader with `input_match`), passes in fixtures and tool grants, keeps the transcripts, and runs `claude plugin eval` with a required cost ceiling.
3. **`evals/results.jsonl`.** One line per run, with the stamp from `machine_stamp.py`, the skill's commit, the model, Claude Code's version, and the scores. This is what makes a comparison possible.
4. **A compare command.** Reports what changed against the last run, and says so when the difference is within the noise.

### Rules it should enforce

- **Compare like with like.** Results count as comparable only with the same model, judge model, tool grants, machine class, installed skills and tool versions.
- **Report counts, not only rates.** "24 of 27" shows the sample size, while "89%" hides it.
- **Three runs show direction, and prove nothing.** A real claim of improvement needs more runs, and an independent look at the transcripts.
- **Grade from artifacts where possible.** Regex and file checks are free and deterministic, so use `llm` graders only where they are needed.

### Rough cost of running everything

Layer 2 on the 115 committed cases (107 plus `skill-builder`'s 8), at 3 runs and 2 arms, is 690 runs. At the $0.03 to $0.08 per run seen here that is about $21 to $55. **Treat that as a floor**, because the pilot's runs had no shell or write tools and real tasks run longer. If every prose expectation became an `llm` grader, that would also add thousands of judge calls. Layer 1 was not metered. Cost on your plan may also not be per-run. Trial one skill and measure before running everything.

### Suggested order

1. Build layer 1 and run it across all skills, once trigger cases exist. This needs the least code, and it is probably where most of the value is.
2. Add the results log.
3. **Trial layer 2 by hand on one real skill,** with graders, tool grants and fixtures, before writing an adapter. `scientific-computing` needs few fixtures. Whether `claude plugin eval` suits skills that run scripts is unknown, because granted shell tools run in a restricted sandbox. If it does not suit them, reconsider the recommendation.
4. Use the results to test the audit's patterns. For example, compare a skill's trigger results with and without negative scope.

## Open questions

- Should cases run on every commit, on release only, or by hand? Hand and release seem right, given cost and noise.
- How much will `llm` graders disagree with a human? That has not been measured, and it decides how far layer 2 can be trusted.
- The other 17 skills have no trigger cases, and the cases that exist would need a real run on your installed set.

## Not verified

- The mechanism behind the concurrency association (inferred from code; no miss was examined).
- Why the with-skill runs scored lower in the second pilot. The traces no longer exist.
- Any real task with tool grants under `claude plugin eval`.
- Any effect of the timeout, since no run reached its limit.
- Triggering of the installed skill. The runner tests a stub of its description.
- `run_loop.py`, the description optimizer, was read about and not run.
- The skill-creator benchmark workflow was not run.
- `llm` graders, scaffolds and mocks in `plugin eval` were not run.
- Everything here on any machine other than this one.
