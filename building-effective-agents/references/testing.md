# Testing the harness

Evals (`evals.md`) ask whether the agent does the task. Tests ask whether the code around the model does what you think: the loop stops at its cap, a denied approval is not retried, a crash resumes without repeating a payment, an oversized tool result gets truncated with a hint. These are deterministic properties of your code, so test them deterministically, with the model replaced by a script.

Anthropic's April 2026 Claude Code postmortem is the case for doing both. Three separate harness changes (a lower default effort, a bug that cleared thinking on every turn, one added system-prompt line about verbosity) looked to users like one broad quality drop; the thinking bug had passed unit tests, end-to-end tests, and review, and the prompt line cost about 3% on an eval. The lessons they drew: run per-model evals and ablations on every prompt change, roll out gradually with soak time, and have staff use the public build (*An update on recent Claude Code quality reports*, Apr 23, 2026). Tests catch the code bugs; evals catch the behavior shifts; neither catches both.

## Layers

| Layer | Model | What it checks | When it runs |
|---|---|---|---|
| Unit | none | Tool functions, budget/truncation, parsers, error formatting, schema validation | Every commit |
| Harness | Scripted (replays fixed turns) | Caps, stop conditions, approval gates, verification hook, resume, event log | Every commit |
| Contract | none | Each tool's schema, error shape, output cap, idempotency; MCP list order | Every commit |
| Replay | Recorded real transcripts | That a harness change doesn't alter how recorded runs are processed (parsing, truncation, logging) | Every commit |
| Smoke eval | Mock or cheapest model | The eval plumbing works: tasks load, scorers score, logs write | Every commit |
| Eval | Real model, several trials | Task success, pass^k, cost, latency | Before merge or nightly |
| Production | Real traffic | Leading indicators, sampled transcripts | Continuously |

## Scripted models

A scripted model is a callable that returns pre-written turns (text plus tool calls). It makes harness tests fast, free, and exact. `scripts/agent_loop.py` includes `ScriptedModel`; Inspect AI's `mockllm/model` accepts `custom_outputs` as a list or a function.

Two rules learned the hard way:

- **Make the script a function of the input when anything runs concurrently.** A fixed queue of outputs breaks as soon as samples or epochs interleave; the outputs arrive at the wrong conversation. This happened while building `scripts/inspect_eval_template.py`.
- **Assert on the event log, not on the model's text.** The test should check which tools ran with which arguments, whether an approval was requested, and what outcome was recorded.

## What to test, concretely

`tests/test_scripts.py` in this skill implements each of these against `scripts/agent_loop.py`; copy the patterns.

- **Stops.** A model that never calls finish ends at `max_steps` with a distinct status; a token budget ends the run too.
- **Sanctioned failure.** `could_not_complete` is accepted and recorded as a normal outcome.
- **Verification.** "Done" with no evidence is rejected and fed back; repeated false "done" ends in a distinct failure status instead of looping forever.
- **Approval.** A gated tool is denied by default, the denial reaches the model as an instructive error, and the tool never ran.
- **Errors.** Bad arguments return the schema and a hint, not a traceback.
- **Resume.** Kill the run after a tool call, resume from the log, and assert the tool did not run twice and the model saw the earlier result. Also test a log that ends with a tool call issued but never answered.
- **Budgets.** A huge tool result arrives truncated with an offset hint; paging through returns every line exactly once.
- **Cache stability.** Consecutive logged requests share their prefix; `scripts/cache_prefix_check.py` flags timestamps, tool reordering, and edited history.
- **Security invariants.** A manifest check (`scripts/rule_of_two_audit.py`) fails CI when a session gains untrusted input, private data, and an outbound channel at once.

## Property and fuzz tests worth having

- Random tool-result sizes and contents never produce a context over budget.
- Random interleavings of crash points always resume to the same final event sequence, with side-effecting tools run at most once (use idempotency keys and check them).
- Random tool-call argument mutations (missing fields, wrong types) always return an instructive error, never an exception that kills the loop.

## Testing changes to prompts and tool descriptions

These are code changes with stochastic effects. Run the eval suite with several trials per task on both versions and compare with a paired interval (`scripts/reliability.py compare`), not a single run. Keep a changelog of prompt and description edits next to eval results so a later regression can be bisected.
