# Verification, guardrails, and human checkpoints

## Why agents report false success

A model that fails will often say it succeeded. Agents stop when the work *looks* done, and they grade their own work leniently: Anthropic reports agents confidently praising mediocre output and, as QA, talking themselves into approving bugs they had found (*Harness design for long-running application development*, Mar 2026). Without a check in the harness, the human is the verification loop.

So when someone has tried three phrasings of "do not claim you filed the ticket unless you actually filed it," the diagnosis is not a prompting problem. A prompt is a request; verification is a constraint. More wording will not close it.

Before prescribing, separate the candidate causes; they need different fixes:

1. **The call never happened.** The model narrated an action it didn't take. Fix: check the tool-call record.
2. **The call happened and failed silently.** Many APIs return HTTP 200 with an `errors` array (GraphQL especially), and a client that only checks status codes reports success. Fix: parse the response body, raise on error payloads.
3. **The call succeeded but the reported identifier is wrong.** The work exists under an ID the user isn't searching for, a very different bug that looks identical from the outside. Fix: echo the real identifier from the response, never a model-composed one.
4. **The call was retried and partially applied.** Fix: idempotency keys, then reconcile.

Ask for a handful of raw transcripts sorted into those buckets before committing to a fix. It's a ten-minute test that prevents a week of prompt tuning. `scripts/triage_transcripts.py` does the sorting from logged tool calls and adds a fifth bucket, `empty_result`, for a lookup that returned nothing while the agent reported data.

## The verification ladder

Rules first (Anthropic, *Building agents with the Claude Agent SDK*, Sep 2025):

1. **Defined rules**: tests, type checks, linters, exit codes, schema validation, state assertions, read-back of the created object. Deterministic, debuggable, and it explains which rule failed and why. Prefer this whenever the outcome is checkable.
2. **Visual or rendered feedback**: screenshots, diffs against a fixture, rendered output. Best for layout, styling, hierarchy, responsiveness.
3. **LLM as judge**: flexible and scalable, but "generally not a very robust method" alone and adds latency. Use it where rules can't reach (tone, completeness), with a specific rubric and calibration against humans.

## Design rules that matter in practice

Verification is among the best-supported interventions. Across HAL's 21,730 rollouts, agents that checked their own work were 13 to 87% more likely to succeed (Kapoor et al., arXiv 2510.11977), and MAST found verification failures common across every framework and model it studied (Cemri et al., arXiv 2503.13657). The checks below are cheap compared with either finding.

- **The agent doing the work shouldn't be the one grading it.** Use a fresh context: a verification subagent, a separate evaluator pass, or a hook that refuses to end the turn until a check script passes. Self-review inherits the same wrong assumptions.
- **Agree on "done" before the work starts.** In Anthropic's long-running harness, generator and evaluator negotiated a sprint contract of testable behaviors before any code was written. A separate evaluator earns its cost only when the task is beyond what the model does reliably alone. Out of the box Claude was a poor QA agent; tuning it meant reading its logs where its judgment diverged from a human's.
- **Make the system legible to the agent.** Queryable logs, metrics, and UI let it check its own work; linters and structural tests enforce invariants that prose can't (OpenAI, *Harness engineering*, Feb 2026).
- **Demand artifacts, not assertions.** Command plus exit code, test output, the API response body, a screenshot. "Done" is not evidence.
- **Verify against the tool-call history**, not the model's summary of it. What the calls show is what happened.
- **Put the check in code you control.** Secrets, auth, guardrails, and anything that must not fail don't belong in a prompt and shouldn't be delegated to a framework's abstraction.
- **Report the failure back into context compactly** so the agent can correct (the error, what it means, what to try) rather than a full traceback.
- **Give the agent a sanctioned failure outcome.** If every path must end in "filed," the model will produce "filed." An explicit `could_not_file(reason)` outcome that the harness treats as a valid, non-punished result removes the incentive to fabricate. (Include this when the symptom is fabricated success; skip it when the question is about something else.) `scripts/agent_loop.py` implements it as a `finish` tool with `done` / `could_not_complete`, plus a verify hook that rejects `done` without evidence ids pointing at successful tool results.

## Guardrails

Layer them; no single layer catches everything.

- **Rules-based**: max steps, max messages, token and cost ceilings, blocklists, input length limits, output schema validation.
- **Classifier-based**: relevance, safety and jailbreak detection, PII filtering, moderation.
- **Tool risk tiers**: rate each tool on read vs write access, reversibility, permission scope, and financial impact; gate the high tiers (OpenAI, *A Practical Guide to Building Agents*).
- **Permissions**: allowlist known-safe operations, sandbox filesystem and network where the platform supports it, scope credentials to the acting user rather than a blanket service account, and keep them out of reach of model-run code (`security.md`).

## Human checkpoints

Route irreversible or high-stakes actions (payments, refunds, deletions, destructive migrations, external communications) through human approval until the agent has a track record, and keep the gate permanently for the highest-stakes categories. Set explicit escalation triggers too: retry limits, and escalation after repeated failure to parse user intent.

Human review is also where mental alignment happens. On systems shipping large volumes of agent-generated work, reviewing plans is more tractable than reviewing every output, and it keeps the humans in touch with how the system actually behaves.
