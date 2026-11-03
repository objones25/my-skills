# Running agents in production

## State, failure, and resumption

Agents are stateful and long-running, so errors compound and restarts are expensive. "We can't just restart from the beginning" (Anthropic, *How we built our multi-agent research system*, Jun 2025). Build resumption from the point of failure.

- Keep execution state (step, retries, pending calls) and business state (what the run is *about*) somewhere durable, not only in the context window. A run should be reconstructible from storage.
- Keep an append-only session log outside the context window and build each model call's context as a view over it. A crashed harness can resume from the last event, and compaction stays recoverable because the log keeps everything (Anthropic, *Scaling Managed Agents*, Apr 2026). Google's ADK uses the same split between durable sessions and per-call working context.
- Support launch / pause / resume as ordinary API operations so a human approval, a rate limit, or a crash doesn't destroy the work. Check that pause state serializes: a run waiting overnight for approval can't live in a Python callback. Frameworks differ a lot here (`frameworks.md`); workflow engines such as Temporal, DBOS, or Prefect cover the hardest cases.
- Make side-effecting tools idempotent (keys derived from run and step) so a resumed run can't send the email twice. `scripts/agent_loop.py` shows resume from an append-only log, including answering a tool call that was issued but never completed.
- Persist progress artifacts as the run proceeds (`context.md`), so a fresh context can pick up with the plan intact.
- Return errors compactly (format in `tools.md`), but don't erase them: failed attempts left in context help the model avoid repeating them (Manus, Jul 2025).

## Deploying without breaking in-flight runs

Anthropic uses **rainbow deployments** (shifting traffic gradually from old to new while both versions serve) because a mid-flight agent whose backend changes under it fails in ways that are hard to attribute. Plan for versioned prompts and tools, and for runs that outlive a deploy.

Treat harness, prompt, and default-setting changes as risky deploys too. In Anthropic's April 2026 Claude Code postmortem, three unrelated harness changes (a default effort level, a context-clearing bug, one system-prompt line) combined into what users experienced as one quality drop; the response was per-model evals on every prompt change, soak periods, and gradual rollout (`testing.md`).

## Observability

Agents are "non-deterministic between runs, even with identical prompts," so outcomes alone can't explain behavior. Trace full runs: decisions, tool calls and results, token spend, and where time went. Read the traces where policy allows. Anthropic's Research team monitored decision patterns and interaction structure without reading conversation contents, but as a privacy constraint, not a better method. Alert on leading indicators (tool error rate, turn counts, cost per task), not only on final failures. MCP's 2026-07-28 spec carries OpenTelemetry trace context in request metadata, so tool calls can join the same trace. Most frameworks export OpenTelemetry (`frameworks.md`); prefer that over a vendor-only dashboard so agent traces sit next to the rest of your system's.

## Cost and latency control

- Attribute spend per task, not per month; the useful unit is cost per successful outcome.
- Check oversized tool responses and needless round-trips before switching models (`tools.md`).
- Design for prompt-cache hits: keep the prefix stable (no timestamps at the top), append rather than edit earlier context, serialize deterministically, and avoid adding or removing tools mid-run. Manus calls KV-cache hit rate the single most important metric for a production agent (Jul 2025), and MCP's 2026-07-28 spec asks servers to list tools in a deterministic order for the same reason. Measure hit rate before and after any context-editing change; `scripts/cache_prefix_check.py` finds the first point where consecutive logged requests diverge and names the likely cause.

## Frameworks

Frameworks are fine for prototyping and become the constraint at production time, when a requirement arrives that the abstraction didn't anticipate. End-user credential scoping is a common collision. They can also hide prompts and responses: "incorrect assumptions about what's under the hood are a common source of customer error" (Anthropic, *Building effective agents*).

There isn't much code in an agent loop: model call, tool dispatch, state update, repeat. Stay close enough to it to leave the framework when it stops paying rent. The vendors now say much of this themselves: Anthropic flags its December 2024 post's tooling section as outdated, LangChain 1.0 added middleware because developers kept leaving its agent abstractions, and CrewAI recommends deterministic Flows with autonomous Crews inside.

`frameworks.md` compares the current Claude Agent SDK, OpenAI Agents SDK, LangGraph/LangChain, Google ADK, CrewAI, Pydantic AI, smolagents, and Microsoft Agent Framework on loop caps, approval and resume, durable state, memory, hooks, tracing, and MCP, with a checklist for choosing.

## Where the durable value sits

Almost never in the system prompt. Prompts are copyable and go stale as models change. The parts that are hard to replicate and worth the engineering time:

- Tool integrations and the security and permission model wrapping them.
- The harness that keeps runs grounded and honest, run after run.
- The eval suite that tells you whether a change helped.
- The user experience around the agent: how work is reviewed, approved, and corrected.

Weight design time accordingly, and package procedural knowledge as loadable skills (`context.md`) rather than growing the always-resident prompt.
