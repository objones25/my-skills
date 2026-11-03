# 07: "LangGraph or the OpenAI Agents SDK?"

*Illustrative system.* An insurance claims agent: reads a claim, pulls policy data, proposes a payout, and needs a human adjuster to approve anything over a threshold. Approvals can take a day.

## Verdict

Either can do it. The deciding requirement is that a run must pause for a day and resume in another process, so pick on how each serializes paused state and lets you put it in your own database, then on whether you want vendor lock-in for hosted tools.

## How the requirement maps (versions as of 2026-11-02, `frameworks.md`)

| Need | LangGraph 1.2 / LangChain 1.4 | OpenAI Agents SDK 0.24 |
|---|---|---|
| Pause for approval | `interrupt()` in a node, or `HumanInTheLoopMiddleware` | `needs_approval` on the tool; run returns `interruptions` |
| Survive a day and a restart | Checkpointer (e.g. Postgres) keyed by `thread_id`; resume with `Command(resume=...)` | `result.to_state().to_json()` stored by you; `RunState.from_json()`, `approve(i)`, `Runner.run(agent, state)` |
| Gotcha | The interrupted node re-runs from its start on resume, so side effects before `interrupt()` repeat unless idempotent | Deserialize state only from trusted storage (the docs warn about this) |
| Loop cap default | `recursion_limit` 10,007 super-steps; set it | `max_turns` 10; raise deliberately |
| Model choice | Any provider | Built around OpenAI models; other providers via adapters |

## What would change the answer

- Claims logic is mostly fixed steps with one judgment call: write the steps as code (a LangGraph graph, a CrewAI Flow, or plain Python) and make only the judgment call agentic.
- The team already runs Temporal: Pydantic AI's Temporal integration keeps durability in the system they operate.
- They need per-user credentials on every tool call: check each framework's hook for injecting the acting user's token before committing (`production.md`).

## Before committing

Build the approval-and-resume path first, kill the process mid-pause in a test, and confirm the payout tool can't run twice (`testing.md`). That one test exercises the part the frameworks differ on most.
