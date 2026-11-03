# One agent or many, and how to orchestrate them

## Default to one, and make the case for more

Build a single agent with a good toolset first. It covers most real use cases, costs far less, and can be explained to an auditor or an executive, a property that disappears once three agents negotiate an answer.

Measured cost (Anthropic, *How we built our multi-agent research system*, Jun 2025): agents use roughly **4× the tokens of a chat interaction, and multi-agent systems roughly 15×**. Both are measured against chat, so quote them that way. When the user already runs an agent, the relevant comparison is 15× against 4×: a marginal increase of roughly three to four times their current spend, not fifteen (a rough estimate: 15/4, from averages whose task mixes aren't stated). Token usage alone explained **80% of performance variance** in their browsing evals, which cuts both ways: spend buys performance, and upgrading the model can beat doubling the budget. Multi-agent only pays on tasks valuable enough to absorb the multiplier.

Split when you can point at one of these, not because the system feels big:

- The single agent's prompt has accumulated conditional branches faster than it can be maintained.
- Tool selection degrades from overlap, *and* better names and descriptions already failed to fix it (`tools.md`).
- The work spans genuinely distinct domains that don't fit one coherent role.
- The work decomposes into independent chunks where parallelism is worth the coordination cost.

## Where multi-agent wins and loses

Wins on breadth-first work: heavy parallelization, information exceeding a single context window, many complex tools. With a Claude Opus 4 lead and Claude Sonnet 4 subagents, Anthropic measured **90.2% improvement over single-agent Opus 4** on their internal research eval (Jun 2025; the figure is model-specific), and parallelization (3–5 subagents at once, each calling 3+ tools at once) cut research time by up to 90% on complex queries.

Loses on tasks needing shared context across agents, tight inter-agent dependencies, real-time coordination, and **most coding tasks**, which have fewer genuinely parallelizable subtasks than research does.

Independent evidence agrees on the shape. Across 260 configurations, Kim et al. found multi-agent coordination gained up to 80.8% on decomposable financial reasoning and lost up to 70.0% on sequential planning, and architectures without centralized verification propagated errors more (*Towards a Science of Scaling Agent Systems*, arXiv 2512.08296 v3, Apr 2026). Cognition names the mechanism: parallel agents make conflicting implicit decisions about shared work (*Don't Build Multi-Agents*, Jun 2025). Ten months later Cognition reported which multi-agent patterns now work in Devin: a reviewer agent that shares no context with the coding agent, and a manager that delegates to child agents. It still finds that decisions fragment when several agents take write actions (*Multi-Agents: What's Actually Working*, Apr 2026). A workable split: parallel subagents read and explore; one agent owns decisions that touch shared artifacts.

**Compare at equal budget.** Many reported multi-agent gains come from spending more compute, not from the architecture. Holding thinking tokens equal, a single agent matched or beat five multi-agent architectures on multi-hop QA across three model families; multi-agent only became competitive when the single agent's context was heavily corrupted (Tran and Kiela, arXiv 2604.02460, Apr 2026; QA tasks, not tool use). The same paper found API budget-control artifacts that inflate apparent multi-agent gains. A smaller 2026 study on ALFWorld found a team using 1.8x the calls statistically tied with a single agent (arXiv 2609.04217, one 7B model, abstract only). When someone shows a multi-agent win, ask whether the single-agent baseline had the same token budget.

**Know how multi-agent systems fail.** MAST classified 1,642 traces from 7 frameworks into 14 failure modes in 3 groups (specification, inter-agent misalignment, verification), with the most frequent being step repetition (15.7%), reasoning-action mismatch (13.2%), not knowing when to stop (12.4%), and ignoring the task spec (11.8%). Better prompts and a new topology helped only modestly (ChatDev on one benchmark: 25.0 to 34.4 and 40.6), and verification failures stayed high regardless of model (Cemri et al., arXiv 2503.13657 v3; older models, LLM-labeled). The practical reading: explicit termination conditions, a written task spec every agent gets, and an independent verifier fix more than adding agents.

A pipeline whose stages always run in the same order (parse → extract → write) is a workflow. Giving each stage a job title doesn't make it an agent; it multiplies cost and failure surface for a sequence that plain code already expresses. Use subagents there only where context isolation earns its keep, for example letting a subagent chew through a large document and return a compact structured result.

## Shapes

| Shape | Use for | Cost |
|---|---|---|
| Manager / orchestrator (default) | Specialists called as tools, one place enforcing business rules and the final answer | Central coordination overhead |
| Sequential pipeline | Genuinely linear, auditable processes | Rigidity |
| Parallel fan-out + aggregate | Independent angles, or raw speed | Duplicated work when boundaries blur |
| Decentralized hand-off | Open-ended exploration where unpredictability helps; specialists talking to the user directly | Hard to explain after the fact |
| Evaluator–optimizer | Clear criteria where iteration measurably improves output | Wasted passes when the first output suffices |

## Operating rules once multi-agent

- **Specify every subagent task with four fields**: objective, output format, guidance on tools and sources, and explicit boundaries. Without detailed task descriptions, agents duplicate work and leave gaps: told to "research the semiconductor shortage," one subagent explored the 2021 automotive chip crisis while two others duplicated the same 2025 supply-chain searches.
- **Scale effort explicitly in the orchestrator prompt.** A working calibration: simple fact-finding → 1 agent, 3–10 tool calls; direct comparisons → 2–4 subagents with 10–15 calls each; complex research → 10+ subagents with clearly divided responsibilities.
- **Write, then coordinate.** Routing every result back through the orchestrator is a game of telephone that also burns tokens. Have subagents write artifacts to a filesystem or store and pass lightweight references; the orchestrator reads what it needs.
- **Match topology to task structure, not to a headcount.** Decomposable work tolerates fan-out; sequential work does not (evidence above).
- **Incentivize the manager toward the outcome**, not a fixed worker sequence. Hardcoding step order rebuilds a workflow at agent prices and forfeits the adaptability that justified the split.
- **Give the orchestrator room to reason before it plans, and give subagents room to reason between tool calls** so they evaluate results, spot gaps, and refine the next query. Anthropic does this with extended and interleaved thinking; other stacks have their own reasoning settings.
- **Spawn a subagent for isolation or for independent judgment, not for a persona.** Isolation keeps exploration out of the parent context (`context.md`). Independence matters because agents grade their own work leniently, so the evaluator should not be the agent that did the work (`verification.md`).
- **Expect error compounding.** Long-running multi-agent systems fail in ways that single calls don't; see `production.md` for resumption, tracing, and deployment.

## Orchestration infrastructure

The coordination logic is usually simple; keeping it alive is not.

- **Durable execution.** A multi-agent run that dies halfway should resume, not restart. Options range from a checkpointer in the framework (LangGraph, ADK resumability, OpenAI Agents SDK `RunState`) to a workflow engine (Temporal, DBOS, Prefect; Pydantic AI integrates all three). Pick by whether runs must survive process restarts and wait on humans for hours (`production.md`, `frameworks.md`).
- **Deterministic outer flow, autonomous inner steps.** CrewAI's Flows-around-Crews, ADK's Sequential / Parallel / Loop agents, and LangGraph graphs all express the same advice: write the parts that always happen as code, and let agents decide only where judgment is needed.
- **Per-agent limits.** Give each subagent its own turn, token, and time caps in addition to the global ones, so one runaway worker can't consume the budget (Inspect AI, ADK, and the Claude Agent SDK all expose per-agent limits).
- **Cross-organization agents.** The Agent2Agent (A2A) protocol for agents talking to other vendors' agents joined the Linux Foundation's Agentic AI Foundation as a growth-stage project on Aug 27, 2026, alongside MCP, with a stable v1.0 spec. Inside one system you control, plain function calls or a shared store are simpler; A2A matters at organizational boundaries. Everything in `security.md` about untrusted input applies to messages from another organization's agent.
