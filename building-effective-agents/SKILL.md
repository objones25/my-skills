---
name: "building-effective-agents"
description: Diagnose and design AI agents and their harnesses, covering agent vs workflow, tool and MCP server design, context engineering and compaction, long-term memory, verification, security and prompt injection, multi-agent orchestration, long-running state, testing and evals (including Inspect AI), framework choice, and cost. Framework-agnostic (Claude Agent SDK, OpenAI Agents SDK, LangGraph, CrewAI, Google ADK, Pydantic AI, MCP, hand-rolled), with runnable scripts. Use when the user describes an LLM that calls tools, loops, or orchestrates other LLM calls, even without saying "agent." Trigger on "should this be an agent or a script", "my agent says it succeeded but didn't", "it runs out of context", "it forgets", "it picks the wrong tool", "how many agents", "it followed instructions in a web page", "how do I resume a long run", "how do I eval or test this", "why is it slow and expensive", "which framework". For writing an MCP server's code, use mcp-builder.
---

# Building Effective Agents

Reference material for agent design decisions. Diagnose first, then load only the reference file that covers the broken part.

## How to use this

1. **Diagnose before prescribing.** Name the broken surface (model, tools, or harness), list the candidate causes, and say which one you'd bet on and why. A confident single fix for an undiagnosed problem is how people end up prompt-tuning for a week.
2. **Load only the reference file that covers the broken part** (table below).
3. **Lead with the verdict.** "No, don't split this yet" or "Stop editing the system prompt" in the first sentence, then the reasoning, then the cheapest test that distinguishes the candidates. Never open with a taxonomy.

The answer is done when it names the surface, the candidate causes, the bet, and the test that separates them.

**Answer the question that was asked.** This skill covers more ground than any one question needs. Pulling in unrelated advice (tool design on an eval question, sanctioned-failure outcomes on an architecture question, context compaction on a tool-naming question) makes the answer longer and worse. Raise a topic only when the user's reported symptom calls for it, and say the relevant thing once.

**Use the user's facts, not this skill's examples.** The tool names, numbers, and scenarios here are illustrations. Don't restate them as though they describe the user's system, and don't invent specifics they didn't give, such as a count of cases, a tool name, or a failure rate. Ask, or say "assuming X," instead. When inventing an example to illustrate a point, mark it as one so it can't be mistaken for a finding about their system.

**Keep numbers and quotes attached to their source.** Every figure here comes from a named, dated publication listed in `references/sources.md` and reflects one team's workload, not a spec. Cite it as a starting point to measure against. Never invent a figure, reword a quoted phrase while still framing it as a quotation, or point a measurement at a comparison its source didn't make. A multiplier measured against chat usage says nothing directly about the marginal cost of a change to a system that is already an agent.

## Reference files

Load the one that matches the problem, occasionally two when the symptom table names two. Do not load all of them.

| Read | When the question is about |
|---|---|
| `references/tools.md` | Tool design, wrong-tool selection, tool responses flooding context, tool evals |
| `references/context.md` | Long sessions degrading, compaction (masking vs summarization), retrieval, notes files, subagent isolation, system prompt structure |
| `references/verification.md` | Agents claiming false success, no way to tell if a run worked, guardrails, human approval, permissions |
| `references/evals.md` | Benchmarking, graders, pass^k, benchmark validity, Inspect AI, "how do I know a change helped" |
| `references/multi-agent.md` | One agent vs many, equal-budget evidence, failure taxonomy, orchestration shapes and infrastructure, subagent task design |
| `references/production.md` | Long-running state, resumption, deploys, tracing, cache hit rate, where durable value sits |
| `references/security.md` | Prompt injection, untrusted content, credentials, data exfiltration, why detectors fail under adaptive attack |
| `references/memory.md` | Remembering across sessions, stale facts, memory stores, whether memory is worth adding |
| `references/mcp.md` | MCP-specific design, spec 2026-07-28 changes, tool poisoning, MCP security |
| `references/frameworks.md` | Choosing or comparing SDKs/frameworks, their caps, approval/resume, state, hooks; provider context features |
| `references/testing.md` | Deterministic tests for harness code (caps, approval, resume, budgets) with a scripted model |

`references/sources.md` lists every source with its date, URL, and the claims it supports. Check it before quoting a figure.

## Is this an agent's job at all?

An agent is an LLM that directs its own process in a loop, choosing what to do next from what it observes. If step 2 always follows step 1, that's a workflow (predefined code paths), and workflows are cheaper, more predictable, and easier to debug. Anthropic recommends "finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all." (*Building effective agents*, Dec 2024)

Build an agent when at least one holds (OpenAI, *A Practical Guide to Building Agents*):

- The task needs judgment that resists enumeration: exceptions, edge cases, gray areas.
- The rules have grown unmaintainable, so updates are costly or error-prone.
- The input is mostly unstructured: documents, natural language, messy logs.

Otherwise take the lightest pattern that works:

| Pattern | Use when | Don't when |
|---|---|---|
| Single LLM call | One well-defined transformation | Steps depend on each other |
| Prompt chain | Fixed, decomposable subtasks | The sequence varies by input |
| Router | Distinct input categories | Classification is unreliable |
| Parallel fan-out | Independent subtasks, or votes raise confidence | Subtasks are coupled |
| Orchestrator–workers | Decomposition unknown until runtime | The decomposition is static |
| Evaluator–optimizer | Criteria are clear *and* iteration demonstrably improves output | First pass already clears the bar |
| Autonomous agent | Open-ended, unpredictable step count, trusted environment | Errors are costly and hard to detect |

When someone asks for an agent but describes a script, say so and name what they're buying: latency, token spend, and a new failure surface. Build the deterministic parts deterministically even inside an agent.

## The three surfaces

Every agent is a model, its tools, and the harness around them. Name which one is broken before changing anything.

```
HARNESS   loop · context management · state · guardrails · verification · permissions
   └── MODEL  ⇄  TOOLS        model reasons and decides; tools act and observe
```

- **Model**: if it's weak, scaffolding won't save it. Baseline on the strongest model available, get it working, then test whether cheaper models hold for narrow sub-calls. Keep the loop-steering model strong.
- **Tools**: the agent-computer interface deserves the effort a human interface gets. See `references/tools.md`.
- **Harness**: everything that keeps runs grounded: verification, context management, state, guardrails. See `references/verification.md` and `references/context.md`.

Every harness component encodes an assumption about what the model can't do, and those assumptions expire. Context resets that Sonnet 4.5 needed became dead weight on Opus 4.5 (Anthropic, *Scaling Managed Agents*, Apr 2026). On each model upgrade, remove components one at a time and keep only those that still move the eval.

## Diagnose by symptom

| Symptom | Usual cause | Start here |
|---|---|---|
| Claims success on a failed task | Nothing checks the claim | `verification.md` |
| Picks the wrong tool | Overlapping tools, too many loaded at once, vague descriptions | `tools.md` |
| Insists data doesn't exist when it does | Response format the model mis-parses | `tools.md` |
| Degrades over long sessions | Context rot | `context.md` |
| Loops or drifts late in a task | No step bounds, no progress artifact | `context.md`, `verification.md` |
| Slow and expensive | Round-trips, oversized tool results | `tools.md` |
| Fabricates when data is missing | No sanctioned way to give up | `verification.md` (sanctioned failure outcome) |
| Acts on instructions found in a document, web page, or tool result | Untrusted content and authority to act in one session | `security.md` |
| Worked in the demo, fails in prod | Capability shipped without reliability | `evals.md` |
| Unclear whether a change helped | No evals | `evals.md` |
| Uses an outdated fact or preference | Memory without revocation | `memory.md` |
| Got worse after adding memory or summarization | Retrieved or summarized context crowding out the task | `memory.md`, `context.md` |
| Behaves differently after an MCP server update | Tool description changed (poisoning or drift) | `mcp.md`, `security.md` |
| Costs jumped after a context change | Prompt-cache prefix broken | `production.md` |
| Multi-agent version "wins" but costs more | Unequal compute budget in the comparison | `multi-agent.md` |
| Harness change quietly broke behavior | Code bug or prompt shift with no test or per-change eval | `testing.md`, `evals.md` |

"Prompt harder" is rarely the fix. Changing a tool's response shape, adding a deterministic verify step, or swapping the model driving the loop usually is.

## Capability is not reliability

Capability is "it can do this sometimes" (pass@k). Reliability is "it does this every time" (pass^k). Agents demo on capability and fail in production on reliability, and closing that gap is engineering around a stochastic component rather than a modeling problem. A one-in-five failure rate is not "80% working"; for anything user-facing it's a broken product. When a user reports a working demo and a failing deployment, go to verification and evals, not to the prompt. Pass^k covers consistency only; `evals.md` lists the other reliability axes.

## Red flags

- **Success reported without evidence.** Assume it will happen; add a deterministic check.
- **Fine-tuning before tools, harness, and evals are fixed.** OpenAI positions its agent fine-tuning (Agent RFT) after tool and prompt fixes, and RL fine-tuning needs a grader, so evals come first.
- **Tool responses dumped at full verbosity.** Truncate, paginate, choose a format.
- **No sanctioned way to say "I can't."** An agent that must always answer will confidently answer wrong.
- **Irreversible actions with no human checkpoint.**
- **Untrusted input, private data, and an outbound channel in one session.** That is the setup prompt-injection exfiltration needs (`security.md`).
- **Multi-agent reached for before a single agent hit a named wall.**
- **Agent personas standing in for a reason to split.** "Frontend agent, QA agent" is an org chart. Context isolation and independent judgment (an evaluator that didn't do the work) are reasons.
- **A "harness" that is only more prompting.** Verification, guardrails, and state belong in code.
- **No evals.** See `evals.md`.
- **A loop cap nobody set.** Several framework defaults allow hundreds or thousands of model calls (`frameworks.md`).
- **Memory that only appends.** Old facts stay retrievable and win (`memory.md`).
- **A prompt-injection defense validated only against static attacks** (`security.md`).

## Scripts, examples, and tests

Run scripts from the skill root; each prints usage with `--help` or no arguments. They use only the standard library except the Inspect template.

| Script | Use it to |
|---|---|
| `scripts/agent_loop.py` | Show or copy a minimal harness: step and token caps, `finish` tool with sanctioned failure, approval gate, verify hook, budgeted tool results, append-only event log with resume. `--demo` runs offline |
| `scripts/tool_budget.py` | Cap and page tool output with a visible "call again with offset=N" hint |
| `scripts/tool_lint.py` | Screen tool definitions for overlap, missing boundaries, vague params, unannotated side effects, catalog size |
| `scripts/rule_of_two_audit.py` | Check a session manifest for untrusted input + private data + outbound action in one session |
| `scripts/triage_transcripts.py` | Sort "claimed success" runs into never-called / silent failure / empty result / wrong id / retry / verified |
| `scripts/reliability.py` | pass@k, pass^k, paired bootstrap comparison of two versions, eval run budgets |
| `scripts/cache_prefix_check.py` | Find what breaks prompt-cache reuse between consecutive requests |
| `scripts/inspect_eval_template.py` | Inspect AI task with outcome-plus-evidence scoring and pass@k / pass^k; `--smoke` runs offline |

`examples/` holds worked diagnoses (inputs, the reasoning, the script output, the fix). They are illustrations with invented systems; never present them as findings about the user's system. `tests/test_scripts.py` covers every script (`python -m pytest tests -q`); its fixtures double as example inputs.

When a script would answer the user's question faster than prose (sizing an eval, checking a manifest, sorting transcripts), run it on their data or offer it. Don't paste script output the user didn't ask for.

## Sources

Every source, with its date, URL, how it was checked, and the claims it supports, is in `references/sources.md` (last verified 2026-11-02).
