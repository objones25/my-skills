# Frameworks and SDKs

Versions and API names below were checked on 2026-11-02 against PyPI and each project's docs (via Context7), with defaults confirmed in the released wheels where noted in `sources.md`. These libraries release weekly; re-check before quoting a default or an API name.

## How to choose

The loop itself is small: call the model, dispatch tool calls, append results, repeat (`scripts/agent_loop.py` is about 300 lines with resume, approval, and verification). A framework earns its place by what it saves you on the parts that are not small. Rank candidates by the questions your system will actually hit:

1. **Can I cap the loop and see why it stopped?** Every framework below has a turn or call cap; check the default, since several are high enough to burn real money (ADK 500 LLM calls, LangGraph 10,007 super-steps).
2. **Can a run pause for a human and resume later, in another process?** An in-process callback isn't enough; the paused state has to serialize. OpenAI Agents SDK (`RunState`), LangGraph (checkpointer + `interrupt`), Pydantic AI (deferred tools), ADK (resumability config) and Claude Agent SDK (`defer` + session resume) all support it; smolagents has no built-in approval pause.
3. **Where does state live, and can I own it?** Durable checkpoints, session stores, and long-term memory are three different things (`memory.md`). Check that you can put each in your own database.
4. **Can I intercept every model call and every tool call?** Middleware or hooks are how guardrails, budgets, PII filters, and verification get into the loop without forking the framework.
5. **Does tracing export to what I already run?** OpenTelemetry export avoids a second observability stack.
6. **How much of the loop can I override?** The failure mode is a requirement the abstraction didn't anticipate (per-user credentials, a custom stop condition). Read the source of the run loop before committing.

Model-vendor SDKs (Claude Agent SDK, OpenAI Agents SDK, Google ADK) give the tightest integration with that vendor's hosted tools and context features. Model-agnostic frameworks (LangGraph, Pydantic AI, CrewAI, Microsoft Agent Framework, smolagents) cost a little integration for portability.

## Comparison (as of 2026-11-02)

| Framework (version) | Loop cap (default) | Pause for approval, resume | Durable state | Long-term memory | Multi-agent | Tracing | MCP client |
|---|---|---|---|---|---|---|---|
| Claude Agent SDK `claude-agent-sdk` 0.2.176 | `max_turns`, `max_budget_usd` | `can_use_tool` callback; `PreToolUse` hook returns `defer`, resume the session later | Sessions: `resume`, `fork_session`, `session_store` | `AgentDefinition.memory` scopes; auto compaction | `agents=` with `AgentDefinition`, invoked via the Agent tool | OpenTelemetry (traces in beta) | `mcp_servers`, in-process `create_sdk_mcp_server` |
| OpenAI Agents SDK `openai-agents` 0.24.0 | `max_turns` (10) | `needs_approval` on tools; `result.interruptions`, `state.approve(i)`, `Runner.run(agent, state)` | `RunState.to_json()` / `from_json()`; Sessions (SQLite, Redis, SQLAlchemy, ...) | Sessions only | `handoffs`, `agent.as_tool()` | Built-in traces; `add_trace_processor` | Stdio / Streamable HTTP / SSE servers; `HostedMCPTool` |
| LangGraph 1.2.19 / LangChain 1.4.8 | `recursion_limit` (10,007); `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` | `interrupt()` + `Command(resume=...)` with a checkpointer and `thread_id`; `HumanInTheLoopMiddleware` | Checkpointers (in-memory, Postgres, SQLite) | Store (`put`, `search`) | Subagent as tool; `SubAgentMiddleware` (deepagents) | LangSmith | `langchain.mcp` adapter |
| Google ADK `google-adk` 2.13.0 | `RunConfig.max_llm_calls` (500) | `require_confirmation` / `tool_context.request_confirmation`; `ResumabilityConfig` | `SessionService` (in-memory, database, SQLite, Vertex) | `MemoryService` + `load_memory` tool | `sub_agents`, `AgentTool`, Sequential / Parallel / Loop agents, `Workflow` graphs | OpenTelemetry, Cloud Trace | `McpToolset` |
| CrewAI 1.15.34 | `Agent.max_iter` (25 in this release; set it explicitly), `max_execution_time` | `Task(human_input=True)`; `@human_feedback` in Flows | Flow `@persist`; checkpoints on `kickoff` | Unified `Memory` (`remember`, `recall`) | Crews with delegation, orchestrated by Flows | OpenTelemetry internally; export not confirmed | `Agent(mcps=[...])` |
| Pydantic AI 2.64.0 | `UsageLimits(request_limit=50, ...)` | `requires_approval` → `DeferredToolRequests` → `deferred_tool_results` | Message history; durable execution via Temporal, DBOS, Prefect | None built in | Call an agent inside a tool; pydantic-graph | OpenTelemetry (`Instrumentation`), Logfire optional | `MCPToolset` |
| smolagents 1.26.0 | `max_steps` (20) | No built-in pause; interrupt from a step callback, `run(reset=False)` | In-process only | `agent.memory` (this run) | `managed_agents` | OpenTelemetry via OpenInference | `MCPClient` |
| Microsoft Agent Framework `agent-framework` 1.24.0 | `function_invocation_configuration["max_iterations"]` (40) | `@tool(approval_mode="always_require")`; reply with an approval response on the same `AgentSession` | `AgentSession`; workflow checkpoint storage | `context_providers` | `as_tool()`, `WorkflowBuilder`, handoff and sequential orchestrations | OpenTelemetry | `MCPStdioTool`, `MCPStreamableHTTPTool` |

Notes on that table:

- smolagents had no release between 2026-05-29 and 2026-11-02. Its `CodeAgent` runs model-written Python; the local executor is not a security sandbox (`security.md`).
- The MCP Python SDK (`mcp` 2.3.0) renamed `FastMCP` to `MCPServer` in v2 and supports the 2026-07-28 `input_required` flow on the client side. It is a protocol SDK, not an agent loop.
- Microsoft Agent Framework replaced AutoGen and Semantic Kernel's agent APIs; migration guides are on Microsoft Learn.

## Hook and middleware surfaces

What each framework lets you intercept, since this is where guardrails and verification attach:

| Framework | Before tool call | After tool call | Around model call | Run level |
|---|---|---|---|---|
| Claude Agent SDK | `PreToolUse` (allow / deny / ask / defer, edit input) | `PostToolUse`, `PostToolUseFailure` (add context, replace output) | `PreCompact` | `Stop`, `SubagentStart/Stop`, `UserPromptSubmit` |
| OpenAI Agents SDK | `tool_input_guardrails`, `on_tool_start` | `tool_output_guardrails`, `on_tool_end` | `on_llm_start/end` | `@input_guardrail`, `@output_guardrail` (tripwires) |
| LangChain 1.x | `wrap_tool_call` | `wrap_tool_call` | `before_model`, `after_model`, `wrap_model_call` | `before_agent`, `after_agent` |
| Google ADK | `before_tool_callback` | `after_tool_callback`, `on_tool_error_callback` | `before/after_model_callback` | Plugins: `before/after_run_callback`, `on_event_callback` |
| CrewAI | `@before_tool_call` (raise `HookAborted` to block) | `@after_tool_call` | `@before/after_llm_call` | `@before/after_kickoff` |
| Pydantic AI | `before_tool_execute`, `wrap_tool_execute` | `after_tool_execute` | `before/after/wrap_model_request` | `before/after/wrap_run` |
| Microsoft Agent Framework | `FunctionMiddleware` | `FunctionMiddleware` | `ChatMiddleware` | `AgentMiddleware` |

## Provider-side context features

These move context management from your harness into the API. Use them when you're on that provider; keep the logic portable otherwise.

- **Claude API:** context editing (`clear_tool_uses_20250919` clears old tool results; `clear_thinking_20251015` clears old thinking), server-side compaction (`compact_20260112`, default trigger 150,000 input tokens), the memory tool (`memory_20250818`), tool search (`tool_search_tool_regex_20251119` / `_bm25_20251119`, with other tools marked `defer_loading`), and programmatic tool calling (tools opt in through `allowed_callers` with a `code_execution` tool version). Beta headers vary by feature; check the docs page for each.
- **OpenAI Responses API:** server-side compaction via `context_management` with a `compact_threshold`, a standalone `/responses/compact` endpoint, and `OpenAIResponsesCompactionSession` in the Agents SDK.

## Running a framework's agent inside an eval

Inspect AI's `agent_bridge()` routes OpenAI or Anthropic client calls from an in-process agent (OpenAI Agents SDK, Pydantic AI, LangChain) to the model under test, and `sandbox_agent_bridge()` does the same for CLI agents (Claude Code, Codex CLI, Gemini CLI) running in a sandbox. That lets you grade the agent you ship rather than a re-implementation of it (`evals.md`).

## Red flags when adopting a framework

- The default turn cap is high and nobody set it.
- Approval works only as an in-process callback, so a run can't wait overnight for a human.
- Prompts and tool results are hidden; you can't print exactly what the model saw on a given turn.
- State lives in the framework's store and can't be moved.
- Upgrading the framework is the only way to get a fix you need in the loop.
