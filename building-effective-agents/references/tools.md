# Tools and the agent-computer interface

Tools are "a contract between deterministic systems and non-deterministic agents" (Anthropic, *Writing effective tools for agents*, Sep 2025). Design them for an agent with a finite context window, not for a developer with cheap memory. Small refinements to tool descriptions can yield dramatic improvements: Claude Sonnet 3.5 reached state of the art on SWE-bench Verified after precise refinements to *tool descriptions*.

## Choosing which tools to build

Don't wrap an API one tool per endpoint. Build few tools shaped like the workflows agents actually run:

- `list_users` + `list_events` + `create_event` → `schedule_event`
- `read_logs` → `search_logs`, returning matching lines with surrounding context
- `get_customer_by_id` + `list_transactions` + `list_notes` → `get_customer_context`

More tools do not mean better outcomes. Count and overlap both hurt: overlap causes wrong-tool picks, and every loaded definition costs context and selection accuracy. OpenAI's guide notes some systems handle 15+ well-defined, distinct tools while others struggle with fewer than 10 overlapping ones. Anthropic suggests loading tools on demand past roughly 10 tools or 10K tokens of definitions (*Introducing advanced tool use*, Nov 2025; guidelines, not hard cutoffs). See "When the tool definitions themselves are the problem" below.

**The ambiguity test:** "If a human engineer can't definitively say which tool should be used in a given situation, an AI agent can't be expected to do better." (Anthropic, *Effective context engineering for AI agents*, Sep 2025.) When a user reports wrong-tool selection among near-duplicate names, run that test on *their* tool names as the diagnosis: merge the ones that fail it, or give each a disjoint name and a description that states when *not* to use it. A cheap version: show two engineers the tool list and a task, and see whether they pick the same tool.

## Namespacing

Group related tools under a common prefix: `asana_search`, `jira_search`, or `asana_projects_search`, `asana_users_search`. Anthropic found "selecting between prefix- and suffix-based namespacing to have non-trivial effects on our tool-use evaluations". Pick one convention and measure it rather than assuming.

## Budgeting response tokens

Tool results are the main way context fills. Build in pagination, range selection, filtering, and truncation with sensible defaults.

- Claude Code caps MCP tool output at 25,000 tokens by default, warns above 10,000, and saves text results over 50,000 characters to a file; a tool can raise its own limit (Claude Code MCP docs). Adopt an explicit cap; make truncation visible and recoverable (`[showing 20 of 1,483 lines; call with offset=20 for more]`).
- Offer a response-format parameter (`concise` / `detailed`). In Anthropic's Slack example the same thread rendered 206 tokens detailed versus 72 concise, about a third.
- Return the fields that carry meaning and drop the plumbing: prefer `name`, `file_type`, `image_url` over `uuid`, `mime_type`, `256px_image_url`. "Merely resolving arbitrary alphanumeric UUIDs to more semantically meaningful and interpretable language ... significantly improves Claude's precision in retrieval tasks by reducing hallucinations."

## Format and argument design

- Test the serialization. JSON vs XML vs Markdown measurably changes eval performance and does not port across models. If a model repeatedly calls a tool in a shape you didn't specify, treat it as a signal and let an eval decide between adopting the shape and fixing the description. When Claude kept appending "2025" to web-search queries, Anthropic fixed the description rather than accept the shape.
- Keep formats close to text the model has seen naturally; avoid overhead like string escaping or hand-counted line offsets.
- Poka-yoke the arguments so misuse is structurally impossible. Requiring absolute file paths eliminated a whole class of relative-path errors. Name parameters unambiguously (`user_id`, not `user`).
- For parameters with conventions a schema can't express (date formats, ID shapes, which optional fields go together), add 1 to 5 realistic input examples to the tool definition. Anthropic measured accuracy on complex parameter handling rising from 72% to 90% (*Introducing advanced tool use*, Nov 2025).
- Make errors instructive: say what to do next, not an opaque code or a traceback.
- Leave the model room to think before it commits to a call it can't undo.
- Document each tool with example usage, edge cases, input requirements, and explicit boundaries.

## When the tool definitions themselves are the problem

Large tool catalogs consume context before the agent reads the request. Anthropic's Tool Search loads definitions on demand (API tools `tool_search_tool_regex_20251119` and `tool_search_tool_bm25_20251119`, with other tools marked `defer_loading`; see `frameworks.md`): in its benchmark, definition tokens fell 85% and Opus 4 tool-selection accuracy rose from 49% to 74% (*Introducing advanced tool use*, Nov 2025). The same post advises keeping the three to five most-used tools always loaded. Two related mitigations come from *Code execution with MCP* (Nov 2025), where presenting MCP servers as code APIs cut one example task from 150,000 tokens to 2,000 (98.7%):

- **Progressive tool disclosure**: expose servers and tools as files the agent lists and reads on demand, or provide a `search_tools` call with a detail-level parameter (name only / name + description / full schema).
- **Filter results in an execution environment** rather than through the model: a 10,000-row spreadsheet becomes the five rows that matter, and intermediate results and PII can stay out of the model's context entirely. Anthropic's programmatic tool calling packages this pattern and cut average token use 37% on complex research tasks.

The tradeoff is explicit in *Code execution with MCP*: generated code needs sandboxing, resource limits, and monitoring, operational overhead that direct tool calls avoid. Recommend it when tool sprawl or payload size is already the bottleneck. *Introducing advanced tool use* recommends programmatic tool calling for multi-step workflows with three or more dependent calls. On code actions versus JSON tool calls, the CodeAct paper reports up to 20% higher success (Wang et al., arXiv 2402.01030, 2024), and the smolagents README cites it for about 30% fewer steps; neither is a measurement of your workload. The same README states that smolagents' local Python executor is not a security sandbox.

## MCP servers

Protocol-specific design (stateless handles, Tasks, `input_required`, deprecations, annotations) and MCP security live in `mcp.md`. The tool-design rules in this file apply to MCP tools unchanged.

## What recent measurements add

- **Descriptions and schemas are most of the interface.** Most published MCP tool descriptions have defects, and parameter schemas move selection more than names do (details and caveats in `mcp.md`). Treat a description rewrite as a change to be evaluated, since it can also add steps or make some cases worse.
- **Retrieval is the bottleneck in large catalogs.** With hundreds of tools, picking the right one fails more often than calling it correctly; on-demand loading with a few always-loaded tools is the usual fix.
- **Verification pays.** Across 21,730 rollouts on its leaderboard, HAL found agents that verified their own work were 13 to 87% more likely to succeed (Kapoor et al., arXiv 2510.11977). A tool that lets the agent check an outcome (read back the record, run the test) is often worth more than another action tool.

## Scripts

- `scripts/tool_lint.py tools.json` screens a tool set for overlapping names and descriptions, missing "when not to use" boundaries, vague parameter names, side-effect tools without annotations, and catalog size. It is a heuristic; confirm with a selection eval.
- `scripts/tool_budget.py` caps and pages tool output with a visible, recoverable hint.

## Evaluating tools

Write tasks that require many real tool calls, grounded in actual usage:

- Good: "Customer 9182 reports being charged three times for one purchase. Find the relevant log entries and determine whether other customers were affected."
- Weak: "Search the payment logs for `purchase_complete` and `customer_id=9182`." Single-call probes test nothing about selection.

Collect accuracy, tool-call count, total tokens, tool errors, and runtime per call and per task. Then feed the transcripts back to a model and let it refactor descriptions and schemas in bulk (Anthropic's Slack and Asana tool suites both improved that way), and hold out a test set so you're not overfitting. A tool-testing agent that rewrote tool descriptions produced a 40% decrease in task completion time for downstream agents (*How we built our multi-agent research system*, Jun 2025).

Splitting into more agents does not fix tool overlap or oversized payloads; it duplicates both across more contexts. Fix the interface first, then reconsider architecture (`multi-agent.md`).
