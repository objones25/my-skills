# Audit, 2026-11-02

Previous version: 2026-10-18 (SKILL.md, 8 reference files, 9 evals; no scripts, examples, or tests).

## Gaps found

- No coverage of long-term memory (cross-session state, stale facts, memory stores).
- No framework comparison; frameworks were discussed only as a risk.
- MCP guidance was six bullets inside tools.md; no MCP security (tool poisoning, spec-named attacks) or registry status.
- Context compaction advice predated the 2025-2026 controlled comparisons (masking vs summarization, summary detail loss).
- Multi-agent section lacked equal-budget comparisons and the MAST failure taxonomy.
- Security section lacked adaptive-attack results and structural defenses (CaMeL, design patterns).
- Evals had no benchmark-validity material and no concrete tooling; no deterministic testing guidance for harness code.
- No runnable scripts, worked examples, or tests.

## Corrections to existing text

- MCP multi round-trip requests: elicitation still exists in 2026-07-28 but travels inside `inputRequests`; what was replaced is the server-to-client request. Rewritten in references/mcp.md.
- Em dashes removed throughout for house style.

## Added

- references: memory.md, frameworks.md, mcp.md, testing.md; new sections in context, evals, multi-agent, security, production, verification, tools; 40 new sources with how each was checked.
- scripts: agent_loop, tool_budget, tool_lint, rule_of_two_audit, triage_transcripts, reliability, cache_prefix_check, inspect_eval_template.
- examples: seven worked diagnoses using real script output.
- tests: 33 script tests and 7 skill-integrity tests (links resolve, every cited arXiv id is in sources.md, evals well formed).
- evals: 11 new A/B prompts (ids 10-20).

## How sources were checked

Papers via Firecrawl's paper tools (abstract or full-text passages, marked per row). Framework versions via the PyPI JSON API; APIs via Context7 docs; some defaults read from released wheels. Inspect AI 0.3.287 was installed, its signatures read with Python's `inspect` module, and the template run end to end against `mockllm`. MCP spec, security page, and registry fetched directly.

## Not verified

- Native OpenTelemetry export in OpenAI Agents SDK, LangGraph, and CrewAI.
- Long-term memory primitives in OpenAI Agents SDK, Pydantic AI, smolagents (none found).
- Beta-header requirements for Claude tool search and programmatic tool calling (docs pages don't mention one).
