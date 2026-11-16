# Audit, 2026-11-16 (version 1.1)

## Fact-check

audit-skill run of 2026-11-15, risk scope. Report: `audits/building-effective-agents/2026-11-15.md` in the My_Skills repo (gitignored, local only); run folder beside it. Coverage line, verbatim: "Checked 454 of 1208 items (38%): 374 confirmed, 30 wrong, 19 stale, 16 disputed, 15 could not be verified. Independent review: stopped at the 3-round cap; 16 item(s) disputed."

Every wrong and stale item was fixed on 2026-11-16 (`CHANGELOG.md`, by item id). Of the 16 disputed items, 12 disagreed only on how an item was split or quoted and needed no change; the four verdict disputes were handled as follows:

- I-W3-0124 (Building effective agents modified date): row updated to the date the page shows now.
- I-W3-0346 (MAST "every framework"): replaced with what the text states (21.3% share; two models compared).
- I-F2-0004 (LangChain 1.4.8 pin): updated to 1.5.0.
- I-F2-0023 (Tool Search 85%): reworded to what the post measures.

## Not verified

- The 754 items the audit did not check: most recommendations and reference prose, `evals/evals.json`, `scripts/cache_prefix_check.py`, `scripts/triage_transcripts.py`, and the tests beyond running them and mutation-testing `reliability.py`. Nothing here vouches for them.
- The 16 claims the audit's reviewers found missing from its inventory (added, not checked).
- Framework API names in `frameworks.md` other than the loop-cap defaults, on the 2026-11-16 versions.
- Evals were run once per case on 2026-11-16 (results in `CHANGELOG.md`); a single run per case does not establish an improvement.
- No independent review of the 1.1 changes themselves.

## Mechanical checks, 2026-11-16

- `check_skill.py` before the changes: 2 errors (evals schema, `.DS_Store`), 4 warnings (no CHANGELOG, no trigger evals, no test-command, no Extend section). After: see `CHANGELOG.md`.
- Tests before: 42 passed, 0 skipped (pytest 9.2.0, inspect-ai 0.3.290, macOS arm64, Python 3.13.2).

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
- examples: seven worked diagnoses using real script output. [Corrected 2026-11-16: only examples 01-04 show script output; 05-07 are reasoning only (audit I-W1-0092).]
- tests: 33 script tests and 7 skill-integrity tests (links resolve, every cited arXiv id is in sources.md, evals well formed). [Count as of 2026-11-02; later commits added more, see `CHANGELOG.md` for the current result.]
- evals: 11 new A/B prompts (ids 10-20).

## How sources were checked

Papers via Firecrawl's paper tools (abstract or full-text passages, marked per row). Framework versions via the PyPI JSON API; APIs via Context7 docs; some defaults read from released wheels. Inspect AI 0.3.287 (latest on 2026-11-02; 0.3.290 by 2026-11-14) was installed, its signatures read with Python's `inspect` module, and the template run end to end against `mockllm`. MCP spec, security page, and registry fetched directly.

## Not verified

- Native OpenTelemetry export in OpenAI Agents SDK, LangGraph, and CrewAI.
- Long-term memory primitives in OpenAI Agents SDK, Pydantic AI, smolagents (none found). [Wrong, corrected 2026-11-16: openai-agents 0.24.0 has `agents.sandbox.capabilities.Memory`, and the separate `pydantic-ai-harness` has a `memory` capability (audit I-W1-0099, I-G1-0003).]
- Beta-header requirements for Claude tool search and programmatic tool calling (docs pages don't mention one). [Too broad: the Agent SDK tool-search page mentions a required beta header for earlier models on Google Cloud's Agent Platform (audit I-F1-0024).]
