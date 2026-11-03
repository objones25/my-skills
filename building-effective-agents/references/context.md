# Context engineering

Context is the scarce resource. "As the number of tokens in the context window increases, the model's ability to accurately recall information from that context decreases." That is context rot (the term comes from [Chroma's research](https://research.trychroma.com/context-rot)): a performance gradient rather than a cliff, caused partly by n² pairwise attention relationships and less training exposure to very long sequences (Anthropic, *Effective context engineering for AI agents*, Sep 2025). Bigger windows don't exempt a system; they postpone the problem.

Working goal: **find the smallest set of high-signal tokens that maximizes the likelihood of the desired outcome.**

## Four techniques, and when each is right

**Just-in-time retrieval.** Hold lightweight identifiers (file paths, stored queries, links) and load content at runtime with `grep`, `head`, or targeted queries rather than pre-loading everything. Folder hierarchies, naming conventions, and timestamps are themselves signal the agent can navigate. Runtime exploration is slower than a precomputed index but never goes stale. The usual default is hybrid: a small always-loaded file (CLAUDE.md-style) plus runtime search. Do the simplest thing that works.

**Compaction.** Summarize the run and reinitialize the window with the summary. Keep architectural decisions, unresolved bugs, and implementation details; discard redundant tool output. Tune by maximizing recall first, then improving precision. The lowest-hanging win is clearing old tool results, which are rarely needed twice. Any edit to earlier context invalidates the prompt cache from that point on (Manus, Jul 2025), so measure cache hit rate before and after (`production.md`).

**Context resets.** Clear the window entirely and start a fresh agent with a structured handoff file. Anthropic needed resets on Sonnet 4.5, which wrapped up early as it sensed its context limit approaching; on Opus 4.5 the behavior was gone and automatic compaction sufficed (*Harness design for long-running application development*, Mar 2026). Re-test before keeping them.

**Note-taking / external memory.** Have the agent write progress, decisions, and remaining work to a file it re-reads: to-do lists, `NOTES.md`, a running plan. This is what lets a long run survive a context reset with its intent intact, and what makes resumption possible after a failure. Memory that must survive across sessions is covered in `memory.md`.

**Subagent isolation.** A subagent can burn tens of thousands of tokens exploring and return a distillate of "often 1,000–2,000 tokens." Use subagents to keep search, file reading, and dead ends out of the parent context. Isolation is one of two reasons to spawn one; see `multi-agent.md` for the other.

Anthropic's guideline (it says the choice depends on task characteristics): compaction for tasks with extensive back-and-forth, note-taking for iterative work with clear milestones, multi-agent for complex research where parallel exploration pays.

## What controlled comparisons say about compaction

Research since 2025 mostly supports starting cheap and adding LLM summarization only where the cheap method fails:

- **Masking old tool output is as good as summarizing it, for less.** On SWE-bench Verified with SWE-agent across five model configurations, replacing old observations with a placeholder halved cost against the raw agent and matched or slightly beat LLM summarization's solve rate; summarization also made trajectories longer. A hybrid (mask first, summarize later) was 7% cheaper than masking alone and 11% cheaper than summarization alone (Lindenbauer et al., *The Complexity Trap*, arXiv 2508.21433). One configuration lost solve rate under both strategies, so measure on your own model.
- **Most of the benefit is avoiding overflow.** In a 2026 harness study of coding agents, the gap between managed and unmanaged context on SWE-bench Verified shrank from 35.7 points at a 32k window to 2.7 points at 128k, and rule-based elision before summarization gave similar accuracy at the lowest cost in 7 of 8 settings; a tool for recalling elided content was rarely used (Fan et al., arXiv 2609.20804, open models only).
- **Summaries lose detail.** Compact-Bench reported LLM summaries keeping at most 16% of recurring details against 79% for mechanical extraction, and summary-only continuations breaking 5 to 64% of runs that succeeded with full history (Sun et al., arXiv 2610.31710, abstract only, small success pool). ACE calls the same effect context collapse: repeated full rewrites shrink a context and drop specifics, which incremental, itemized updates avoid (Zhang et al., arXiv 2510.04618).
- **Learned compression can close most of the gap but costs latency.** ACON cut peak tokens 26 to 54% with small accuracy loss against full context, while adding roughly 15 to 30 seconds per AppWorld task (Kang et al., arXiv 2510.00615, vendor-authored, mostly GPT models).

Practical order: clear or mask old tool results; keep decisions, open problems, and identifiers verbatim in a notes file; summarize only the remainder, with itemized updates rather than rewrites; and check the prompt-cache hit rate before and after (`production.md`). Hosted versions of these steps exist (context editing and compaction on the Claude API, compaction on the OpenAI Responses API; `frameworks.md`).

## Structuring long agentic work

For hard tasks in large codebases, the research → plan → implement shape is well field-tested (Dex Horthy, HumanLayer, *Getting AI to Work in Complex Codebases*, in the `advanced-context-engineering-for-coding-agents` repo):

1. **Research**: delegate to subagents so raw file reads never enter the parent context. Output: which files matter, how information flows, likely causes.
2. **Plan**: exact steps, exact files and edits, and the verification step for each phase, short enough that a human will actually read it.
3. **Implement**: phase by phase, compacting status back into the plan at each verification milestone.

HumanLayer keeps context utilization in roughly the 40–60% range, depending on problem complexity, and designs the workflow around staying there. The leverage argument behind the ordering: a bad line of research can produce thousands of bad lines of code, a bad line of plan hundreds, a bad line of code one. So human review goes to research first, plans second, code last. The payoff depends on that review actually happening.

## System prompts at the right altitude

Two failure modes bracket the useful range: brittle hardcoded logic that breaks on the first unanticipated input, and vague high-level guidance that assumes context the model doesn't have.

- Organize into distinct sections (XML tags or Markdown headers): background, instructions, tool guidance, output format.
- Aim for the minimal set of information that fully outlines expected behavior. Minimal does not mean short.
- Start minimal on the strongest model, then add instructions and examples targeted at observed failures.
- Curate a few diverse canonical examples instead of a laundry list of edge cases.
- Knowledge that is only sometimes relevant belongs in a loadable skill or reference file, not in the always-resident prompt. Skills load in three stages (metadata always, body when relevant, bundled files only when needed), which is the same principle applied to procedural knowledge.
- A bloated always-on prompt makes the model ignore the instructions that matter. If everything is marked important, nothing is.

## Diagnosing context problems

- **Degrades late in long sessions** → context rot. Add compaction, clear stale tool results, move exploration into subagents.
- **Forgets constraints stated early** → the same, plus write the constraints to a durable notes file the agent re-reads.
- **Repeats work it already did** → no external progress artifact.
- **Dies mid-task** → oversized tool payloads more often than conversation length; check `tools.md` first.
- **Got worse after adding summarization** → detail lost in the summary. Keep identifiers and decisions verbatim outside it; try masking instead.
- **Costs jumped after a context change** → the edit broke prompt-cache reuse; run `scripts/cache_prefix_check.py` on logged requests.
