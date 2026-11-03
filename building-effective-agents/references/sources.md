# Sources

Last verified: 2026-11-02 (entries added in the 2026-11-02 audit); earlier rows last verified 2026-10-18. Re-check any source whose page shows an edit date after this. "Opened" means the page was read during verification and the listed claims were checked against it; "excerpt" means only a search excerpt of the primary page; "secondary" means only third-party coverage. Most sources are vendors describing their own products, so prefer claims that more than one vendor makes.

## Anthropic

| Source | Date | Checked | Supports |
|---|---|---|---|
| [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | Dec 19, 2024; modified Aug 10, 2026 (page metadata) | Opened | Workflow vs agent, pattern table, "simplest solution" quote, framework quote, absolute file paths, tool-format advice. The 2026 edit notes that much of the tooling landscape it describes has changed |
| [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) | Jun 13, 2025 | Opened | 4x and 15x tokens, 80% of variance, model upgrade beating doubled budget, 90.2% (Opus 4 lead, Sonnet 4 subagents), 90% faster research, 40% faster after tool rewrites, effort scaling, four-field briefs, semiconductor example, the hypothetical 30% to 80% prompt tweak (phrased as "might", not a measured result), single-judge result, rainbow deploys, resumption, privacy-limited monitoring |
| [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | Sep 11, 2025 | Opened | Tool contract quote, consolidated tools, namespacing, 25,000-token cap, 206 vs 72 tokens, UUID quote, SWE-bench note, the "2025" query example, held-out test sets, tool annotations |
| [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Sep 29, 2025 | Opened | Context rot, n² attention and long-sequence training, smallest high-signal token set, ambiguity test, system-prompt altitude, just-in-time retrieval, compaction (recall then precision, tool-result clearing), note-taking, 1,000 to 2,000 token subagent summaries, technique-selection guideline |
| [Building agents with the Claude Agent SDK](https://claude.com/resources/articles/building-agents-with-the-claude-agent-sdk) | Sep 29, 2025 | Opened | Verification ladder: rules, visual feedback, LLM judge ("generally not a very robust method"). The old anthropic.com/engineering URL redirects here |
| [Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) | Oct 16, 2025 | Opened | Three-stage skill loading: metadata at startup, SKILL.md when relevant, bundled files as needed |
| [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) | Nov 4, 2025 | Opened | 150,000 to 2,000 tokens (98.7%), 10,000 rows to five, `search_tools` detail level, PII kept out of context, sandboxing tradeoff |
| [Introducing advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) | Nov 24, 2025 | Opened | Tool Search (85% fewer definition tokens; Opus 4 49% to 74%), 10+ tools or 10K tokens as guidelines, three to five tools always loaded, programmatic tool calling (37%; three or more dependent calls), input examples (1 to 5; 72% to 90%) |
| [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | Jan 9, 2026 | Opened | 20 to 50 tasks, two-expert bar, 0% pass@100, balanced sets, grader table, per-dimension judges, "Unknown" escape hatch, trial isolation and the git-history example, reading transcripts, pass@k and pass^k, saturation, hack-resistant graders, simulated users |
| [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) | Mar 24, 2026 | Opened | Self-evaluation leniency, QA agents approving bugs they found, generator/evaluator split, sprint contracts, context resets vs compaction, when an evaluator pays off, tuning the QA prompt from logs |
| [Scaling Managed Agents](https://www.anthropic.com/engineering/managed-agents) | Apr 8, 2026 | Opened | Harness assumptions expire, session log outside the context window, credentials unreachable from the sandbox |
| [Claude Code MCP docs](https://code.claude.com/docs/en/mcp) | Current docs | Opened | MCP output: 10,000-token warning, 25,000-token default cap, per-tool limits, 50,000-character file save |
| [An update on recent Claude Code quality reports](https://www.anthropic.com/engineering/april-23-postmortem) | Apr 23, 2026 | Opened | Three harness changes (default effort, thinking-clearing bug that passed tests and review, verbosity prompt line costing ~3% on an eval) read as one regression; per-model evals, soak, gradual rollout, dogfooding |
| [How we contain Claude across products](https://www.anthropic.com/engineering/how-we-contain-claude) | May 25, 2026 | Opened | ~93% of permission prompts approved; OS sandbox cut prompts 84%; ~0.1% single-attempt and ~5-6% after 100 adaptive attempts injection success; 24/25 user-originated exfiltration in red team, stopped only by deterministic egress/filesystem controls |
| [Claude API: memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) | Current docs | Opened | `memory_20250818`, client-side file operations |
| [Claude API: context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) and compaction | Current docs | Via Context7 | `clear_tool_uses_20250919`, `clear_thinking_20251015`, `compact_20260112` (default trigger 150,000 input tokens), memory-tool warning before clearing |
| [Claude API: tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool), [programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) | Current docs | Opened | Regex/BM25 tool search with `defer_loading`; `allowed_callers` with a `code_execution` tool version |

## Other vendors and projects

| Source | Date | Checked | Supports |
|---|---|---|---|
| [OpenAI, A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) | 2025 (no date on the PDF) | Opened | When to build an agent, 15+ vs under 10 tools, guardrail layers, tool risk tiers, manager vs decentralized patterns, escalation triggers |
| [OpenAI, The next evolution of the Agents SDK](https://openai.com/index/the-next-evolution-of-the-agents-sdk/) | Apr 15, 2026 | Opened | Sandbox-aware harness separated from compute; credentials kept out of sandboxes |
| [OpenAI, Designing AI agents to resist prompt injection](https://openai.com/index/designing-agents-to-resist-prompt-injection/) | Mar 11, 2026 | Opened | Injection resembles social engineering; limit impact rather than rely on detection |
| [OpenAI, Responses API compaction](https://developers.openai.com/api/docs/guides/compaction) | Current docs | Opened | `context_management` compaction with `compact_threshold`; `/responses/compact` |
| [OpenAI, Harness engineering](https://openai.com/index/harness-engineering/) | Feb 11, 2026 | Opened | Agent legibility (logs, metrics, UI queryable by the agent), invariants enforced by custom linters and structural tests |
| [OpenAI, Agent RFT talk](https://ai.engineer/talks/agent-reinforcement-fine-tuning) and [InfoQ coverage](https://www.infoq.com/news/2025/12/qcon-openai-rft) | Dec 17, 2025 (InfoQ) | Opened (InfoQ), secondary | Prompt and task optimization come before changing weights; graders are a core artifact |
| [MCP specification 2026-07-28, changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog) | Jul 28, 2026 | Opened | Stateless protocol, server-minted handles, Tasks extension with `tasks/get` polling, multi round-trip requests, Sampling/Roots/Logging deprecated, OpenTelemetry trace context, deterministic tool order (SHOULD) |
| [MCP 2026-07-28 tools page](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) and [schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2026-07-28/schema.ts) | Jul 28, 2026 | Opened | Deterministic order SHOULD; annotations and defaults; annotations untrusted unless from trusted servers; `input_required` carrying elicitation; 2026-07-28 still latest on 2026-11-02 (draft changelog empty) |
| [MCP security best practices (2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/basic/security_best_practices) | Jul 28, 2026 | Opened | Confused deputy, token passthrough, SSRF, state handle hijacking, local server compromise, authorization URL validation, scope minimization |
| [MCP Tasks extension](https://modelcontextprotocol.io/extensions/tasks/overview) | Current | Opened | `tasks/get`, `tasks/update`, `tasks/cancel`; `taskId`, `ttlMs`, `pollIntervalMs` |
| [MCP Registry](https://modelcontextprotocol.io/registry/about) | Current | Opened | Preview status, metadata only, no security scanning |
| [A2A joins the Agentic AI Foundation](https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/) | Aug 27, 2026 | Opened | Growth-stage project at the Linux Foundation's AAIF; stable v1.0 spec |
| [Manus, Context engineering lessons](https://manus.im/en/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) | Jul 18, 2025 | Opened | KV-cache hit rate, stable prefixes, append-only context, mask rather than remove tools, keep failed attempts in context |
| [Cognition, Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) | Jun 12, 2025 | Excerpt | Conflicting implicit decisions in parallel agents |
| [Cognition, Multi-Agents: What's Actually Working](https://cognition.com/blog/multi-agents-working) | Apr 22, 2026 | Opened | Clean-context reviewer agents, manager/child delegation live in Devin, parallel-writer swarms still fragment decisions |
| [Google, Architecting context-aware multi-agent frameworks (ADK)](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/) | 2025 | Excerpt | Durable sessions vs per-call working context |
| [LangChain, Agent Middleware](https://www.langchain.com/blog/agent-middleware) | Sep 8, 2025 | Opened | Developers graduating off agent abstractions for non-trivial use; middleware as the fix |
| [CrewAI, What is CrewAI?](https://docs.crewai.com/agentic) | Current docs | Excerpt | Deterministic Flows with autonomous Crews inside |
| [Hugging Face, smolagents](https://github.com/huggingface/smolagents) | Current README | Opened | Cites CodeAct for about 30% fewer steps (not its own benchmark); local Python executor is not a security sandbox |
| [Meta, Agents Rule of Two](https://ai.meta.com/blog/practical-ai-agent-security/) | Oct 31, 2025 | Opened | Properties A, B, C; at most two per session; supervision or a fresh session when all three are needed; credits the lethal trifecta |
| [Simon Willison, The lethal trifecta for AI agents](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) | Jun 16, 2025 | Opened | Private data, untrusted content, external communication; exfiltration through HTTP requests, images, and links |
| [Dex Horthy (HumanLayer), Getting AI to Work in Complex Codebases](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md) | No date on page | Opened | Research, plan, implement; 40 to 60% context utilization; leverage of reviewing research and plans |

## Frameworks and eval tooling

Versions from the PyPI JSON API (`https://pypi.org/pypi/<name>/json`) on 2026-11-02; APIs from each project's docs via Context7; defaults marked "wheel" were confirmed by reading the released package source.

| Package | Version (release date) | Checked | Supports |
|---|---|---|---|
| claude-agent-sdk ([docs](https://code.claude.com/docs/en/agent-sdk/python)) | 0.2.176 (2026-10-31) | PyPI, docs | `max_turns`, `max_budget_usd`, hooks, `can_use_tool`, `defer`, sessions, `AgentDefinition`, MCP |
| openai-agents ([docs](https://openai.github.io/openai-agents-python/)) | 0.24.0 (2026-10-15) | PyPI, docs, wheel | `max_turns` default 10, guardrails, `needs_approval` / `RunState`, sessions, handoffs, MCP |
| langgraph / langchain ([docs](https://docs.langchain.com/oss/python/langgraph/interrupts)) | 1.2.19 (2026-10-30) / 1.4.8 (2026-11-02) | PyPI, docs | `recursion_limit` default 10,007, `interrupt` / `Command`, checkpointers, Store, middleware |
| google-adk ([repo docs](https://github.com/google/adk-python)) | 2.13.0 (2026-10-23) | PyPI, docs, wheel | `max_llm_calls` default 500, callbacks, confirmations, SessionService / MemoryService, `McpToolset` |
| crewai ([docs](https://docs.crewai.com/)) | 1.15.34 (2026-11-01) | PyPI, docs, wheel | `max_iter` default 25 in this release, hooks, Flows `@persist`, unified Memory |
| pydantic-ai ([docs](https://github.com/pydantic/pydantic-ai/tree/main/docs)) | 2.64.0 (2026-10-31) | PyPI, docs, wheel | `UsageLimits` request_limit default 50, hooks, deferred tools, Temporal/DBOS/Prefect |
| smolagents ([repo](https://github.com/huggingface/smolagents)) | 1.26.0 (2026-05-29) | PyPI, docs, wheel | `max_steps` default 20, step callbacks, `managed_agents`, `MCPClient` |
| agent-framework ([repo](https://github.com/microsoft/agent-framework)) | 1.24.0 (2026-10-30) | PyPI, docs, wheel | `max_iterations` default 40, middleware, approval mode, `AgentSession`, OTel |
| mcp ([docs](https://py.sdk.modelcontextprotocol.io/)) | 2.3.0 (2026-10-02) | PyPI, docs | `FastMCP` renamed `MCPServer`; client support for `input_required` |
| inspect-ai ([docs](https://inspect.aisi.org.uk/)) | 0.3.287 (2026-10-31) | PyPI, docs, installed and run | `Task`, `react()`, `Epochs` with `pass_at(k)` / `pass_k(k)` (estimators read from source), sandboxes, approval, agent bridges, `mockllm`, limits, `read_eval_log`. `scripts/inspect_eval_template.py --smoke` ran successfully on this version |

## Research

| Source | Date | Checked | Supports |
|---|---|---|---|
| [Kim et al., Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296), v3 | Apr 8, 2026 | Opened (abstract) | 260 configurations; +80.8% decomposable, -70.0% sequential; centralized verification limits error propagation. v1 figures (180 configurations, 45%, 17.2x) are superseded |
| [Rabanser, Kapoor, Narayanan et al., Towards a Science of AI Agent Reliability](https://arxiv.org/abs/2602.16666), v3 | Jun 2, 2026 (v1 Feb 18, 2026) | Opened (abstract) | Consistency, robustness, predictability, safety; 12 metrics, 15 models; capability gains bought small reliability gains |
| [Wang et al., Executable Code Actions Elicit Better LLM Agents (CodeAct)](https://arxiv.org/abs/2402.01030) | 2024 | Opened (abstract) | Code actions vs JSON/text actions, up to 20% higher success. The 30% fewer-steps figure smolagents cites is not in the abstract |
| [Yao et al., τ-bench](https://arxiv.org/abs/2406.12045) | 2024 | Linked from Anthropic evals post | Origin of pass^k; simulated users |
| [Chroma, Context rot](https://research.trychroma.com/context-rot) | 2025 | Linked from Anthropic, not opened | Origin of the term |
| [Chen et al., Evaluating LLMs Trained on Code (Codex)](https://arxiv.org/abs/2107.03374) | 2021 | Cited in Inspect source | Unbiased pass@k estimator |

### Added in the 2026-11-02 audit

Retrieved with Firecrawl's paper tools. "Full text" means in-body passages were read; "abstract" means only the abstract. Dates are the arXiv version cited.

| Source | Date | Checked | Supports |
|---|---|---|---|
| [Hu et al., MemoryAgentBench](https://arxiv.org/abs/2507.05257) | 2025 | Full text | Selective forgetting ≤28% multi-hop for all methods; "prefer later fact" prompt barely helps; Mem0 35 vs GraphRAG 64 vs BM25 89 on one classification task |
| [Huang et al., Harness the Memory](https://arxiv.org/abs/2608.15008) | Aug 2026 | Full text | 11 substrates, 3 open models, 4 suites; no winner; broader retrieval helps QA, hurts sequential decisions; 3.7-8x latency below no-memory on a coding suite; "trade read breadth for write depth" |
| [Shutova et al., StructMemEval](https://arxiv.org/abs/2602.11243) | v2, Oct 2026 | Full text | Retrieval-only fails structured-memory tasks; structure hints matter more than framework; hallucinated memories after many updates |
| [Xu et al., MemGym](https://arxiv.org/abs/2605.20833) | May 2026 | Full text | Memory roughly neutral on coding; single-digit gains on dialogue and web |
| [Poikarayil et al., Obsolete Facts in Agent Memory](https://arxiv.org/abs/2610.36308) | Oct 2026 | Abstract | One obsolete fact: 100% to 44.6% (7B) |
| [Zhou et al., TEPA](https://arxiv.org/abs/2608.07429) | Aug 2026 | Abstract | Append-only and last-write-wins below no memory under reversal drift; explicit revocation fixes it |
| [Lindenbauer et al., The Complexity Trap](https://arxiv.org/abs/2508.21433) | v3, Oct 2025 | Full text (abstract re-checked) | Masking halves cost, matches summarization; hybrid 7%/11% cheaper; summarization lengthens trajectories |
| [Fan et al., Harness Design for Coding Agents](https://arxiv.org/abs/2609.20804) | Sep 2026 | Full text | Managed vs unmanaged gap 35.7 pts at 32k vs 2.7 at 128k; elision-first lowest cost in 7 of 8; recall tool rarely used |
| [Kang et al., ACON](https://arxiv.org/abs/2510.00615) | v3 | Full text | 26-54% peak-token cut, small loss; added latency |
| [Zhang et al., Agentic Context Engineering (ACE)](https://arxiv.org/abs/2510.04618) | v3 | Full text | Context collapse from full rewrites; incremental delta updates |
| [Sun et al., Compact-Bench](https://arxiv.org/abs/2610.31710) | Oct 2026 | Abstract | Summaries keep ≤16% of recurring details vs 79.1% extraction; 5-64% of successes broken |
| [Cemri et al., MAST](https://arxiv.org/abs/2503.13657) | v3, Oct 2025 | Full text | 14 failure modes, 1,642 traces, 7 frameworks; top modes and rates; modest gains from prompt/topology fixes |
| [Tran and Kiela, Single-agent vs multi-agent at equal budget](https://arxiv.org/abs/2604.02460) | v2, Apr 2026 | Full text (abstract re-checked) | Single agent matches or beats MAS at equal thinking tokens on multi-hop QA; budget-control artifacts |
| [At Equal Inference Cost, Multi-Agent Structure Does Not Beat a Single Frozen Agent](https://arxiv.org/abs/2609.04217) | Jun 2026 | Abstract | ALFWorld team 0.769 vs single 0.754, p=0.80, 1.8x calls |
| [Wu et al., MCPMark](https://arxiv.org/abs/2509.24002) | 2025 | Abstract | Best model 52.56% pass@1, 33.86% pass^4 on 127 tasks |
| [Mo et al., LiveMCPBench](https://arxiv.org/abs/2508.01780) | rev. Feb 2026 | Abstract | 70 servers, 527 tools; retrieval errors ~half of failures |
| [Hasan et al., MCP Tool Descriptions Are Smelly](https://arxiv.org/abs/2602.14878) | rev. Jun 2026 | Abstract | 97.1% of 856 descriptions defective; +5.85 median, more steps, 16.67% worse |
| [Li et al., Which Metadata Matters for LLM Tool Selection in MCP?](https://arxiv.org/abs/2610.14334) | Oct 2026 | Abstract | Schema edits shift selection most |
| [Wang et al., MCPTox](https://arxiv.org/abs/2508.14925) | 2025 (AAAI) | Abstract | Tool-poisoning success up to 72.8%; refusals <3% |
| [Debenedetti et al., CaMeL](https://arxiv.org/abs/2503.18813) | rev. Jun 2025 | Full text | 77% with provable security vs 84% undefended; ~2.8x tokens; utility drops |
| [Beurer-Kellner et al., Design Patterns for Securing LLM Agents](https://arxiv.org/abs/2506.08837) | v3 | Full text | Six patterns; plan-then-execute still lets data alter arguments |
| [Nasr, Carlini et al., The Attacker Moves Second](https://arxiv.org/abs/2510.09023) | Oct 2025 | Full text (abstract re-checked) | 12 defenses bypassed, most >90%; spotlighting/sandwiching ~1% to >95% |
| [Zhu et al., ABC checklist](https://arxiv.org/abs/2507.02825) | v5 | Full text | 7/10 task-validity and outcome-validity flaws; do-nothing agent 38% on τ-bench airline |
| [Kapoor et al., HAL](https://arxiv.org/abs/2510.11977) | 2025 | Full text | 21,730 rollouts; effort not monotonic; 9x cost for 2 pts; verifying agents 13-87% more likely to succeed |
| [Wang et al., BenchJack](https://arxiv.org/abs/2605.12673) | May 2026 | Abstract | Near-perfect scores without solving tasks; 219 flaws |
| [Hardy et al., Agent Evaluation Reliability](https://arxiv.org/abs/2610.00651) | Oct 2026 | Abstract | System rankings reliable (0.935-0.994), model rankings not (0.148-0.841) |
