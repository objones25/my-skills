# Long-term memory

Memory here means state that outlives one context window or one session: user preferences, facts learned last week, what a long run has already done. Within a single run, that is context engineering (`context.md`). Across runs, it is this file.

## First question: does the task need it?

Memory pays where re-deriving state is expensive and hurts where it competes with the task for attention. Controlled studies agree on that shape, not on a winning design:

- In MemGym (Xu et al., arXiv 2605.20833, May 2026), memory was roughly neutral on coding, where progress already lives in the filesystem, and gave mid- to high-single-digit success gains on tool-use dialogue (τ2-bench) and web tasks.
- *Harness the Memory* (Huang et al., arXiv 2608.15008, Aug 2026) compared 11 memory substrates (dense and sparse indices, text records, graphs, hierarchical stores, parametric updates) across 3 open models and 4 suites. No substrate won everywhere. Retrieving more entries helped long-context QA and hurt sequential decision-making (ALFWorld), where some retrievers fell below the no-memory baseline; several substrates cost 3.7x to 8x the latency on a coding suite and still scored below no memory.

So: a coding agent that can grep the repo and read its own progress file usually needs no memory store. A support or assistant agent that must remember a customer across sessions does. Ask what would go wrong without it before building one.

## Kinds of memory, and where each lives

| Kind | Holds | Simplest implementation |
|---|---|---|
| Working | The current run's plan, progress, decisions | A notes/progress file the agent re-reads (`context.md`) |
| Semantic | Stable facts and preferences ("prefers metric units") | Key-value or document store, written explicitly |
| Episodic | What happened in past sessions | Session logs plus search; summaries if logs are too long |
| Procedural | How to do recurring tasks | Loadable skills or instructions, versioned like code |

Procedural memory is the one teams most often get wrong by storing it as retrievable "memories" instead of as skills or prompts under version control. If the agent learned a procedure, write it down where a human can review and diff it.

## Design rules

**Write deliberately, read narrowly.** *Harness the Memory* sums up its findings as trading read breadth for write depth: spend effort deciding what to store and how to organize it, and retrieve few entries. More retrieved context shifted the model's attention away from action-relevant context in their probe.

**Give memory a structure the task needs.** StructMemEval (Shutova et al., arXiv 2602.11243 v2, Oct 2026) built tasks that humans solve by organizing knowledge (ledgers, to-do lists, trees). Retrieval-only agents failed them even with top-20 retrieval; memory agents did much better, and telling the model *which* structure to keep mattered more than which memory framework it ran on. Failure modes were not organizing memory at all and hallucinated memories after hundreds of updates. If the domain has a natural shape (an account ledger, a task list), make that shape the schema.

**Invalidate stale facts explicitly.** This is the failure most memory systems ship with.

- In MemoryAgentBench (Hu et al., arXiv 2507.05257), every method scored at most 28% on multi-hop selective forgetting, and telling the model to prefer the later fact barely moved scores.
- In TEPA (Zhou et al., arXiv 2608.07429, Aug 2026, synthetic drift), append-only and last-write-wins memory both scored below *no memory* when facts reversed; explicit revocation fixed it.
- Placing one obsolete fact next to the current one cut a 7B model's accuracy from 100% to 44.6% (Poikarayil et al., arXiv 2610.36308, Oct 2026, abstract only, small models).

Store an update as a supersession (old value, new value, date, source), delete or mark the old record, and never rely on a prompt rule ("use the newest") to resolve conflicts at read time.

**Keep provenance.** Each memory should say where it came from (user said, tool returned, model inferred) and when. Inferences are the first thing to drop when a memory conflicts with something the user stated. This also limits memory poisoning: a fact written from an untrusted web page should not later be read back as if the user said it (`security.md`).

**Scope memory to the user and the task.** Namespaces per user, per tenant, per project. Cross-user leakage through a shared store is a privacy bug, not a recall feature.

**Let users see and delete it.** A memory the user can't inspect becomes a source of unexplained behavior. Build view, correct, and forget as ordinary operations.

## What the platforms provide

Check the framework you already use before adding a memory product (versions and API names in `frameworks.md`):

- Claude API memory tool: a client-side file directory the model reads and writes with `view`, `create`, `str_replace`, `insert`, `delete`, `rename`. Your code executes the operations, so you control storage, scoping, and retention. Paired with context editing, the model is warned before old tool results are cleared so it can save what matters first.
- LangGraph: a Store (`put`, `search`) separate from per-thread checkpoints.
- Google ADK: `MemoryService` (`add_session_to_memory`, `search_memory`, the `load_memory` tool) separate from `SessionService`.
- CrewAI: a unified `Memory` class with `remember` / `recall`.
- OpenAI Agents SDK, Pydantic AI, smolagents: sessions or message history; no long-term memory store found in their docs as of 2026-11-02.

Off-the-shelf memory products are not automatically better than plain retrieval. In MemoryAgentBench's multi-class classification task, Mem0 scored 35, GraphRAG 64, and BM25 89. One task, one paper; test on yours.

## Testing memory

- Build eval tasks that span sessions: write in session 1, query in session 5, with distractor sessions between.
- Include updates and reversals (the user moved, changed a preference) and check that the old fact is gone as well as that the new one is retrievable.
- Include a no-memory baseline. If memory doesn't beat it on your tasks, don't ship it.
- Grade with outcome checks where possible (the agent used the right address) rather than asking a judge whether the answer "seems informed."
- Watch latency and tokens per turn; memory reads are on the critical path.

## Diagnosing memory problems

| Symptom | Likely cause | First check |
|---|---|---|
| Uses an old preference after the user changed it | Append-only store, no revocation | How updates are written; is the old record still retrievable? |
| Brings up unrelated past details | Reads too broadly; no relevance gate | Retrieval count and threshold; is memory read on every turn? |
| Got worse at its main task after adding memory | Retrieved memory crowds out task context | Ablate memory on your eval; reduce read depth |
| "Remembers" things the user never said | Model-inferred memories stored as facts; poisoning from tool content | Provenance on each record |
| One user sees another's data | Shared namespace | Store keys and access checks |
