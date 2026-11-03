# 06: "Let's split our research agent into five agents"

*Illustrative system.* A single-agent research assistant answers analyst questions. An engineer proposes a planner, three specialist researchers, and a writer, citing a demo where the multi-agent version gave better answers.

## Verdict

Not yet. First check whether the demo compared equal budgets, and which wall the single agent actually hit.

## Questions that decide it

1. **Equal compute?** If the five-agent version spent several times the tokens, the comparison measured spend. At equal thinking-token budgets, a single agent matched or beat five multi-agent designs on multi-hop QA (`multi-agent.md`). Re-run the single agent with the same token budget before deciding.
2. **Which named wall?** Splitting pays for breadth-first work that exceeds one context window, or for independent subtasks worth running in parallel. Anthropic's research system is the canonical win, at roughly 15x chat tokens against roughly 4x for a single agent (both measured against chat). "It feels big" is not a wall.
3. **Personas or isolation?** "Planner, researcher, writer" is an org chart. The two real reasons to spawn a subagent are keeping exploration out of the parent context and getting a verifier that didn't do the work.

## If the split is justified

- Specialists read and explore in parallel; one agent owns the final answer and any writes.
- Every subagent gets a four-field brief (objective, output format, tool guidance, boundaries) and its own turn and token caps.
- Subagents write findings to a store and return references, not full text.
- An explicit stop condition and an independent check, since step repetition, not knowing when to stop, and weak verification are the most common multi-agent failure modes in MAST's trace study.

## Test

Same eval tasks, several trials each, cost per successful task reported for both versions (`evals.md`). Use `scripts/reliability.py compare` on per-task results; ship the split only if the gain survives the interval and the cost.
