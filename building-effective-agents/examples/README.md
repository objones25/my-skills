# Worked examples

Each example is an invented system, used to show a diagnosis end to end: the reported symptom, the candidate causes, the cheapest test that separates them, and the fix. Examples 01-04 also show the script output from that test: real runs on the fixtures in `tests/fixtures/`, rerun by `tests/test_skill_integrity.py` (a `...` marks omitted output). Examples 05-07 are reasoning only, with no script output. The systems and names are invented. Numbers are either invented illustrations or cited findings, and a cited finding names its source (Anthropic's 15x tokens in 06, the ABC audit's 38% in 03, framework versions in 07); neither kind is a finding about the user's deployment; don't present them to a user as facts about their system.

| File | Symptom | Main references |
|---|---|---|
| `01-false-success-support-agent.md` | Agent tells customers things happened that didn't | `verification.md` |
| `02-email-triage-injection.md` | "Is our email agent safe against prompt injection?" | `security.md` |
| `03-eval-plan-sizing.md` | "Is 85% good enough to ship? How many runs do we need?" | `evals.md` |
| `04-wrong-tool-and-cache-cost.md` | Picks the wrong tool; bill went up after a prompt tweak | `tools.md` (tool overlap); `production.md` covers the cache-prefix half |
| `05-memory-for-an-assistant.md` | "Should we add a memory layer?" | `memory.md` |
| `06-multi-agent-proposal.md` | "Let's split this into five agents" | `multi-agent.md` |
| `07-choosing-a-framework.md` | "LangGraph or the OpenAI Agents SDK?" | `frameworks.md` |

The shape every answer should keep: verdict first, the surface (model, tools, harness), the candidate causes, the bet, and the test.
