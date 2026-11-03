# Worked examples

Each example is an invented system, used to show a diagnosis end to end: the reported symptom, the candidate causes, the cheapest test that separates them, the script output from that test, and the fix. Script outputs are real runs on the fixtures in `tests/fixtures/`. The systems, names, and numbers are illustrations, not findings about any real deployment; don't present them to a user as facts about their system.

| File | Symptom | Main references |
|---|---|---|
| `01-false-success-support-agent.md` | Agent tells customers things happened that didn't | `verification.md` |
| `02-email-triage-injection.md` | "Is our email agent safe against prompt injection?" | `security.md` |
| `03-eval-plan-sizing.md` | "Is 85% good enough to ship? How many runs do we need?" | `evals.md` |
| `04-wrong-tool-and-cache-cost.md` | Picks the wrong tool; bill went up after a prompt tweak | `tools.md`, `production.md` |
| `05-memory-for-an-assistant.md` | "Should we add a memory layer?" | `memory.md` |
| `06-multi-agent-proposal.md` | "Let's split this into five agents" | `multi-agent.md` |
| `07-choosing-a-framework.md` | "LangGraph or the OpenAI Agents SDK?" | `frameworks.md` |

The shape every answer should keep: verdict first, the surface (model, tools, harness), the candidate causes, the bet, and the test.
