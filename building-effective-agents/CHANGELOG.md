# Changelog

## 1.1, 2026-11-16: fixes from the 2026-11-15 audit

Every fix answers a finding of the 2026-11-15 audit-skill run (risk scope, 454 of 1,208 items checked, blind review stopped at the 3-round cap with 16 items disputed; details in `AUDIT.md`). Item ids refer to that report.

### Scripts and tests (each test watched failing before the fix)

| Item | Problem | Fix | Test |
|---|---|---|---|
| I-F1-0034, I-G1-0020, I-G1-0021, I-G1-0022 | No test constrained `reliability.py`'s `suite_metric`, the `compare()` guard, the delta and interval arithmetic, or the CLI: 36 of 84 mutants killed (audit measurement) | New tests; no code change | 12 new tests in `TestReliability`; mutation score now 72/84 (86%). The 12 survivors are equivalent or nearly so (argparse `required`, JSON `indent`, bootstrap defaults, an `n - c < k` branch that `math.comb` makes redundant, the upper-interval index `- 1 -> - 2`) |
| I-W4-0099 | `tool_budget.py` docstring showed a hint format the code does not print | Docstring now shows the real format | `test_docstring_hint_matches_real_output` |
| I-W4-0011 | `agent_loop.py` docstring called `finish` "the only way to end" | Says it is the only way the model chooses to end, and names the three harness stops | `test_docstring_does_not_call_finish_the_only_way_to_end` |
| I-F1-0002 | SKILL.md said every script but the Inspect template is stdlib-only; `agent_loop.py`'s adapter imports `anthropic` | SKILL.md names both optional packages | `test_scripts_use_only_the_stdlib_apart_from_the_named_optional_packages` |
| I-F1-0026, I-W1-0092, I-W1-0083 | Example 04 dropped an output line without marking it; claims that all seven examples show script output | Line restored; README and SKILL.md say 01-04 show output and 05-07 are reasoning only | `test_example_script_output_matches_a_real_run` (reruns every `$ python` block). It also caught an unmarked omission in example 02 that the audit missed; fixed |
| I-W1-0093 | AUDIT.md's "7 integrity tests" went stale | Count no longer stated as current (see `AUDIT.md`) | none (record only) |

### Wrong claims corrected

- Memory primitives (I-W1-0099, I-W2-0273, I-F2-0010, I-G1-0003): `memory.md` and `frameworks.md` now say the OpenAI Agents SDK has `agents.sandbox.capabilities.Memory`, and the separate `pydantic-ai-harness` package has a `memory` capability. Both imported on 2026-11-16 (openai-agents 0.24.0, pydantic-ai-harness 0.68.0).
- MemoryAgentBench (I-W2-0274): multi-class classification scores are 32.4 / 39.8 / 75.4; 35 / 64 / 89 are labelled as the BANKING77 column.
- HAL (I-W3-0327, I-W3-0345): the 13-87% verification effect is attributed to log analysis of 1,634 transcripts on three benchmarks, not to all 21,730 rollouts.
- MAST (I-F2-0012, I-W1-0149, I-W3-0346): v3 group names ("system design issues", "task verification"); example 06 no longer ranks verification among the most common modes; `verification.md` states the 21.3% share and the two models compared, not "every framework and model".
- Tool Search 85% (I-F2-0023, disputed): now described as the post's illustrative 50+ tool setup (about 77K to 8.7K tokens), with the 49% to 74% accuracy figure attributed to internal testing.
- Figures sourcing (I-W1-0009, I-F1-0001, I-F1-0025): SKILL.md no longer says every figure comes from a dated publication; invented numbers are marked as illustrations, and undated sources are named as such.
- `sources.md`: context-editing row (I-W3-0153), Building effective agents modified date (I-W3-0124), CrewAI URL and wording (I-W3-0177), Kim et al. supersession (I-W3-0214), Chroma and the origin of "context rot" (I-W3-0220, also `context.md`), Harness the Memory latency direction (I-W3-0229), StructMemEval version (I-G1-0009), Hasan revision date (I-W3-0252), Hardy submission date (I-G1-0015).
- `examples/README.md` named `production.md` as a reference that example 04 never cites (I-W1-0106); the row now says what each file covers.

### Stale versions updated (PyPI, read 2026-11-16)

claude-agent-sdk 0.2.182, langgraph 1.2.22, langchain 1.5.0, google-adk 2.14.1, crewai 1.15.41, pydantic-ai 2.68.0, agent-framework 1.26.0, inspect-ai 0.3.290 (`frameworks.md`, `sources.md`, `evals.md`, example 07, the Inspect template docstring). The loop-cap defaults in `frameworks.md` were confirmed on these releases by the audit; the other API names in the table were not all rechecked.

### Components added

- `CHANGELOG.md`, `metadata.version` and `metadata.test-command`, an "Extend this skill" section in SKILL.md.
- `evals/trigger_evals.json` (10 should-trigger, 9 near misses).
- `evals/evals.json`: `assertions` renamed `expectations` (skill-creator's schema); cases 21-22 from audit findings and case 23, a control where the right answer is to change nothing.

### Results

- Tests: 58 passed, 0 skipped, on 2026-11-16 on macOS arm64 with Python 3.13.2, pytest 9.2.0 and inspect-ai 0.3.290 installed (`anthropic` not installed; no test needs it).
- `check_skill.py --strict`: 0 errors, 0 warnings; `package_skill.py`: tests passed in the unpacked copy (2026-11-16).
- Evals (`evals.json`, `trigger_evals.json`): not run for this version.

## 1.0, 2026-11-02

Unversioned at the time. Rewrite after the 2026-11-02 audit: memory, frameworks, MCP and testing references, eight scripts, seven examples, tests and evals 10-20. See `AUDIT.md`.
