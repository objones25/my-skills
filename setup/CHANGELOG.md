# Changelog

## 0.1.0, 2026-11-25

First version.

- Seven-step workflow: mode, scan, interview, context trim, propose and write, verify, hand off.
- `references/interview.md`: question bank of 22 categories, each routed to the file its answer lands in; interview method adapted from GitHub spec-kit `/clarify` and the Claude Code best-practices interview pattern.
- `references/artifacts.md`: routing rule and formats for CLAUDE.md, rules, settings, subagents, project skills, `.mcp.json` and memory; hooks are delegated to the hook-writer skill.
- `references/context.md`: what loads at startup and every per-project switch (`enabledPlugins`, `skillOverrides`, `disabledMcpjsonServers`, `Agent(...)` deny, `/mcp` toggle).
- `scripts/check_setup.py`: lint with codes ST001 to ST010.
- `scripts/inventory.py`: per-project inventory of plugins, skills, subagents, MCP servers, instruction files and memory index, names only.
- Independent review: 20 findings, all accepted and fixed or recorded under Not verified in `AUDIT.md`.
- Facts read from the Claude Code docs on 2026-11-25 against Claude Code 2.1.332; quotes grep-matched against downloaded pages (see `references/sources.md`).

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-25 | macOS arm64 | Python 3.13.2, Claude Code 2.1.332 | 71 passed (`uv run --no-project --with pytest python -m pytest tests`), after fixing the independent review's findings (see AUDIT.md) |

### Evals

Not run yet. `evals/evals.json` (6 cases, one control) and `evals/trigger_evals.json` (10 should-trigger, 10 near misses) are written but have not been run with or without the skill.
