# Sources

Checked means how much was read: Full, Partial, Abstract only, or Installed (read from the installed program or files on disk). Claude Code docs pages carry no date; "Date checked" is when they were read, against Claude Code 2.1.332 on macOS arm64.

Pages marked "Partial, grep" were downloaded as markdown (`https://code.claude.com/docs/en/<page>.md`) and every quote this skill uses was matched against the downloaded text. Three research subagents read more of each page; their summaries were not trusted without that match, and one claim they made (an 8,000-character fallback for the skill listing budget) was dropped because it was not found on the page.

| Source | Date checked | Checked | Supports |
|---|---|---|---|
| code.claude.com/docs/en/memory | 2026-11-25 | Partial, grep | CLAUDE.md locations, 200-line target, `@import` four hops, rules and `paths` as the only field, AGENTS.md interplay, auto memory location and 200 lines or 25KB, `autoMemoryEnabled`, `CLAUDE_CODE_NEW_INIT`, `/init` suggests improvements, `/doctor prompt-audit` (v2.1.283) |
| code.claude.com/docs/en/best-practices | 2026-11-25 | Partial, grep | "Only write down what Claude can't find by reading the repo"; hook vs CLAUDE.md routing; keep enforced set small ("forty of them"); interview pattern and SPEC.md in a fresh session; the five questions before unattended runs; subagents do not improve a delegated answer; servers cost context when unused |
| code.claude.com/docs/en/skills | 2026-11-25 | Partial, grep | Frontmatter fields all optional, `description` recommended; 1,536-character cap; listing budget 1% of context window, least-invoked dropped first; `skillOverrides` states; plugin skills ignore `skillOverrides`; `/skill-doctor` (v2.1.252) |
| code.claude.com/docs/en/mcp | 2026-11-25 | Partial, grep | `.mcp.json` format and `${VAR}`/`${VAR:-default}` expansion; `/mcp` toggle stored per project in `~/.claude.json` as `disabledMcpServers`; `claude mcp list` states; tool search loads names only; `ENABLE_TOOL_SEARCH=false`; `disableClaudeAiConnectors` |
| code.claude.com/docs/en/settings | 2026-11-25 | Partial, grep | Precedence order; `defaultMode` `auto`/`bypassPermissions` ignored from project and local files; keys that wait for workspace trust; `$schema` URL and lag warning |
| code.claude.com/docs/en/settings-reference | 2026-11-25 | Partial, grep | `disabledMcpjsonServers`, `enabledPlugins`, `skillOverrides` keys |
| code.claude.com/docs/en/permissions | 2026-11-25 | Partial, grep | deny, then ask, then allow; bare tool name in deny removes it from context; `Agent(Explore)` rule form |
| code.claude.com/docs/en/sub-agents | 2026-11-25 | Partial, grep | `.claude/agents/` frontmatter; unknown fields ignored without error; 15,000-token warning for custom subagent descriptions |
| code.claude.com/docs/en/claude-directory | 2026-11-25 | Partial, grep | What to commit; `settings.local.json` gitignored only when Claude Code saves to it; "For a new command, use a skill" |
| code.claude.com/docs/en/context-window | 2026-11-25 | Partial, grep | What loads before the first prompt |
| code.claude.com/docs/en/plugins/cli-reference | 2026-11-25 | Partial, grep | `claude plugin disable --scope user, project or local` |
| github.com/github/spec-kit `templates/commands/clarify.md`, main branch | 2026-11-25 | Full by research subagent; quotes grep-matched | Rank by impact times uncertainty; recommended option; "why it matters"; 5-question cap per spec; one question at a time; skip trivial style; save after each answer |
| kiro.dev/docs/steering and /docs/specs/feature-specs | 2026-11-25 | Partial, by research subagent | product, tech and structure split of project context; EARS acceptance criteria. Used only to shape categories |
| cursor.com/docs/context/rules | 2026-11-25 | Partial, by research subagent | Anti-patterns (copying style guides, documenting every command). Used only as background |
| AskUserQuestion tool schema in this Claude Code session | 2026-11-25 | Installed | 1 to 4 questions per call, `multiSelect` |
| `~/.claude/plugins/installed_plugins.json`, `~/.claude.json` key names, `~/.claude/projects/` folder names | 2026-11-25 | Installed (key names only) | Layout `inventory.py` reads: `plugins.<id>[].installPath`; per-project `mcpServers` and `disabledMcpServers`; memory folder name is the repo path with each non-alphanumeric character as `-`. Observed on one machine, not documented |
| `claude --version` | 2026-11-25 | Installed | 2.1.332 |
