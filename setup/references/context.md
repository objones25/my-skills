# Trimming what loads for one project

How to see what a session carries before the first prompt, and how to switch things off for this project only. Read from the Claude Code docs on 2026-11-25 (Claude Code 2.1.332 installed); `sources.md` lists the pages.

## What loads at startup

The context-window page: "Before you type anything: CLAUDE.md, auto memory, MCP tool names, and skill descriptions all load into context." Also loaded: custom subagent descriptions, unscoped rules, and the system prompt and built-in tools, which you cannot change.

- **Skills.** "Every skill in the skill listing adds to your context on every turn, whether or not Claude ever uses it." Each description is cut at 1,536 characters, and the whole listing has a character budget that "scales at 1% of the model's context window"; when it overflows, "Claude Code drops descriptions starting with the skills you invoke least".
- **MCP.** "Only tool names and server instructions load at session start" while tool search defers full schemas; `ENABLE_TOOL_SEARCH=false` loads every schema up front.
- **Subagents.** A startup warning appears when custom subagent descriptions together exceed 15,000 tokens.
- **Instructions.** Every CLAUDE.md from the working directory up, `~/.claude/CLAUDE.md`, and every rule file without `paths:`.

## Measure

1. `python3 <skill-dir>/scripts/inventory.py` from the project root lists plugins, skills, subagents, MCP servers, instruction files and the memory index, each with its source, its state for this project and which setting decided it. Character counts cover descriptions and instruction text; tokens are estimated as characters / 4.
2. In a session, for real numbers: `/context` (the colored grid), `/skill-doctor` (what each skill costs and how often it is used; Claude Code 2.1.252 or later, and missing in sessions that skip feature-flag fetching), `/doctor` (skill listing estimate), `/mcp` (servers and their state).
3. From a shell: `claude mcp list` (shows `⊘ Disabled for this project` and `✘ Rejected` states), `claude plugin list`.

The inventory cannot see claude.ai connectors (they are fetched at runtime) or MCP tool schemas, and counts MCP servers as 0 characters, so the before/after totals measure skills, subagents and instruction files only. Say so when you report it.

## Decide

Turn something off for this project when the interview gives no use for it here: a RunPod plugin in a static website repo, a LaTeX skill in a Go service, a browser MCP server in a library with no UI. Keep anything the user names, and anything cross-cutting they rely on everywhere (their workflow plugin, a docs lookup). Show the list and let the user pick (AskUserQuestion with `multiSelect: true`), because a wrong guess removes a tool they expected.

Do not uninstall anything. Every switch below is per project and reversible.

## Switch off, per project

| Thing | How | File |
|---|---|---|
| A plugin, with all its skills, agents, hooks and servers | `"enabledPlugins": {"name@marketplace": false}`, or `claude plugin disable name --scope local` | `settings.local.json` for your own installs; `settings.json` only for team-installed plugins |
| A user or project skill | `"skillOverrides": {"name": "off"}`; `"name-only"` keeps it listed with no description; `"user-invocable-only"` hides it from Claude but keeps `/name` | settings file. Plugin skills ignore `skillOverrides`: disable the plugin instead |
| A server from the project `.mcp.json` | `"disabledMcpjsonServers": ["name"]` | any settings file; a rejection anywhere applies |
| A user, plugin or claude.ai server | The user toggles it off in `/mcp`. Claude Code records that per project in `~/.claude.json` (`disabledMcpServers`; plugin servers appear as `plugin:<plugin>:<server>`) | not a settings key. Do not edit `~/.claude.json`: tell the user which servers to toggle |
| All claude.ai connectors | `"disableClaudeAiConnectors": true` | settings; `true` in any source wins |
| A subagent | `"permissions": {"deny": ["Agent(name)"]}` | settings file |
| A built-in tool | bare name in `permissions.deny`, such as `"WebSearch"`, removes it from context | settings file |

Example `.claude/settings.local.json` for a web project on a machine with ML plugins installed:

```json
{
  "enabledPlugins": { "runpod@runpod": false, "huggingface-skills@claude-plugins-official": false },
  "skillOverrides": { "transformer-architecture": "off", "pytorch": "off", "lesson-design": "off" },
  "permissions": { "deny": ["Agent(robot-researcher)"] }
}
```

## Verify

Run `inventory.py` again and compare the total with the first run; report both numbers. Then ask the user to start a fresh session and run `/context` and `/mcp`, because settings load at session start and the current session still carries the old set.
