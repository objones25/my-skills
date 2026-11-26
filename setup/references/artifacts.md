# Where each decision goes, and the file formats

Facts here were read from the Claude Code docs on 2026-11-25 (Claude Code 2.1.332 installed); `sources.md` has the pages. Formats change between versions: when a field matters, check the page.

## Contents

- Routing rule
- Commit or keep personal
- CLAUDE.md
- Rules
- Settings and permissions
- Hooks
- Subagents
- Project skills
- MCP servers
- Memory
- Traps

## Routing rule

The best-practices page: "If something has to happen every time, make it a hook, a deny rule, or a managed setting. `CLAUDE.md` is for guidelines, and Claude is allowed to decide a guideline doesn't apply." The memory page: "If an entry is a multi-step procedure or only matters for one part of the codebase, move it to a skill or a path-scoped rule instead."

| The decision is | Put it in |
|---|---|
| A fact every session needs (commands, one-line goal, gotchas) | `CLAUDE.md` |
| A convention for one area of the code | `.claude/rules/<topic>.md` with `paths:` |
| Something that must never happen | `permissions.deny`, or a PreToolUse hook when a rule cannot express it |
| Something that must happen every time (format, typecheck, test gate) | Hook, written with the hook-writer skill |
| A multi-step procedure (release, migration) | Project skill |
| A role with its own tools or a fresh view (reviewer, explorer) | Subagent |
| An external system with no CLI | Project `.mcp.json` |
| The user's own preference, not the team's | `CLAUDE.local.md` or auto memory |

Keep the enforced set small. The best-practices page: "Every hook and deny rule will occasionally fire when it shouldn't, and someone has to maintain each one. If you ever find yourself at forty of them, most are probably preferences."

## Commit or keep personal

From the `.claude` directory page: commit `CLAUDE.md`, `.claude/settings.json`, `.claude/rules/`, `.claude/skills/`, `.claude/agents/`, `.mcp.json`. Keep `.claude/settings.local.json` and `CLAUDE.local.md` out of git; `settings.local.json` is "gitignored when Claude Code saves a setting to it", but a file you write by hand is not, so add both to `.gitignore`. `check_setup.py` flags either one when git does not ignore it (ST002).

Disabling something the user installed at user scope is personal: other people on the team have different plugins. Write those switches to `settings.local.json`. Team-wide choices go in `settings.json`.

## CLAUDE.md

`./CLAUDE.md` or `./.claude/CLAUDE.md`. The memory page: "target under 200 lines per CLAUDE.md file. Longer files consume more context and reduce adherence." `@path` imports resolve relative to the importing file, nest at most four hops, and do not reduce context, because imported files load with the file that imports them.

Test each line with the best-practices question: could Claude find this by reading the code, and would it go and look? If both are yes, leave it out. A skeleton that usually fits in 40 lines:

```markdown
# <project>

<One or two sentences: what it is and who uses it.>

## Commands
- Test: `<exact command>`; one file: `<exact command>`
- Lint and typecheck: `<exact commands>`
- Run locally: `<exact command>`

## Done means
- <the check that decides a change is finished>

## Gotchas
- <things the code does not tell you: live legacy dirs, generated files, misleading patterns>

## Workflow
- <branch, commit and PR rules that differ from the defaults>
```

If the repo already has `AGENTS.md` for other tools, import it from CLAUDE.md (`@AGENTS.md`) rather than copying it. By default Claude Code (2.1.277 or later) reads AGENTS.md only when there is no `CLAUDE.md` or `CLAUDE.local.md` in the working directory or above, so adding a `CLAUDE.local.md` also stops AGENTS.md loading; the **Project instructions** setting in `/config` changes that.

## Rules

`.claude/rules/*.md`, found recursively. Without frontmatter a rule loads at launch like CLAUDE.md. With `paths:` it loads when Claude reads, writes or edits a matching file. "`paths` is the only field Claude Code reads from a rule; any other field is ignored without an error" (ST009 warns on others).

```markdown
---
paths:
  - "src/api/**/*.py"
---

Return errors as RFC 9457 problem details using `api.errors.problem()`.
```

## Settings and permissions

`.claude/settings.json` (team) and `.claude/settings.local.json` (personal). Precedence, highest first: managed, command line, local, project, user. Permission lists merge across files rather than replace; deny is checked first, then ask, then allow. Add `"$schema": "https://json.schemastore.org/claude-code-settings.json"` for editor validation; the settings page warns the schema can lag the CLI.

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allow": ["Bash(uv run pytest *)", "Bash(uv run ruff check *)"],
    "ask": ["Bash(git push *)", "Bash(uv add *)"],
    "deny": ["Read(./.env)", "Read(./.env.*)", "Read(./secrets/**)"]
  }
}
```

What deny rules do not cover, from the permissions page: Read and Edit rules "don't apply to arbitrary subprocesses that read or write files indirectly, like a Python or Node script that opens files itself", and a Bash rule matches the command text Claude writes, not the same program run another way. For a secret that must stay unread, add the sandbox, which enforces at the OS level for sandboxed shell commands:

```json
{
  "sandbox": {
    "enabled": true,
    "filesystem": { "denyRead": ["./.env", "~/.aws/credentials"] }
  }
}
```

Rule forms: `Bash(npm run build)` exact, `Bash(npm run *)` prefix, `Read(./path)`, `Edit(**/migrations/**)`, `WebFetch(domain:example.com)`, `mcp__server__tool`, `Agent(Name)`, `Skill(name)`. A bare tool name in `deny` removes the tool from context entirely. Allow only commands that are safe to repeat; the `fewer-permission-prompts` skill can build the allowlist later from real usage.

## Hooks

Hooks live under `"hooks"` in settings. Do not hand-write them in this skill: load the `hook-writer` skill, which fetches the current reference, writes the handler, and dry-runs it. Hand it the event ("after each edit of a .py file"), the action, and whether it must block.

## Subagents

`.claude/agents/<name>.md`, frontmatter `name` and `description` required (ST005); optional fields include `tools`, `disallowedTools`, `model`, `effort`, `permissionMode`, `skills`, `memory`, `isolation`, `mcpServers`, `hooks`, `maxTurns`. Unknown fields are ignored without an error.

```markdown
---
name: reviewer
description: Reviews a diff against SPEC.md and the project rules. Use after a feature is implemented, before opening a PR.
tools: Read, Grep, Glob, Bash
---

You review changes you did not write. Check every requirement in SPEC.md ...
```

Add one only for a real need. The best-practices page: "Handing the same task to a subagent doesn't make the answer any better. A fresh reviewer does help, because it hasn't seen your reasoning." Combined custom subagent descriptions above 15,000 tokens trigger a startup warning.

## Project skills

`.claude/skills/<name>/SKILL.md`; every frontmatter field is optional and `description` is recommended. Use `disable-model-invocation: true` for procedures with side effects (deploy, release) so only the user starts them. For a small project procedure, write it directly. For a skill with scripts, tests or evals, hand off to the `skill-builder` skill. Each listed skill costs context on every turn, so a project should have few.

```markdown
---
name: release
description: Cut a release of this package. Use when the user asks to release or publish a version.
disable-model-invocation: true
---

1. ...
```

## MCP servers

Prefer a CLI when one exists (`gh`, `aws`, `psql`): the best-practices page notes connected servers "add their tool names and instructions to the context even when you're not using them". For a team server, `.mcp.json` at the root, or `claude mcp add --scope project`:

```json
{
  "mcpServers": {
    "api": {
      "type": "http",
      "url": "${API_BASE_URL:-https://api.example.com}/mcp",
      "headers": { "Authorization": "Bearer ${API_KEY}" }
    }
  }
}
```

`${VAR}` and `${VAR:-default}` expand in `command`, `args`, `env`, `url` and `headers`. Never write a literal token (ST007). Interactive sessions ask each person to approve project servers.

## Memory

- Team knowledge goes in CLAUDE.md or rules, not memory: auto memory lives in `~/.claude/projects/<project>/memory/`, on one machine, outside the repo.
- Auto memory is on by default. The first 200 lines or 25KB of `MEMORY.md` load each session; worktrees and subdirectories of one repo share it. Turn it off for a project with `"autoMemoryEnabled": false` in project settings.
- Claude writes auto memory itself from corrections and preferences. During setup, save only what the interview taught you about the user that is not project fact (their role, how they like to work), using the memory format the system prompt gives. Never copy CLAUDE.md content into memory.
- Personal project notes that should load every session go in `CLAUDE.local.md`.

## Traps

- `permissions.defaultMode` values `auto` and `bypassPermissions` do not take effect from project or local settings.
- `permissions.allow`, `additionalDirectories`, `extraKnownMarketplaces` and most `env` values wait until each teammate trusts the folder.
- In `-p` and SDK sessions the folder counts as trusted, so committed hooks run without a trust dialog (hook-writer's safety note).
