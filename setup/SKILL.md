---
name: setup
description: "Set up or re-tune a project for agentic coding with Claude Code before work starts: interview the user on the decisions only they can make, then write a lean CLAUDE.md, path-scoped rules, project settings and permissions, hooks, subagents, project skills, .mcp.json and memory, and switch off plugins, skills, MCP servers and subagents this project does not need. Use when the user types /setup, starts a new project or adopts a cloned repo and wants Claude configured for it, says set this repo up for Claude, configure Claude Code for this project, or what should we decide before you start building, asks which plugins or MCP servers to turn off here, complains that startup context is bloated in one project, or wants an existing setup checked or redone. Not for only generating CLAUDE.md from code (use /init), writing a single hook (hook-writer), building a full skill (skill-builder), changing one setting or permission (update-config), or allowlisting from past prompts (fewer-permission-prompts)."
metadata:
  version: 0.1.0
  test-command: "uv run --no-project --with pytest python -m pytest tests"
---

# Setup

Configure a project so that Claude works well in it from the first prompt: ask what only the user knows, write each decision to the one place Claude Code enforces or loads it, and remove what this project does not need. Built-in `/init` writes a CLAUDE.md from the code (and with `CLAUDE_CODE_NEW_INIT=1` also offers skills and hooks); this skill adds the interview, permissions, subagents, MCP and the context trim.

The rule that decides close calls: **write the smallest configuration that changes what Claude does.** Every line of CLAUDE.md is read on every turn, every skill description costs context, and every hook and deny rule needs maintaining. If the repo already says it, or Claude would do it anyway, leave it out.

`<skill-dir>` below means this skill's folder.

## Workflow

1. **Pick the mode and check scope.** Greenfield (empty or near-empty folder), existing repo, or re-run of an earlier setup. If the user only wants a CLAUDE.md from the code, say `/init` does that and stop. This skill writes project files. Outside the project it writes nothing, except auto memory through the session's own memory instructions (step 5). Never edit `~/.claude/settings.json` or `~/.claude.json` unless the user asks for that file by name.
   *Done when:* you have stated the mode and the project root.

2. **Scan before asking.** Read existing instruction files (`CLAUDE.md`, `CLAUDE.local.md`, `AGENTS.md`, `.cursor/rules/`, `.claude/`), manifests and lockfiles, CI config, README, task runners, test layout and recent git history. Note that `.env` or key files exist, but never open them. Then take a baseline:
   ```bash
   python3 <skill-dir>/scripts/inventory.py          # what loads for this project, by scope
   python3 <skill-dir>/scripts/check_setup.py        # problems in existing config
   ```
   *Done when:* you have a list of known facts, each with the file it came from, a list of gaps, and both baselines (the inventory's estimated total and the checker's error and warning counts).

3. **Interview for the gaps.** Follow `references/interview.md`: rounds of up to 4 AskUserQuestion questions, ranked by impact times uncertainty, a recommended option first, and no question the repo already answers. Greenfield needs about three rounds; an existing repo usually needs one or two, focused on verification, no-touch areas, git, autonomy, never-do rules and what Claude gets wrong here. Ask about context last (step 4).
   *Done when:* every category that would change a file is answered or recorded as open, or the user said stop.

4. **Choose what to trim.** Show the inventory as a short table of what is on and unrelated to this project, with your recommendation, and let the user choose (multi-select). Never offer to turn off `hook-writer`, `skill-builder` or this skill while step 5 still needs them. Map each choice to a per-project switch from `references/context.md` and add it to the step 5 proposal; write nothing yet. Never uninstall anything. User, plugin and claude.ai MCP servers have no settings switch: list them as `/mcp` toggles for the user, because that choice lives in `~/.claude.json`, which Claude Code manages.
   *Done when:* the user picked a list, and each item is either a line in the proposal or a `/mcp` step for the user.

5. **Propose, then write.** Route each decision with the table in `references/artifacts.md`, then show the user the list of files you will create or change, with one line on each and the full text of CLAUDE.md. Get approval, then write. Edit existing files instead of replacing them, and keep what the user wrote.
   - `CLAUDE.md`: under 200 lines (the memory docs' target); only what the code does not tell Claude.
   - `.claude/rules/*.md` with `paths:` for area conventions.
   - `.claude/settings.json` (team) and `.claude/settings.local.json` (personal, gitignored): permissions, `$schema`, switches from step 4.
   - Hooks: load the `hook-writer` skill and give it each automation the interview produced. Do not hand-write hooks here.
   - `.claude/agents/*.md` and `.claude/skills/*/SKILL.md` only for needs the interview named. A skill with scripts or tests goes to `skill-builder`.
   - `.mcp.json` only for services with no usable CLI, with `${VAR}` for every secret.
   - Memory: team facts go in CLAUDE.md. Personal project notes go in `CLAUDE.local.md` (note: its presence stops Claude Code reading `AGENTS.md`). Save personal facts about the user to auto memory only if this session's system prompt describes a memory directory and format; otherwise put them in `CLAUDE.local.md`.
   - Greenfield: also write `SPEC.md` with goal, non-goals, acceptance criteria and first milestones.
   *Done when:* every approved file is written and every decision from the interview is in exactly one place.

6. **Verify.**
   ```bash
   python3 <skill-dir>/scripts/check_setup.py --strict
   python3 <skill-dir>/scripts/inventory.py
   claude mcp list                      # if .mcp.json changed
   ```
   If hooks were added, hook-writer's `check_hooks.py` and one triggering and one non-triggering `dry_run.py` case must pass. Fix every error; explain each remaining warning. Compare the inventory total with the baseline from step 2. The checker does not know the user's never-do list, so check that by hand: for each never-do item, name the deny rule, hook or sandbox setting that enforces it, or say it is enforced only as a guideline.
   *Done when:* the checker exits 0 under `--strict` or each remaining warning has a stated reason, every never-do item is mapped to its enforcement, and you have read the output of every command above.

7. **Hand off.** Tell the user to start a fresh session, because settings, rules and CLAUDE.md load at session start, and to run `/context`, `/mcp` and `/skill-doctor` there to confirm the trim (`/skill-doctor` needs Claude Code 2.1.252 or later and is missing in sessions that skip feature-flag fetching). For greenfield, suggest building from `SPEC.md` in that fresh session.
   *Done when:* the report below has been given.

## Rules

- **Ask only what the repo cannot answer.** The best-practices page: "Only write down what Claude can't find by reading the repo." The same test applies to questions: a question the lockfile answers costs the user a turn and gives you nothing.
- **Must-always goes to enforcement, guidelines to CLAUDE.md.** "If something has to happen every time, make it a hook, a deny rule, or a managed setting. `CLAUDE.md` is for guidelines" (best-practices). A never-do rule written as prose works most of the time, not every time.
- **Keep the enforced set small.** The same page warns that every hook and deny rule will sometimes fire wrongly and needs upkeep. Add one per real risk the user named, not per preference.
- **Never read secrets.** Do not open `.env`, key or credential files; add `Read(...)` deny rules for them instead. Anything read lands in the transcript. Say what such a rule does not cover: the permissions docs note that Read deny rules do not stop a script or subprocess that opens the file itself, and Bash rules match the command text, not every way of running a program. Where that matters, offer the sandbox (`references/artifacts.md`).
- **Personal switches stay personal.** Turning off plugins or skills the user installed for themselves goes in `settings.local.json`, since teammates have different installs. Shared choices go in `settings.json`.
- **Propose before writing, and never overwrite.** The user approves the file list first. Existing CLAUDE.md, settings and rules are edited in place, keeping what the user wrote.
- **Prefer a CLI to an MCP server.** Connected servers "add their tool names and instructions to the context even when you're not using them" (best-practices). If `gh`, `psql` or a cloud CLI does the job, allow the command instead.
- **Know which settings a project file cannot set.** `permissions.defaultMode` of `auto` or `bypassPermissions` is ignored in project and local settings; allow rules and most `env` values wait for workspace trust. Say so instead of writing a setting that will not apply.
- **Re-runs change only what changed.** Diff against the existing files and leave correct parts alone.

## Reference files

| Read | When |
|---|---|
| `references/interview.md` | Step 3: which questions to ask, how to ask them, greenfield vs existing repo |
| `references/artifacts.md` | Step 5: where each decision goes, file formats and examples, commit vs personal, settings traps |
| `references/context.md` | Step 4: what loads at startup, how to measure it, and every per-project switch |
| `references/sources.md` | Checking where a fact came from and when it was read |

## Scripts

| Script | Purpose | Exit codes |
|---|---|---|
| `scripts/inventory.py [DIR] [--home H] [--json]` | Lists plugins, skills, subagents, MCP servers, instruction files and the memory index for a project, with source, state, the setting that decided it, and an estimated size. Prints names only, never config values | 0 printed, 2 could not read a config file |
| `scripts/check_setup.py [DIR] [--strict] [--json]` | Lints CLAUDE.md, settings, rules, subagents, skills and `.mcp.json`: invalid JSON, unignored personal files, broken `@imports`, oversized CLAUDE.md, missing frontmatter, literal secrets. Codes ST001 to ST010. It does not check hooks (hook-writer does), permission rule syntax, or whether the never-do list is enforced | 0 clean, 1 findings, 2 could not check |

The scripts use only the standard library. Run this skill's tests with `uv run --no-project --with pytest python -m pytest tests` from the skill folder.

## Extend this skill

1. A new check: add a failing test with a fixture in `tests/test_check_setup.py`, give it the next `ST` code in `check_setup.py`, and add the code to the Scripts table above.
2. A new per-project switch or config file: confirm it on the Claude Code docs page, add it to `references/context.md` or `references/artifacts.md`, add the page and date to `references/sources.md`, and teach `inventory.py` to read it with a test first.
3. A new interview category: add a row to the question bank in `references/interview.md` with where the answer lands.
4. After any change: run the tests, then skill-builder's `check_skill.py --strict` and `package_skill.py`.

## Reporting

Lead with the state: which files were created or changed, the checker result, and the inventory total before and after (estimated characters and tokens). Then list:

- decisions recorded and where each one lives;
- what was switched off, and the `/mcp` toggles left for the user;
- hooks added, with what each blocks or changes;
- open items: questions not asked or answered, and anything not verified (claude.ai connectors and MCP tool schemas are invisible to the inventory; `/context` in a fresh session is the real measure).
