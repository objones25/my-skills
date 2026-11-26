# The setup interview

What to ask before a project starts, how to ask it, and where each answer goes. Sources are listed in `sources.md`; Claude Code docs pages carry no date and were read on 2026-11-25 against Claude Code 2.1.332.

## Contents

- Ask only for gaps
- How to ask
- Question bank by category
- Greenfield vs existing repo
- Ending the interview

## Ask only for gaps

Scan the repo before asking anything. The best-practices page says to "Only write down what Claude can't find by reading the repo, or what it would otherwise spend time rediscovering every session." A question whose answer is in `pyproject.toml`, the lockfile, CI config, the README or an existing CLAUDE.md wastes a round and signals you did not look.

Read, in this order: existing `CLAUDE.md`, `CLAUDE.local.md`, `AGENTS.md`, `.cursor/rules/`, `.claude/`; manifests and lockfiles; CI workflows; README and CONTRIBUTING; `Makefile`, `justfile`, `package.json` scripts; test layout; `git log --oneline -20` and branch names. Never open `.env` or credential files: note that they exist, and propose a deny rule.

Write down what you found as "known" facts with the file each came from. Every remaining category with an unknown answer that would change a config file is a candidate question.

## How to ask

These rules adapt GitHub spec-kit's `/clarify` command and Anthropic's interview pattern. Spec-kit caps one feature spec at 5 questions asked one at a time; a project setup covers more ground, so this skill uses rounds. The round size and count below are this skill's choice, not a sourced number.

- **Rank by impact times uncertainty.** Spec-kit selects questions "by (Impact * Uncertainty)". A question that changes permissions or the verification loop outranks one about naming style.
- **Use AskUserQuestion, up to 4 questions per round,** grouped by theme. That is the tool's own limit per call.
- **Offer choices with a recommended default first,** marked "(Recommended)", with a one-line reason. Spec-kit shows `**Recommended:** Option [X] - <reasoning>` and accepts "yes". Free text only for facts such as a command or a URL.
- **Give each question its "why".** Spec-kit adds a plain "Why it matters" sentence. In AskUserQuestion, put it in the option descriptions.
- **Skip what does not change a file.** Spec-kit skips "trivial stylistic preferences" and anything that would not change implementation or validation. A linter config already answers style.
- **Show the inferred answer and ask for a correction** when the repo strongly suggests one ("CI runs `uv run pytest -q`; is that the command that decides done?"). Confirming is cheaper than asking.
- **Keep a running decision log** in your working notes, so a long interview survives compaction. Spec-kit saves "AFTER each integration to minimize risk of context loss".

## Question bank by category

The "Lands in" column routes each answer; `artifacts.md` explains each artifact. Categories 1 to 5 and 15 to 16 come first for a greenfield project.

| # | Category | Questions to pick from | Lands in |
|---|---|---|---|
| 1 | Goal and users | What is this, for whom, in one sentence? What does done for v1 look like? | CLAUDE.md (two or three lines); SPEC.md for greenfield |
| 2 | Scope and non-goals | What is out of scope? Which paths must not be touched (generated, vendored, legacy that is still live)? | CLAUDE.md; `permissions.deny` for no-touch paths |
| 3 | Verification | Which command decides that a change works? What must pass before done? How do you check UI or behaviour by hand? | CLAUDE.md commands; Stop or PostToolUse hook via hook-writer |
| 4 | Stack and versions | Language, runtime and pinned versions? Framework? Package manager? Is the stack fixed or open? | CLAUDE.md; path-scoped rule for stack specifics |
| 5 | Commands | Install, build, test, one test file, lint, typecheck, run locally? Any slow or destructive ones? | CLAUDE.md; `permissions.allow` for safe read-only commands |
| 6 | Existing conventions (existing repos) | Which patterns must not be copied? What looks dead but is live? What have you corrected Claude on twice? | CLAUDE.md gotchas; path-scoped `.claude/rules/` |
| 7 | Architecture | Module boundaries or layering? Required helpers (errors, logging, DI)? Forbidden patterns? | `.claude/rules/architecture.md` with `paths:` |
| 8 | Data and secrets | Where do secrets live? Which files must never be read? Any PII or regulated data? | `permissions.deny` such as `Read(./.env)`; CLAUDE.md points to `.env.example` |
| 9 | Testing | Framework and layout? TDD expected? What may be mocked? Coverage or mutation targets? | `.claude/rules/testing.md` scoped to test paths |
| 10 | Git | Branch naming, commit style, PR flow? May Claude commit, push, open PRs? Protected branches? | CLAUDE.md; `ask` or `deny` for `git push`; PreToolUse hook if it must never happen |
| 11 | Deploy and environments | How do deploys happen? What needs a migration review? Who signs off? | Project skill with `disable-model-invocation: true`; one CLAUDE.md line |
| 12 | Security and compliance | Auth model? Untrusted input boundaries? Compliance regime? | `.claude/rules/security.md`; deny rules; sandbox settings |
| 13 | Performance | Latency, throughput or memory targets as numbers? Scale assumptions? | CLAUDE.md or a path rule; acceptance criteria in SPEC.md |
| 14 | Dependencies | May Claude add dependencies? Licence or registry limits? Pinning and lockfile policy? | CLAUDE.md or a rule; `ask` for `uv add *`, `npm install *` |
| 15 | Autonomy | Review each edit, accept edits, or auto mode? Will it run unattended? Which commands may run without asking? | `permissions` in `.claude/settings.json`; `defaultMode` note below |
| 16 | Never-do list | What must never happen (force push, edit migrations, touch prod, read secrets)? | `permissions.deny` or a PreToolUse hook, never prose; the sandbox for secrets (deny rules miss scripts that open files themselves) |
| 17 | External services | Issue tracker, database, docs, browser, cloud? Does a CLI exist for it? | CLI `allow` rule if a CLI exists; otherwise project `.mcp.json` |
| 18 | Recurring workflows | Which procedure do you repeat (release, migration, incident, API docs)? | Project skill in `.claude/skills/` |
| 19 | Review and parallel work | Do you want an independent reviewer? Independent tasks that could run in parallel? | Subagent in `.claude/agents/` |
| 20 | Automation | What must happen after every edit or before every stop (format, typecheck, tests)? | Hook, written with hook-writer |
| 21 | Personal preferences | Anything that is yours rather than the team's (verbosity, your sandbox URLs)? | `CLAUDE.local.md` (gitignored) or auto memory |
| 22 | Context budget | Which installed plugins, skills and MCP servers matter here? (Ask after showing the inventory.) | `enabledPlugins`, `skillOverrides`, `disabledMcpjsonServers`, `Agent(...)` deny; `/mcp` toggle |

`defaultMode` of `auto` or `bypassPermissions` does not take effect from project or local settings (settings docs); if the user wants auto mode, tell them it belongs in user settings or the `/permissions` UI.

## Greenfield vs existing repo

**Greenfield.** Most categories are unknown. Round 1: goal, verification, stack, autonomy. Round 2: never-do, git, dependencies, testing. Round 3: external services, recurring workflows, reviewers, automation. Then write `SPEC.md` (goal, non-goals, acceptance criteria, first milestones) along with the config, and recommend building from a fresh session. The best-practices page says: "Once `SPEC.md` exists, start a fresh session to build from it, so the build begins with a clean context and a written definition of done."

**Existing repo.** Most of 4, 5, 7 and 9 are in the code. Ask mainly about 2, 3, 6, 10, 15 and 16, and about what Claude gets wrong here. If a CLAUDE.md exists, edit it rather than replace it; the docs suggest `/init` "suggests improvements rather than overwriting it", and `/doctor prompt-audit` (Claude Code 2.1.283 or later) flags stale or contradictory instructions.

**Re-run.** Run both scripts first and diff against the current files. Ask only about what changed (a new package manager, a new service) and about anything the linter flags.

## Ending the interview

Stop when the user says done, when the remaining categories would not change any file, or after the third round. Record the unasked ones as open items in the report rather than guessing. Before unattended runs, check that the five questions on the best-practices page have answers: what done looks like, how Claude checks its own work, what it may touch, what stops it, and where the result goes and who signs off.
