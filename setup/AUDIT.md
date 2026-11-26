# Audit of setup

Date: 2026-11-25. Reviewer: a fresh general-purpose subagent that wrote none of this skill, given the skill folder and the downloaded docs pages, and asked to break the claims and scripts. Each finding was then checked by the author against the docs or by running the case; all 20 were accepted.

## Errors found

| Claim or behaviour | Evidence | Fix |
|---|---|---|
| `check_setup.py` exits 2 on unreadable input | A non-UTF-8 CLAUDE.md raised a `UnicodeDecodeError` traceback | `read_text` raises `CannotCheck`, main returns 2; test `test_non_utf8_instruction_file_cannot_be_checked` |
| `inventory.py` exits 2 on malformed config | A non-list plugin entry, `mcpServers: [1]` and an install record without `installPath` crashed or read `./skills` | `_obj` and `_install_path` validate and raise `CannotCheck`; three tests |
| ST006: skill `name` must match its folder, missing description is an error | skills docs: `name` is optional and `.claude/skills/deploy-staging/` with `name: deploy` runs as `/deploy`; `description` is "Recommended" | Name check removed; missing description is a warning. The old test `test_skill_name_must_match_folder` encoded the wrong rule and was replaced with `test_skill_name_may_differ_from_folder` on that docs evidence |
| Inventory counts instruction files once | With the project under the home folder, `~/.claude/CLAUDE.md` was counted twice (ancestor walk included `.claude/CLAUDE.md`) | Ancestors contribute only `CLAUDE.md` and `CLAUDE.local.md` (memory docs); test |
| Inventory reports memory state | `autoMemoryEnabled: false` still showed memory on | Reads the merged setting and `CLAUDE_CODE_DISABLE_AUTO_MEMORY`; test |
| Inventory reports plugin state | Unlisted plugins ignored `defaultEnabled` (settings reference); plugin MCP servers toggled in `/mcp` stayed "on"; `@imports` were not counted though they load with the importing file | `defaultEnabled` fallback; `plugin:<plugin>:<server>` names checked against `disabledMcpServers` (mcp docs); imports followed up to four hops; three tests |
| AGENTS.md loads when there is no CLAUDE.md | memory docs: also no `CLAUDE.local.md`, and 2.1.277 or later | `artifacts.md` and SKILL.md step 5 state both |
| Deny rules enforce never-do items | permissions docs: Read rules miss subprocesses that open files themselves; Bash rules match command text | Caveat in Rules and `artifacts.md`, with a sandbox `denyRead` example |
| ST003 import parsing | `@docs/a.md.` flagged; a directory passed; a nested 4-backtick fence leaked an import | Trailing punctuation stripped, `is_file()`, fence-length-aware stripping; three tests |
| ST007 secret detection | Tokens in `args` and in a URL query were missed | `args` and `url` scanned for known prefixes and `key=value` literals; three tests including a `${VAR}` control |
| ST009 empty paths | `paths: []` passed | Flow-style lists parsed; two tests |
| `/skill-doctor` availability | skills docs: missing in sessions that skip feature-flag fetching | Qualifier added in SKILL.md and `context.md` |
| Step 1 forbade `~/.claude/`, step 5 wrote auto memory there | Contradiction | Step 1 allows auto memory only through the session's own memory instructions; otherwise `CLAUDE.local.md` |
| Step 4 applied switches before step 5 approval, and could turn off skills step 5 needs | Contradiction | Step 4 adds choices to the proposal only, and protects hook-writer, skill-builder and this skill |
| Step 6 cannot confirm never-do enforcement | Checker does not know the list | Step 6 maps each never-do item to its rule, hook or sandbox setting by hand; Scripts table states what the checker does not check |
| Before/after totals | MCP counted as 0, connectors invisible | Stated in `context.md` and the report section |
| Eval 1 `.env` tripwire lost on commit | Repo `.gitignore` ignores `.env` | Must be added with `git add -f`; `package_skill.py` also drops it, so the eval's `setup_notes` gives the line to recreate it |
| Eval 3 had no fixture | Depended on the grader's machine | `evals/files/static-site` added; `setup_notes` says to grade mechanism, not totals |
| Health-check requests not in the description | Eval 4 prompt | Description now says "wants an existing setup checked or redone"; trigger case added |
| Template structure tests are shallow | Routing test is a substring match; path test sees only backticked paths | Not changed: kept as skill-builder's template. Listed under Not verified |

## How things were checked

- Docs pages downloaded as markdown on 2026-11-25 and every quote grep-matched: memory, best-practices, skills, mcp, settings, settings-reference, permissions, sub-agents, claude-directory, context-window, plugins/cli-reference; spec-kit `clarify.md` from `main`.
- `uv run --no-project --with pytest python -m pytest tests`: see CHANGELOG for the count and machine.
- `pyright` on the folder with the repository's `pyrightconfig.json`: 0 errors.
- Both scripts run on the real My_Skills checkout (names only); the first run exposed the block-scalar description bug, fixed test-first.
- All three eval fixtures run through `check_setup.py` and the results recorded in `verified_facts`.

## Not verified

- Output evals and trigger evals have not been run, with or without the skill.
- The auto memory folder naming (repo path with non-alphanumerics as `-`) is observed on one macOS machine, not documented; `CLAUDE_CODE_PROJECT_DIR_NAME` and `autoMemoryDirectory` relocate it and the inventory does not follow them.
- `inventory.py` ignores managed settings, `--settings` and `--setting-sources`, nested `.claude/skills` in subdirectories, `when_to_use` text that counts toward the 1,536-character cap, and `Skill(...)` deny rules.
- `check_setup.py` does not validate permission rule syntax, CLAUDE.md `@path\ with\ spaces` escapes, or imports without a `.` or `/` (such as `@README`), which it skips to avoid flagging @-mentions.
- spec-kit, Kiro and Cursor sources were read by a research subagent; only the spec-kit quotes were re-matched by the author.
- The skill-builder template's structure tests are shallow (substring routing match, backticked paths only).
