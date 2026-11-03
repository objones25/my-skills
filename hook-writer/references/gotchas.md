# Hook gotchas

Checked against the Claude Code hooks reference and guide on 2026-10-25 (https://code.claude.com/docs/en/hooks.md and hooks-guide.md). Each entry names the reference section it comes from so you can re-check it with `scripts/fetch_reference.py --section "<name>"`. If the fresh reference says otherwise, follow the reference.

Contents: 1. Exit codes and blocking · 2. Failure and timeouts · 3. JSON output · 4. Matchers and `if` · 5. Coverage holes · 6. Per-event traps · 7. Paths and environment · 8. Scope, merging, trust · 9. Context injection

## 1. Exit codes and blocking

- **Exit 1 does not block.** A non-zero code other than 2 with no valid JSON on stdout is a non-blocking error and the action goes ahead. Policy hooks must `exit 2` (or exit 0 with a deny decision in JSON). Two exceptions: WorktreeCreate and WorktreeRemove fail on any non-zero exit, and on most events a hook whose stdout is valid JSON is judged by the JSON alone, whatever its exit code. (Other exit codes; Exit code output)
- **Exit 0 on PreToolUse is not approval.** It means "no decision"; the call continues through the normal permission flow. Approving requires `permissionDecision: "allow"`. (How a hook resolves)
- **What exit 2 does depends on the event.** It blocks on PreToolUse, UserPromptSubmit, UserPromptExpansion, Stop, SubagentStop, TeammateIdle, TaskCreated, TaskCompleted, ConfigChange (except policy settings), PostToolBatch, PreCompact, PreModelSwitch, Elicitation, ElicitationResult, WorktreeCreate, WorktreeRemove. On PostToolUse and PostToolUseFailure it only shows stderr to Claude (the tool already ran). On SessionStart, SubagentStart, SessionEnd, CwdChanged, FileChanged, PostCompact, PostModelSwitch it shows stderr to the user only. On PermissionRequest, Notification, Setup, StopFailure, PermissionDenied, InstructionsLoaded, MessageDisplay it is ignored. On DirectoryAdded stderr goes to the debug log. (Exit code 2 behavior per event)
- **PermissionRequest ignores exit 2 and discards its stderr.** Deny with `hookSpecificOutput.decision.behavior: "deny"` instead. (PermissionRequest decision control)
- **Claude never sees stderr from an exit-0 hook.** To warn Claude after a tool ran, exit 2 from PostToolUse, or return `additionalContext`. (Exit code 0)
- **Pick one style per hook.** Exit 2 plus JSON still blocks, and JSON fields cannot override the block, even `"allow"`. (Exit code 2)
- **One hook's deny doesn't stop the others.** All matching hooks run in parallel to completion before results merge, so a logging hook still logs a call another hook denied. For PreToolUse the most restrictive answer wins: deny > defer > ask > allow. (Combine results from multiple hooks, in the guide)
- **A hook's `"allow"` can't override deny or ask rules** from settings, and can't suppress prompts for network-path reads or MCP tools that require user interaction. A hook's `"deny"` holds in every permission mode, including bypassPermissions, with one documented exception: on a machine with no managed settings and no Team or Enterprise `/login`, an installed mod can approve a call a PreToolUse hook blocked. (PreToolUse decision control; Hooks and permission modes, in the guide)

## 2. Failure and timeouts

- **Guards fail open by default.** If a hook can't start (wrong path, not executable, missing interpreter), crashes, prints invalid JSON, or times out, most events proceed as if the hook had said nothing. For a guard, set `"onFailure": "block"` on the command or HTTP handler. Requires Claude Code v2.1.295 or later. (Block the action when a hook fails)
- **`onFailure` has no effect** on Stop, SubagentStop, TaskCompleted, TeammateIdle, or async hooks. (Block the action when a hook fails)
- **A timed-out PreToolUse hook does not block** unless `onFailure: "block"` is set. Don't count on a slow hook as a gate. (Timeouts)
- **Default timeouts**: 600 s for command, http, mcp_tool; 30 s for prompt; 60 s for agent. Command/http/mcp_tool drop to 30 s on UserPromptSubmit, PreModelSwitch, PostModelSwitch, and 10 s on MessageDisplay. All SessionEnd hooks share a 1.5 s budget unless a longer per-hook `timeout` raises it (max 60 s). A timed-out UserPromptSubmit hook's `additionalContext` is discarded. (Common fields; UserPromptSubmit)
- **A typo in the script path makes the hook error on every run, and what that does depends on the launcher.** In shell form the shell exits 127, which is a non-blocking error: the action goes ahead and the transcript shows a `hook error` notice with `No such file or directory`. In exec form the interpreter's own exit code counts. Observed, not documented: `python3 missing.py` exits 2 (so it blocks on PreToolUse and other blocking events), `node missing.js` exits 1 (non-blocking). The reference special-cases "exit 2 with a missing-file message on stderr" as non-blocking only on Stop, SubagentStop, TaskCompleted, and a plugin's UserPromptSubmit. Watch for the notice on the first run, run `scripts/check_hooks.py`, and set `onFailure: "block"` on guards so the outcome is deliberate. (Other exit codes; Exit code output)
- **Background processes**: if the script forks work and exits, print the decision first and redirect the child's stdout and stderr (`>/dev/null 2>&1 &`), or Claude Code may block the call with `refusing to honor the suspect capture`. (Hooks that start a background process)
- **Async hooks (`"async": true`) can't decide anything.** `decision`, `permissionDecision`, and `continue` have no effect; output arrives on the next turn; `timeout` isn't enforced (it is on `asyncRewake` hooks); in `-p` mode they're killed at teardown. (Run hooks in the background)

## 3. JSON output

- **Stdout must be exactly one JSON object** (plus whitespace) for a synchronous hook. Output that doesn't start with `{` and end with `}`, including JSON preceded by a line your shell profile echoes, is treated as plain text: your fields are ignored with no error. Output that starts with `{` and ends with `}` but isn't valid JSON is a parse failure, reported as a non-blocking error. Guard profile echoes with `if [[ $- == *i* ]]`. Debug log message: `Hook output does not start with {`. (Exit code 0; Hook JSON has no effect, in the guide)
- **Misplaced fields are silently ignored.** `permissionDecision`, `permissionDecisionReason`, `updatedInput`, and `additionalContext` belong inside `hookSpecificOutput`, which also needs `hookEventName` set to the event name. Debug log message: `Hook JSON output had unrecognized keys`. (Fields at the wrong level, in the guide)
- **Different events use different decision shapes.** PreToolUse: `hookSpecificOutput.permissionDecision`. PermissionRequest: `hookSpecificOutput.decision.behavior`. PostToolUse, PostToolUseFailure, PostToolBatch, Stop, SubagentStop, UserPromptSubmit, UserPromptExpansion, ConfigChange, PreCompact: top-level `decision: "block"` plus `reason`. Look up the event in the Decision control table before writing output. (Decision control)
- **PreToolUse top-level `decision`/`reason` is deprecated** (`approve`/`block` map to allow/deny). Use `hookSpecificOutput`. (PreToolUse decision control)
- **Stop with `decision: "block"` requires `reason`**; Claude reads the reason as why to keep going. (Stop decision control)
- **`updatedInput` replaces the whole tool input**, so include unchanged fields. If two hooks both rewrite the same tool's input, which rewrite wins is not defined; keep it to one. (PreToolUse decision control; Limitations, in the guide)
- **`updatedToolOutput` on PostToolUse only changes what Claude sees**; the tool already ran. For built-in tools a value that doesn't match the tool's output shape is ignored. (PostToolUse decision control)
- **Build JSON with an encoder** (`json.dumps`, `jq -n --arg`), never string concatenation, or quotes in a file path break the parse. (Hook error in output, in the guide)
- **`suppressOutput` does nothing.** (JSON output)
- **10,000-character cap** on each `additionalContext`, `systemMessage`, `initialUserMessage`, and plain stdout. Over the cap, Claude gets a file path and a 2,000-character preview, and isn't asked to read the file. (JSON output)
- **No TTY.** Hooks can't write to `/dev/tty`. Use `systemMessage` to show the user something, or `terminalSequence` for a notification or bell. (Hook input and output)

## 4. Matchers and `if`

- **Matchers are case-sensitive.** `bash` never matches the `Bash` tool. (Hook not firing, in the guide)
- **Exact vs regex is decided by the characters.** Only letters, digits, `_`, `-`, spaces, `,`, `|` means exact names (`Edit|Write`). Any other character makes it an unanchored JavaScript regex, so `Edit.*` also matches `NotebookEdit`; anchor with `^...$`. FileChanged and StopFailure use a narrower exact set (letters, digits, `_`, `|` only; `|` is the only separator). (Matcher patterns)
- **Model-switch matchers use a canonical name.** PreModelSwitch and PostModelSwitch compare the matcher against the canonical name of the target model (aliases, dated IDs, and provider IDs all map to it; a `[1m]` suffix is ignored). If Claude Code can't find a canonical name, every hook runs regardless of matcher. (PreModelSwitch)
- **MCP matchers need the `.*`.** `mcp__github` is an exact string and matches no tool; use `mcp__github__.*`. Plugin-bundled MCP servers use scoped names like `mcp__plugin_<plugin>_<server>__<tool>`. (Match MCP tools)
- **A matcher on an event without matcher support is silently ignored**: UserPromptSubmit, PostToolBatch, Stop, TeammateIdle, TaskCreated, TaskCompleted, WorktreeCreate, WorktreeRemove, MessageDisplay, CwdChanged. (Matcher patterns)
- **`if` only works on tool events** (PreToolUse, PostToolUse, PostToolUseFailure, PermissionRequest, PermissionDenied). On any other event, a handler with `if` never runs. (Common fields)
- **`if` holds exactly one rule.** No `&&`, `||`, or lists; use separate handlers. (Common fields)
- **`if` is best-effort for Bash.** It checks each subcommand, including inside `$()` and backticks, and runs the hook anyway when it can't tell what the command runs. Don't use it as the only enforcement; use permission rules for hard denies. (How if patterns match Bash commands)
- **`Edit(src/**)` matches only `src` under the working directory.** For a `src` directory at any depth write `Edit(**/src/**)`. (Common fields)
- **FileChanged matchers are literal filenames,** split on `|`. A regex or `*` is watched as a file with that literal name. (FileChanged)

## 5. Coverage holes

- **`Edit|Write` misses files changed through Bash** (`sed -i`, codegen, formatters Claude runs) and by outside processes. For full coverage add a Stop hook that scans `git status --porcelain` once per turn, or use FileChanged for specific files. (Filter hooks with matchers, in the guide; PostToolUse)
- **Shell guards should match `Bash|PowerShell`.** Wherever the PowerShell tool is enabled on Windows, Claude routes shell commands through it; on Windows without Git Bash the Bash tool isn't registered at all, so a `Bash`-only hook never fires. An `if` rule names one tool, so `Bash(rm *)` never matches a PowerShell call; add a separate handler with a `PowerShell(...)` rule. Matters for repos with Windows collaborators. (PowerShell; How a hook resolves)
- **PreToolUse doesn't fire for `@`-referenced files** in a prompt, since no tool call happens. Block specific paths with a `Read` deny rule, or set `CLAUDE_CODE_DISABLE_FILE_MENTIONS=1` (v2.1.299+, shell/user/managed settings only) so Claude reads mentioned files through its tools and your PreToolUse hooks run. (PreToolUse)
- **Windows paths arrive with backslashes.** Normalize (`path.replace("\\", "/")`) before matching path segments, and match a segment like `/src/` rather than anchoring with `^`. File-tool paths are always absolute. (PreToolUse input)

## 6. Per-event traps

- **Stop fires every time Claude finishes responding**, including short conversational replies, not only at task completion. It doesn't fire on user interrupt; API errors fire StopFailure instead. Gate expensive checks (for example, only run tests if source files changed). (Limitations, in the guide)
- **Stop loops.** A Stop hook that blocks on a condition Claude can't meet keeps the turn going until the cap of 8 consecutive continuations (resets when Claude calls a tool). Check `stop_hook_active` in the input and allow the stop when it's true, or bound retries another way. (Stop input)
- **Use `last_assistant_message` on Stop/SubagentStop**, not `transcript_path`; the transcript file is written asynchronously and can lag. (Common input fields; Stop input)
- **SessionStart supports only command and mcp_tool handlers,** and its mcp_tool hooks are skipped at launch (MCP servers aren't up yet); they run on later SessionStart events such as after `/clear` or compaction. Setup's mcp_tool hooks are always skipped. (SessionStart; MCP tool hook fields; Setup decision control)
- **Static context belongs in CLAUDE.md,** not a SessionStart hook. Use SessionStart for dynamic state (branch, open issues) or the `compact` matcher to restore context after compaction. (SessionStart)
- **PermissionRequest fires only when a prompt would appear** (or when a call that can't prompt would otherwise be auto-denied), not on every call; for every call use PreToolUse. It doesn't support agent handlers. (PermissionRequest; Prompt-based hooks)
- **UserPromptSubmit runs before every prompt** (including scheduled tasks and background-subagent reports) and blocks the model until it finishes. Keep it fast. (UserPromptSubmit)
- **FileChanged hooks that rewrite the watched file can loop.** Guard on the exact condition you fix so the second run is a no-op. (FileChanged)
- **Agent hooks are experimental**; prefer command hooks for anything that matters. (Agent-based hooks, in the guide)

## 7. Paths and environment

- **Reference scripts through `${CLAUDE_PROJECT_DIR}`,** not relative paths; handlers run in the current directory, which moves when Claude `cd`s. Prefer exec form (`command` + `args`) so the path needs no quoting; in shell form wrap the placeholder in double quotes. (Reference scripts by path)
- **In exec form, `command` is only the executable.** `"command": "node script.js"` with `args` fails; move tokens into `args`. (Exec form and shell form)
- **A script run directly as the command must be executable** (`chmod +x`); the guide's fix for a hook that isn't running at all. A script passed to an interpreter (`python3 x.py`, in either form) doesn't need the bit; that's ordinary OS behavior, not something the hooks docs state. (Hook error in output, in the guide)
- **In a worktree, `${CLAUDE_PROJECT_DIR}` stays at the original project root;** the `cwd` input field follows Claude. Read `cwd` when the hook needs the directory Claude is working in. (Reference scripts by path)
- **`CLAUDE_ENV_FILE`** (persist env vars for later Bash commands) exists only for SessionStart, Setup, CwdChanged, and FileChanged hooks. Append with `>>`. (Persist environment variables)
- **Dependencies**: many examples use `jq`. If you use it, check it's installed; otherwise parse JSON in Python. (Hook error in output, in the guide)

## 8. Scope, merging, trust

- **Hooks merge across all levels;** a project hook and a user hook on the same event both run. An identical handler defined in two settings files runs once. (Hook locations; Hook handler fields)
- **A user-level formatter fires in every repo,** including ones with a different toolchain. Keep toolchain hooks in project settings.
- **Cloud sessions don't read `~/.claude/settings.json`.** (Hook locations)
- **Workspace trust**: interactive sessions hold back all settings-file hooks until the folder is trusted; `-p` and SDK sessions treat the folder as trusted, so a cloned repo's committed hooks run. Review `.claude/` before scripting `claude -p` over someone else's repo, or pass `--settings '{"disableAllHooks": true}'`. (Workspace trust)
- **Skill-frontmatter hooks stay registered for the rest of the session** after the skill is invoked, not just its turn. Use `once: true` (only honored in skill frontmatter) for one-shot hooks. (Hooks in skills and agents)
- **There's no per-hook disable switch.** `disableAllHooks` turns off all of them (except managed hooks); otherwise delete the entry. (Disable or remove hooks)

## 9. Context injection

- **Write `additionalContext` as facts** ("This repo uses `uv run pytest`"), not as system-style commands; text framed as out-of-band instructions can trip prompt-injection defenses and get surfaced to the user instead. (Add context for Claude)
- **Injected text is replayed on resume** for mid-session events, so timestamps or SHAs go stale; SessionStart re-runs on resume and can refresh. (Add context for Claude)
- **Plain stdout reaches Claude only on** UserPromptSubmit, UserPromptExpansion, SessionStart, and PostModelSwitch. Elsewhere it goes to the debug log. (Exit code 0)
