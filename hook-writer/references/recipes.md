# Recipes

Settings snippets for common project hooks. Each handler script is in `assets/templates/`; copy it into the project's `.claude/hooks/`, adapt the constants at the top, then merge the snippet into `.claude/settings.json`. Merge means: each event is one key inside the single top-level `"hooks"` object; don't replace an existing `"hooks"` block.

All handlers use exec form (`command` is the interpreter, `args` holds the script path), so the path needs no quoting, and since `python3` runs the script it doesn't need an execute bit. They need `python3` on PATH. Note that if the script path is wrong, `python3` exits 2, which blocks on PreToolUse; run `scripts/check_hooks.py` after copying.

Before shipping any of these, re-check the fields used against the fresh reference (`scripts/fetch_reference.py`), then run `scripts/check_hooks.py` and `scripts/dry_run.py`.

## Format after edits

Fires after every successful Edit or Write; the script exits early for unknown extensions, missing formatters, and files outside the project.

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python3",
            "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/format_on_edit.py"],
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```

Narrower: to format only one language, put the filter in `if` instead of the script, one handler per tool, e.g. `"if": "Edit(**/*.py)"` and a second handler with `"if": "Write(**/*.py)"`. `if` holds one rule, so two tools need two handlers.

Gap: files Claude changes through Bash aren't seen. If that matters, add a Stop hook that formats whatever `git status --porcelain` lists.

## Protect paths

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python3",
            "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/protect_paths.py"],
            "onFailure": "block"
          }
        ]
      }
    ]
  }
}
```

`onFailure: "block"` (v2.1.295+) makes a broken guard block edits rather than silently allow them. Pair with permission rules for secrets: `@`-mentioned files are inserted without a tool call, so no PreToolUse hook sees them. A `Read` deny rule blocks specific paths from `@`, and `CLAUDE_CODE_DISABLE_FILE_MENTIONS=1` (v2.1.299+, set in the shell, user, or managed settings) makes Claude read mentioned files through its tools so hooks run.

## Guard shell commands

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|PowerShell",
        "hooks": [
          {
            "type": "command",
            "command": "python3",
            "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/bash_guard.py"],
            "onFailure": "block"
          }
        ]
      }
    ]
  }
}
```

To spawn the guard only for commands that could match, add one handler per rule with an `if`, such as `"if": "Bash(rm *)"` and `"if": "Bash(git *)"`. An `if` rule names one tool, so a `Bash(...)` rule never matches a PowerShell call; if PowerShell coverage matters, add matching `PowerShell(...)` handlers (for example `"if": "PowerShell(Remove-Item *)"`) or keep the unfiltered handler. Because `if` is best-effort, keep a permission deny rule as the real enforcement for anything that must never run.

## Test gate before Claude finishes

Stop has no matcher, so all narrowing happens in the script: it skips unless source files changed, skips if the tree is unchanged since the last pass, and gives up after a few attempts.

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3",
            "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/stop_test_gate.py"],
            "timeout": 330,
            "statusMessage": "Running tests before finishing"
          }
        ]
      }
    ]
  }
}
```

Set `timeout` above the script's own test timeout so the script, not Claude Code, decides what happens on a slow run. `onFailure` has no effect on Stop.

For slow suites, the alternative is an async PostToolUse hook that runs tests in the background and reports through `additionalContext` on the next turn (see "Run tests after file changes" in the reference). It can't stop Claude from finishing.

## Context at session start and after compaction

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|compact",
        "hooks": [
          {
            "type": "command",
            "command": "python3",
            "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/session_context.py"],
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

SessionStart runs on every session, so keep it fast. Only for dynamic state; fixed conventions go in CLAUDE.md.

## Desktop notification (user level, macOS)

Belongs in `~/.claude/settings.json`, since it's about the machine, not the repo.

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "permission_prompt|idle_prompt",
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"Claude Code needs your attention\" with title \"Claude Code\"'"
          }
        ]
      }
    ]
  }
}
```

If nothing appears, Script Editor needs notification permission in System Settings > Notifications (a note under "Get notified when Claude needs input" in the guide).

## Hook that lives in a skill

For a hook that should exist only once a particular workflow starts, declare it in that skill's frontmatter. It registers when the skill is invoked and stays for the rest of the session; `once: true` removes it after its first successful run.

```yaml
---
name: db-migration
description: ...
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          if: "Bash(psql *)"
          command: "python3"
          args: ["${CLAUDE_PROJECT_DIR}/.claude/hooks/check_migration_target.py"]
---
```

`scripts/check_hooks.py` does not parse frontmatter hooks; check them with `/hooks` after invoking the skill.
