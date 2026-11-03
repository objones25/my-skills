#!/usr/bin/env python3
"""PreToolUse hook (matcher "Bash|PowerShell"): deny shell commands that match a rule.

Defense in depth only. For a hard rule, also add a permission deny rule
(permissions.deny in settings), because hook `if` filtering and parsing
shell text are both best-effort.

Register with "onFailure": "block". Adapt RULES.
"""
import json
import re
import shlex
import sys

SEPARATORS = re.compile(r"&&|\|\||;|\||\n|\$\(|`")


def segments(command):
    """Split a command line into simple commands, tokenized."""
    for part in SEPARATORS.split(command):
        try:
            tokens = shlex.split(part)
        except ValueError:
            tokens = part.split()
        while tokens and (tokens[0] in {"sudo", "command", "env"} or "=" in tokens[0]):
            tokens = tokens[1:]
        if tokens:
            yield tokens


def short_flags(tokens):
    return "".join(t[1:] for t in tokens if t.startswith("-") and not t.startswith("--"))


def is_recursive_force_rm(tokens):
    if tokens[0] != "rm":
        return False
    flags = short_flags(tokens)
    recursive = "r" in flags or "R" in flags or "--recursive" in tokens
    force = "f" in flags or "--force" in tokens
    return recursive and force


def is_force_push(tokens):
    if tokens[:2] != ["git", "push"]:
        return False
    return "--force" in tokens or "-f" in tokens or "f" in short_flags(tokens[2:])


def is_hard_reset(tokens):
    return tokens[:2] == ["git", "reset"] and "--hard" in tokens


RULES = [
    (is_recursive_force_rm, "Recursive force delete is blocked. Delete specific files, or ask the user."),
    (is_force_push, "Force push is blocked. Use --force-with-lease after asking the user."),
    (is_hard_reset, "git reset --hard discards work. Ask the user first."),
]


def main():
    event = json.load(sys.stdin)
    command = (event.get("tool_input") or {}).get("command") or ""
    for tokens in segments(command):
        for check, reason in RULES:
            if check(tokens):
                print(json.dumps({
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }))
                return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
