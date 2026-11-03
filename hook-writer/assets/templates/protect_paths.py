#!/usr/bin/env python3
"""PreToolUse hook (matcher "Edit|Write"): deny edits to protected paths.

Register it with "onFailure": "block" so a crash or a missing interpreter
blocks the edit instead of letting it through.

Patterns are glob patterns (fnmatch) tested against the path relative to
the project root and against the bare file name. Adapt PROTECTED.
"""
import fnmatch
import json
import os
import sys
from pathlib import PurePosixPath

PROTECTED = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    ".git/*",
    "uv.lock",
    "package-lock.json",
    "go.sum",
]


def deny(reason):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    return 0


def main():
    event = json.load(sys.stdin)
    file_path = (event.get("tool_input") or {}).get("file_path")
    if not file_path:
        return 0
    file_path = file_path.replace("\\", "/")
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", event.get("cwd", "")).replace("\\", "/")

    rel = file_path
    if project_dir and file_path.startswith(project_dir.rstrip("/") + "/"):
        rel = file_path[len(project_dir.rstrip("/")) + 1:]
    name = PurePosixPath(file_path).name

    for pattern in PROTECTED:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
            return deny(f"{rel} is protected (pattern '{pattern}'). "
                        "Ask the user before changing it, or edit the source it is generated from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
