#!/usr/bin/env python3
"""PostToolUse hook (matcher "Edit|Write"): format the file Claude just edited.

Picks a formatter by file extension and runs it only if it is installed
(project-local first, then PATH). Formatting problems never block Claude,
so this hook always exits 0 and prints nothing.

Adapt FORMATTERS to the project's toolchain.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# extension -> (candidate executables relative to project root or on PATH, args before the file path)
FORMATTERS = {
    ".py": ([".venv/bin/ruff", "ruff"], ["format", "--quiet"]),
    ".go": (["gofmt"], ["-w"]),
    ".ts": (["node_modules/.bin/prettier"], ["--write", "--log-level", "warn"]),
    ".tsx": (["node_modules/.bin/prettier"], ["--write", "--log-level", "warn"]),
    ".js": (["node_modules/.bin/prettier"], ["--write", "--log-level", "warn"]),
    ".json": (["node_modules/.bin/prettier"], ["--write", "--log-level", "warn"]),
}


def find_executable(candidates, project_dir):
    for cand in candidates:
        if "/" in cand:
            path = project_dir / cand
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        else:
            found = shutil.which(cand)
            if found:
                return found
    return None


def main():
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    file_path = (event.get("tool_input") or {}).get("file_path")
    if not file_path:
        return 0
    path = Path(file_path.replace("\\", "/"))
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", event.get("cwd", "."))).resolve()

    try:
        path.resolve().relative_to(project_dir)
    except ValueError:
        return 0  # never touch files outside the project

    spec = FORMATTERS.get(path.suffix.lower())
    if spec is None or not path.is_file():
        return 0
    candidates, args = spec
    exe = find_executable(candidates, project_dir)
    if exe is None:
        return 0

    try:
        subprocess.run([exe, *args, str(path)], cwd=project_dir,
                       capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
