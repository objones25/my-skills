#!/usr/bin/env python3
"""Stop hook: keep Claude working until the test suite passes.

Fires only when it needs to:
  * skips unless tracked source files have uncommitted changes,
  * skips if the working tree is unchanged since the last passing run,
  * gives up after MAX_RETRIES consecutive blocks in one turn, so a test
    Claude can't fix doesn't hold the session (Claude Code also caps
    consecutive Stop-hook continuations at 8).

Adapt TEST_CMD, SOURCE_SUFFIXES, MAX_RETRIES.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TEST_CMD = ["uv", "run", "pytest", "-x", "-q"]
SOURCE_SUFFIXES = {".py"}
MAX_RETRIES = 3
TEST_TIMEOUT_S = 300
TAIL_CHARS = 3000


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout


def state_file(session_id):
    return Path(tempfile.gettempdir()) / f"hook-stop-gate-{session_id}.json"


def load_state(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def main():
    event = json.load(sys.stdin)
    project = os.environ.get("CLAUDE_PROJECT_DIR", event.get("cwd", "."))
    sf = state_file(event.get("session_id", "unknown"))
    state = load_state(sf)

    if not event.get("stop_hook_active"):
        state["retries"] = 0  # a fresh turn, not a continuation we caused

    status = git(["status", "--porcelain"], project)
    changed = [line[3:] for line in status.splitlines() if line[3:].strip()]
    if not any(Path(p.strip('"')).suffix in SOURCE_SUFFIXES for p in changed):
        return 0

    fingerprint = hashlib.sha256((status + git(["diff"], project)).encode()).hexdigest()
    if state.get("last_pass") == fingerprint:
        return 0

    if state.get("retries", 0) >= MAX_RETRIES:
        state["retries"] = 0
        sf.write_text(json.dumps(state))
        print(json.dumps({"systemMessage": f"Stop test gate gave up after {MAX_RETRIES} attempts; tests still fail."}))
        return 0

    try:
        result = subprocess.run(TEST_CMD, cwd=project, capture_output=True, text=True, timeout=TEST_TIMEOUT_S)
        passed = result.returncode == 0
        output = (result.stdout + result.stderr)[-TAIL_CHARS:]
    except subprocess.TimeoutExpired:
        passed, output = False, f"Test command timed out after {TEST_TIMEOUT_S}s."
    except OSError as exc:
        print(json.dumps({"systemMessage": f"Stop test gate could not run {TEST_CMD[0]}: {exc}"}))
        return 0

    if passed:
        state.update(last_pass=fingerprint, retries=0)
        sf.write_text(json.dumps(state))
        return 0

    state["retries"] = state.get("retries", 0) + 1
    sf.write_text(json.dumps(state))
    print(json.dumps({
        "decision": "block",
        "reason": f"The test suite ({' '.join(TEST_CMD)}) fails. Fix the failures before finishing.\n\n{output}",
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
