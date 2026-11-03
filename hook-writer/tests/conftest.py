"""Shared helpers: a throwaway project with a settings file and a hook script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def project(tmp_path):
    """A project whose PreToolUse hook blocks any Bash command containing "rm"."""
    hooks = tmp_path / ".claude" / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "guard.py").write_text(
        "import json, sys\n"
        "command = json.load(sys.stdin)['tool_input'].get('command', '')\n"
        "if 'rm' in command:\n"
        "    print('blocked', file=sys.stderr)\n"
        "    sys.exit(2)\n"
    )
    settings = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [
        {"type": "command", "command": f"{sys.executable} .claude/hooks/guard.py"}]}]}}
    (tmp_path / ".claude" / "settings.json").write_text(json.dumps(settings))
    return tmp_path


@pytest.fixture
def run_script():
    def run(name, *args, cwd):
        return subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=cwd,
                              capture_output=True, text=True, timeout=60)
    return run
