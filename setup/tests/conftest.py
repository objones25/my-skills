"""Shared helpers: run a script from scripts/ as a subprocess, and build throwaway projects."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def isolated_env(tmp_path_factory, monkeypatch):
    """Hide the developer's global git ignore and config, which can ignore .claude/settings.local.json."""
    home = tmp_path_factory.mktemp("home")
    env = {"HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
           "GIT_CONFIG_GLOBAL": str(home / "gitconfig"), "GIT_CONFIG_NOSYSTEM": "1"}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return home


@pytest.fixture
def run_script(isolated_env):
    def run(name, *args, cwd):
        return subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=cwd,
                              capture_output=True, text=True, timeout=60, check=False)
    return run


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, data) -> Path:
    return write(path, json.dumps(data, indent=2))


@pytest.fixture
def clean_project(tmp_path, isolated_env):
    """A project configured the way the setup skill should leave it."""
    write(tmp_path / "CLAUDE.md", "# Demo\n\nRun tests with `uv run pytest`.\n\n@docs/arch.md\n")
    write(tmp_path / "docs" / "arch.md", "Architecture notes.\n")
    write(tmp_path / ".gitignore", ".claude/settings.local.json\nCLAUDE.local.md\n.env\n")
    write_json(tmp_path / ".claude" / "settings.json", {
        "$schema": "https://json.schemastore.org/claude-code-settings.json",
        "permissions": {"allow": ["Bash(uv run pytest *)"], "deny": ["Read(./.env)"]},
        "env": {"PYTHONDONTWRITEBYTECODE": "1"},
    })
    write_json(tmp_path / ".claude" / "settings.local.json", {"enabledPlugins": {"x@y": False}})
    write(tmp_path / ".claude" / "rules" / "api.md", "---\npaths:\n  - \"src/api/**/*.py\"\n---\n\nUse problem details.\n")
    write(tmp_path / ".claude" / "agents" / "reviewer.md",
          "---\nname: reviewer\ndescription: Reviews diffs against SPEC.md.\ntools: Read, Grep, Glob\n---\n\nReview.\n")
    write(tmp_path / ".claude" / "skills" / "release" / "SKILL.md",
          "---\nname: release\ndescription: Cut a release. Use when asked to release.\n---\n\nSteps.\n")
    write_json(tmp_path / ".mcp.json", {"mcpServers": {"db": {
        "command": "npx", "args": ["-y", "some-db-mcp"], "env": {"DB_URL": "${DB_URL}"}}}})
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path
