"""check_hooks.py lints settings and treats its own event table as possibly out of date."""
from __future__ import annotations

import json


def _settings(project, event):
    settings = {"hooks": {event: [{"hooks": [{"type": "command", "command": "true"}]}]}}
    (project / ".claude" / "settings.json").write_text(json.dumps(settings))


def test_valid_settings_pass(project, run_script):
    assert run_script("check_hooks.py", cwd=project).returncode == 0


def test_miscapitalised_event_is_an_error(project, run_script):
    _settings(project, "pretooluse")
    result = run_script("check_hooks.py", cwd=project)
    assert result.returncode == 1
    assert "did you mean 'PreToolUse'" in result.stdout


def test_event_missing_from_the_table_warns_instead_of_failing(project, run_script):
    _settings(project, "SomeNewEvent")
    result = run_script("check_hooks.py", cwd=project)
    assert result.returncode == 0
    assert "SomeNewEvent" in result.stdout
    assert "2026-10-25" in result.stdout


def test_misspelled_event_is_an_error_with_a_suggestion(project, run_script):
    _settings(project, "PreToolUze")
    result = run_script("check_hooks.py", cwd=project)
    assert result.returncode == 1
    assert "did you mean 'PreToolUse'" in result.stdout
