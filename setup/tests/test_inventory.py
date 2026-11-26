"""inventory.py lists what loads into a session for a project, by scope, without printing config values."""
from __future__ import annotations

import json

import pytest
from conftest import write, write_json

SECRET = "sk-live-do-not-print-0123456789"


def _skill(folder, name, description):
    write(folder / name / "SKILL.md", f"---\nname: {name}\ndescription: {description}\n---\n\nBody.\n")


@pytest.fixture
def world(tmp_path):
    """A fake home with one plugin, user skills, agents and MCP servers, plus a project."""
    home = tmp_path / "home"
    project = tmp_path / "work" / "my_app"
    claude = home / ".claude"
    plugin = claude / "plugins" / "cache" / "mk" / "tools" / "1.0.0"
    _skill(plugin / "skills", "plug-skill", "A plugin skill. Use when plugging.")
    write(plugin / "agents" / "plug-agent.md", "---\nname: plug-agent\ndescription: Plugin agent.\n---\n")
    write_json(plugin / ".mcp.json", {"mcpServers": {"plug-mcp": {"command": "x", "env": {"K": SECRET}}}})
    write_json(claude / "plugins" / "installed_plugins.json", {"version": 2, "plugins": {
        "tools@mk": [{"scope": "user", "installPath": str(plugin), "version": "1.0.0"}],
        "other@mk": [{"scope": "user", "installPath": str(tmp_path / "nowhere"), "version": "1"}]}})
    write_json(claude / "settings.json", {"enabledPlugins": {"tools@mk": True, "other@mk": False}})
    _skill(claude / "skills", "alpha", "Alpha skill. Use when alpha.")
    _skill(claude / "skills", "beta", "Beta skill. Use when beta.")
    write(claude / "agents" / "helper.md", "---\nname: helper\ndescription: Helps.\n---\n")
    write(claude / "CLAUDE.md", "global rules\n")
    write(claude / "rules" / "style.md", "Prefer small functions.\n")
    write(claude / "rules" / "py.md", "---\npaths:\n  - \"**/*.py\"\n---\nPython only.\n")
    write_json(home / ".claude.json", {
        "mcpServers": {"user-mcp": {"command": "y", "env": {"TOKEN": SECRET}}},
        "projects": {str(project): {"mcpServers": {"local-mcp": {"command": "z"}},
                                    "disabledMcpServers": ["user-mcp"]}}})
    write(project / "CLAUDE.md", "project rules\n")
    write_json(project / ".mcp.json", {"mcpServers": {"db": {"command": "d"}, "search": {"command": "s"}}})
    return {"home": home, "project": project, "plugin": plugin}


def _inventory(run_script, world, *extra):
    result = run_script("inventory.py", str(world["project"]), "--home", str(world["home"]), "--json", *extra,
                        cwd=world["project"])
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _item(data, kind, name):
    matches = [i for i in data["items"] if i["kind"] == kind and i["name"] == name]
    assert len(matches) == 1, (kind, name, data["items"])
    return matches[0]


def test_lists_user_and_plugin_skills_with_their_source(run_script, world):
    data = _inventory(run_script, world)
    assert _item(data, "skill", "alpha")["source"] == "user"
    assert _item(data, "skill", "plug-skill")["source"] == "plugin:tools@mk"


def test_disabled_plugin_contributes_nothing(run_script, world):
    data = _inventory(run_script, world)
    assert _item(data, "plugin", "other@mk")["state"] == "off"
    assert not [i for i in data["items"] if i["source"] == "plugin:other@mk"]


def test_local_settings_override_user_plugin_choice(run_script, world):
    write_json(world["project"] / ".claude" / "settings.local.json", {"enabledPlugins": {"tools@mk": False}})
    data = _inventory(run_script, world)
    assert _item(data, "plugin", "tools@mk")["state"] == "off"
    assert _item(data, "plugin", "tools@mk")["decided_by"] == "local"
    assert not [i for i in data["items"] if i["source"] == "plugin:tools@mk"]


def test_skill_override_off_and_name_only(run_script, world):
    write_json(world["project"] / ".claude" / "settings.json",
               {"skillOverrides": {"alpha": "off", "beta": "name-only", "plug-skill": "off"}})
    data = _inventory(run_script, world)
    assert _item(data, "skill", "alpha")["state"] == "off"
    assert _item(data, "skill", "alpha")["chars"] == 0
    beta = _item(data, "skill", "beta")
    assert beta["state"] == "name-only"
    assert beta["chars"] == len("beta")
    # the skills docs: "Plugin skills are not affected by skillOverrides"
    assert _item(data, "skill", "plug-skill")["state"] == "on"


def test_skill_description_chars_are_counted_and_capped(run_script, world):
    _skill(world["home"] / ".claude" / "skills", "wordy", "x" * 2000)
    data = _inventory(run_script, world)
    assert _item(data, "skill", "alpha")["chars"] == len("Alpha skill. Use when alpha.")
    assert _item(data, "skill", "wordy")["chars"] == 1536


def test_denied_agent_is_marked_off(run_script, world):
    write_json(world["project"] / ".claude" / "settings.json", {"permissions": {"deny": ["Agent(helper)"]}})
    data = _inventory(run_script, world)
    assert _item(data, "agent", "helper")["state"] == "off"
    assert _item(data, "agent", "plug-agent")["state"] == "on"


def test_mcp_servers_from_every_scope_with_project_state(run_script, world):
    write_json(world["project"] / ".claude" / "settings.json", {"disabledMcpjsonServers": ["search"]})
    data = _inventory(run_script, world)
    assert _item(data, "mcp", "user-mcp")["state"] == "off"
    assert _item(data, "mcp", "user-mcp")["decided_by"] == "/mcp toggle"
    assert _item(data, "mcp", "local-mcp")["source"] == "local"
    assert _item(data, "mcp", "plug-mcp")["source"] == "plugin:tools@mk"
    assert _item(data, "mcp", "db")["source"] == "project"
    assert _item(data, "mcp", "search")["state"] == "off"


def test_never_prints_config_values(run_script, world):
    for extra in ((), ("--json",)):
        result = run_script("inventory.py", str(world["project"]), "--home", str(world["home"]), *extra,
                            cwd=world["project"])
        assert result.returncode == 0
        assert SECRET not in result.stdout + result.stderr


def test_instruction_files_unscoped_rules_only(run_script, world):
    data = _inventory(run_script, world)
    names = {i["name"] for i in data["items"] if i["kind"] == "instructions"}
    assert "~/.claude/CLAUDE.md" in names
    assert "~/.claude/rules/style.md" in names
    assert "~/.claude/rules/py.md" not in names  # path-scoped rules load on demand
    assert "CLAUDE.md" in names


def test_auto_memory_index_found_by_project_slug(run_script, world):
    slug = "".join(c if c.isalnum() else "-" for c in str(world["project"]))
    write(world["home"] / ".claude" / "projects" / slug / "memory" / "MEMORY.md", "- [a](a.md) - hook\n")
    data = _inventory(run_script, world)
    assert _item(data, "memory", "MEMORY.md")["chars"] == len("- [a](a.md) - hook\n")


def test_totals_estimate_tokens(run_script, world):
    data = _inventory(run_script, world)
    on_chars = sum(i["chars"] for i in data["items"] if i["state"] != "off")
    assert data["totals"]["chars"] == on_chars
    assert data["totals"]["tokens_estimate"] == on_chars // 4


def test_text_output_has_a_table_and_points_to_context(run_script, world):
    result = run_script("inventory.py", str(world["project"]), "--home", str(world["home"]), cwd=world["project"])
    assert "plug-skill" in result.stdout
    assert "/context" in result.stdout


def test_malformed_user_settings_cannot_be_checked(run_script, world):
    write(world["home"] / ".claude" / "settings.json", "{not json")
    result = run_script("inventory.py", str(world["project"]), "--home", str(world["home"]), cwd=world["project"])
    assert result.returncode == 2
    assert "settings.json" in result.stderr


def test_missing_project_cannot_be_checked(run_script, world, tmp_path):
    result = run_script("inventory.py", str(tmp_path / "nope"), "--home", str(world["home"]), cwd=tmp_path)
    assert result.returncode == 2


def test_block_scalar_descriptions_are_read(run_script, world):
    folder = world["home"] / ".claude" / "skills"
    write(folder / "folded" / "SKILL.md",
          "---\nname: folded\ndescription: >-\n  First line\n  second line.\nmetadata:\n  v: 1\n---\n")
    write(folder / "literal" / "SKILL.md", "---\nname: literal\ndescription: |\n  One.\n  Two.\n---\n")
    data = _inventory(run_script, world)
    assert _item(data, "skill", "folded")["chars"] == len("First line second line.")
    assert _item(data, "skill", "literal")["chars"] == len("One.\nTwo.")


def test_auto_memory_is_shared_by_the_whole_git_repo(run_script, world):
    import subprocess
    repo = world["project"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    sub = repo / "packages" / "api"
    sub.mkdir(parents=True)
    slug = "".join(c if c.isalnum() else "-" for c in str(repo))
    write(world["home"] / ".claude" / "projects" / slug / "memory" / "MEMORY.md", "- fact\n")
    result = run_script("inventory.py", str(sub), "--home", str(world["home"]), "--json", cwd=sub)
    data = json.loads(result.stdout)
    assert _item(data, "memory", "MEMORY.md")["chars"] == len("- fact\n")


def test_user_claude_md_is_not_counted_twice_when_project_is_under_home(run_script, world, tmp_path):
    project = world["home"] / "work" / "proj"
    write(project / "README.md", "x\n")
    result = run_script("inventory.py", str(project), "--home", str(world["home"]), "--json", cwd=project)
    data = json.loads(result.stdout)
    user_file = world["home"] / ".claude" / "CLAUDE.md"
    labels = [i["name"] for i in data["items"] if i["kind"] == "instructions"]
    assert "~/.claude/CLAUDE.md" in labels
    assert str(user_file) not in labels


def test_auto_memory_disabled_in_project_settings_is_off(run_script, world):
    slug = "".join(c if c.isalnum() else "-" for c in str(world["project"]))
    write(world["home"] / ".claude" / "projects" / slug / "memory" / "MEMORY.md", "- fact\n")
    write_json(world["project"] / ".claude" / "settings.json", {"autoMemoryEnabled": False})
    data = _inventory(run_script, world)
    memory = _item(data, "memory", "MEMORY.md")
    assert memory["state"] == "off"
    assert memory["decided_by"] == "project"


def test_unlisted_plugin_follows_its_default_enabled(run_script, world, tmp_path):
    plugin = tmp_path / "p2"
    _skill(plugin / "skills", "p2-skill", "P2 skill.")
    write_json(plugin / ".claude-plugin" / "plugin.json", {"name": "p2", "defaultEnabled": True})
    installed = json.loads((world["home"] / ".claude" / "plugins" / "installed_plugins.json").read_text())
    installed["plugins"]["p2@mk"] = [{"scope": "user", "installPath": str(plugin)}]
    write_json(world["home"] / ".claude" / "plugins" / "installed_plugins.json", installed)
    data = _inventory(run_script, world)
    assert _item(data, "plugin", "p2@mk")["state"] == "on"
    assert _item(data, "plugin", "p2@mk")["decided_by"] == "defaultEnabled"
    assert _item(data, "skill", "p2-skill")["state"] == "on"


def test_plugin_mcp_server_toggled_off_in_mcp_is_off(run_script, world):
    claude_json = json.loads((world["home"] / ".claude.json").read_text())
    claude_json["projects"][str(world["project"])]["disabledMcpServers"].append("plugin:tools:plug-mcp")
    write_json(world["home"] / ".claude.json", claude_json)
    data = _inventory(run_script, world)
    assert _item(data, "mcp", "plug-mcp")["state"] == "off"


def test_imported_files_are_counted(run_script, world):
    write(world["project"] / "CLAUDE.md", "project rules\n@docs/extra.md\n")
    write(world["project"] / "docs" / "extra.md", "extra text\n")
    data = _inventory(run_script, world)
    assert _item(data, "instructions", "docs/extra.md")["chars"] == len("extra text\n")


def test_malformed_plugin_entry_cannot_be_checked(run_script, world):
    installed = json.loads((world["home"] / ".claude" / "plugins" / "installed_plugins.json").read_text())
    installed["plugins"]["tools@mk"] = "not-a-list"
    write_json(world["home"] / ".claude" / "plugins" / "installed_plugins.json", installed)
    result = run_script("inventory.py", str(world["project"]), "--home", str(world["home"]), cwd=world["project"])
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_non_object_mcp_servers_cannot_be_checked(run_script, world):
    write_json(world["home"] / ".claude.json", {"mcpServers": [1]})
    result = run_script("inventory.py", str(world["project"]), "--home", str(world["home"]), cwd=world["project"])
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_plugin_without_install_path_cannot_be_checked(run_script, world):
    installed = json.loads((world["home"] / ".claude" / "plugins" / "installed_plugins.json").read_text())
    installed["plugins"]["tools@mk"] = [{}]
    write_json(world["home"] / ".claude" / "plugins" / "installed_plugins.json", installed)
    result = run_script("inventory.py", str(world["project"]), "--home", str(world["home"]), cwd=world["project"])
    assert result.returncode == 2
