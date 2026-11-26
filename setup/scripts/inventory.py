#!/usr/bin/env python3
"""List what loads into a Claude Code session for one project, by scope, with a size estimate.

Usage: inventory.py [PROJECT_DIR] [--home DIR] [--json]

Reads user, project and local settings, installed plugins, skills, subagents, MCP server
definitions, instruction files and the auto memory index, and reports each item's source,
whether it is on for this project, and which setting decided that. Sizes are characters of
text that load at startup (skill and agent descriptions, instruction files); tokens are
estimated as characters / 4. MCP tool schemas are not counted: run /context in a session
for the real numbers.

Only names are printed, never configuration values, so secrets in env or headers stay hidden.
Exit codes: 0 inventory printed, 2 could not read the project or a config file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from check_setup import IMPORT, CannotCheck, parse_frontmatter, read_text, strip_code

SKILL_DESCRIPTION_CHARS_MAX = 1536  # skills docs: listing caps each description at 1,536 characters
CHARS_PER_TOKEN = 4
SCOPES = ("user", "project", "local")  # increasing precedence; managed and CLI flags are out of scope
AGENT_DENY = re.compile(r"^Agent\(([^)\s]+)\)$")
IMPORT_HOPS_MAX = 4  # memory docs: imports nest "with a maximum depth of four hops"


@dataclass(frozen=True)
class Item:
    kind: str  # plugin, skill, agent, mcp, instructions, memory
    name: str
    source: str  # user, project, local, or plugin:<id>
    state: str  # on, off, name-only, or not listed
    decided_by: str
    chars: int


def _read_json(path: Path, *, required: bool = False) -> dict:
    if not path.is_file():
        if required:
            raise CannotCheck(f"{path} not found")
        return {}
    try:
        data = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise CannotCheck(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CannotCheck(f"{path} top level is not an object")
    return data


def _obj(value: object, where: str) -> dict:
    """Config sections this script iterates must be objects; anything else means it cannot check."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise CannotCheck(f"{where} is a {type(value).__name__}, expected an object")
    return value


def _settings(home: Path, project: Path) -> dict[str, dict]:
    return {
        "user": _read_json(home / ".claude" / "settings.json"),
        "project": _read_json(project / ".claude" / "settings.json"),
        "local": _read_json(project / ".claude" / "settings.local.json"),
    }


def _merged_map(settings: dict[str, dict], key: str) -> dict[str, tuple[object, str]]:
    """Merge an object-valued setting across scopes; later scopes win per entry."""
    merged: dict[str, tuple[object, str]] = {}
    for scope in SCOPES:
        value = settings[scope].get(key, {})
        for name, entry in value.items() if isinstance(value, dict) else []:
            merged[name] = (entry, scope)
    return merged


def _merged_scalar(settings: dict[str, dict], key: str) -> tuple[object, str]:
    value: object = None
    decided = "default"
    for scope in SCOPES:
        if key in settings[scope]:
            value, decided = settings[scope][key], scope
    return value, decided


def _merged_list(settings: dict[str, dict], *keys: str) -> set[str]:
    """Union a list-valued setting across scopes (permission lists merge rather than override)."""
    out: set[str] = set()
    for scope in SCOPES:
        node: object = settings[scope]
        for key in keys:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if isinstance(node, list):
            out.update(v for v in node if isinstance(v, str))
    return out


def _description(path: Path) -> tuple[str, str]:
    meta = parse_frontmatter(read_text(path)) or {}
    name = meta.get("name") or (path.parent.name if path.name == "SKILL.md" else path.stem)
    return str(name), str(meta.get("description") or "")


def _skills(folder: Path, source: str, overrides: dict[str, tuple[object, str]]) -> list[Item]:
    items = []
    for path in sorted(folder.glob("*/SKILL.md")) if folder.is_dir() else []:
        name, description = _description(path)
        chars = min(len(description), SKILL_DESCRIPTION_CHARS_MAX)
        override, scope = overrides.get(name, ("on", "default"))
        if source.startswith("plugin:"):  # skills docs: plugin skills ignore skillOverrides
            override, scope = "on", "default"
        if override == "off":
            chars = 0
        elif override == "name-only":
            chars = len(name)
        items.append(Item("skill", name, source, str(override), scope, chars))
    return items


def _agents(folder: Path, source: str, denied: set[str]) -> list[Item]:
    items = []
    for path in sorted(folder.glob("*.md")) if folder.is_dir() else []:
        name, description = _description(path)
        off = name in denied
        items.append(Item("agent", name, source, "off" if off else "on",
                          "permissions.deny" if off else "default", 0 if off else len(description)))
    return items


def _plugin_mcp_names(install: Path) -> list[str]:
    names = list(_read_json(install / ".mcp.json").get("mcpServers", {}))
    manifest = _read_json(install / ".claude-plugin" / "plugin.json").get("mcpServers")
    return names + (list(manifest) if isinstance(manifest, dict) else [])


def _install_path(plugin_id: str, installs: object) -> Path:
    if not isinstance(installs, list) or not installs or not isinstance(installs[-1], dict):
        raise CannotCheck(f"installed_plugins.json entry for {plugin_id} is not a list of install records")
    path = installs[-1].get("installPath")
    if not isinstance(path, str) or not path:
        raise CannotCheck(f"installed_plugins.json entry for {plugin_id} has no installPath")
    return Path(path)


def _plugins(home: Path, settings: dict[str, dict], denied: set[str], toggled_off: set[str]) -> list[Item]:
    path = home / ".claude" / "plugins" / "installed_plugins.json"
    installed = _obj(_read_json(path).get("plugins"), f"{path} 'plugins'")
    enabled = _merged_map(settings, "enabledPlugins")
    items: list[Item] = []
    for plugin_id, installs in sorted(installed.items()):
        install = _install_path(plugin_id, installs)
        value, scope = enabled.get(plugin_id, (None, "none"))
        if value is None:  # settings reference: no entry at any scope falls back to defaultEnabled
            default = _read_json(install / ".claude-plugin" / "plugin.json").get("defaultEnabled")
            value, scope = (default, "defaultEnabled") if isinstance(default, bool) else (None, "none")
        state = "on" if value is True else "off" if value is False else "not listed"
        items.append(Item("plugin", plugin_id, "user", state, scope, 0))
        if value is not True:
            continue
        source = f"plugin:{plugin_id}"
        items += _skills(install / "skills", source, {})
        items += _agents(install / "agents", source, denied)
        plugin_name = plugin_id.split("@", 1)[0]
        for server in _plugin_mcp_names(install):
            off = f"plugin:{plugin_name}:{server}" in toggled_off  # mcp docs: plugin:<plugin>:<server>
            items.append(Item("mcp", server, source, "off" if off else "on", "/mcp toggle" if off else "default", 0))
    return items


def _project_entry(home: Path, project: Path) -> dict:
    claude_json = home / ".claude.json"
    projects = _obj(_read_json(claude_json).get("projects"), f"{claude_json} 'projects'")
    return _obj(projects.get(str(project)), f"{claude_json} entry for {project}")


def _toggled_off(home: Path, project: Path) -> set[str]:
    names = _project_entry(home, project).get("disabledMcpServers", [])
    return {n for n in names if isinstance(n, str)} if isinstance(names, list) else set()


def _mcp(home: Path, project: Path, settings: dict[str, dict]) -> list[Item]:
    claude_json = home / ".claude.json"
    project_entry = _project_entry(home, project)
    toggled_off = _toggled_off(home, project)
    rejected = _merged_list(settings, "disabledMcpjsonServers")
    items = []
    for name in _obj(_read_json(claude_json).get("mcpServers"), f"{claude_json} 'mcpServers'"):
        off = name in toggled_off
        items.append(Item("mcp", name, "user", "off" if off else "on", "/mcp toggle" if off else "default", 0))
    for name in _obj(project_entry.get("mcpServers"), f"{claude_json} project 'mcpServers'"):
        items.append(Item("mcp", name, "local", "on", "default", 0))
    for name in _obj(_read_json(project / ".mcp.json").get("mcpServers"), ".mcp.json 'mcpServers'"):
        off = name in rejected
        items.append(Item("mcp", name, "project", "off" if off else "on",
                          "disabledMcpjsonServers" if off else "approval prompt", 0))
    return items


def _label(path: Path, home: Path, project: Path) -> str:
    if path.is_relative_to(project):
        return str(path.relative_to(project))
    if path.is_relative_to(home / ".claude"):
        return f"~/{path.relative_to(home)}"
    return str(path)


def _imports(path: Path, text: str) -> list[Path]:
    targets = (raw.rstrip(".,;:") for raw in IMPORT.findall(strip_code(text)))
    found = (Path(t).expanduser() if t.startswith("~") else path.parent / t for t in targets)
    return [p.resolve() for p in found if p.is_file()]


def _instructions(home: Path, project: Path) -> list[Item]:
    files: list[tuple[Path, str]] = [(home / ".claude" / "CLAUDE.md", "user")]
    rules = home / ".claude" / "rules"
    files += [(p, "user") for p in sorted(rules.rglob("*.md"))] if rules.is_dir() else []
    # memory docs: CLAUDE.md and CLAUDE.local.md load from the working directory and every directory above
    for directory in [project, *project.parents]:
        files += [(directory / "CLAUDE.md", "project"), (directory / "CLAUDE.local.md", "project")]
    files.append((project / ".claude" / "CLAUDE.md", "project"))
    project_rules = project / ".claude" / "rules"
    files += [(p, "project") for p in sorted(project_rules.rglob("*.md"))] if project_rules.is_dir() else []
    seen: set[Path] = set()
    items = []
    queue = [(path, source, 0) for path, source in files]
    while queue:
        path, source, hops = queue.pop(0)
        if not path.is_file() or path.resolve() in seen:
            continue
        seen.add(path.resolve())
        text = read_text(path)
        if hops == 0 and (parse_frontmatter(text) or {}).get("paths"):
            continue  # path-scoped rules load only when a matching file is touched
        items.append(Item("instructions", _label(path, home, project), source, "on",
                          "default" if hops == 0 else "@import", len(text)))
        if hops < IMPORT_HOPS_MAX:
            queue += [(target, source, hops + 1) for target in _imports(path, text)]
    return items


def _memory_root(project: Path) -> Path:
    """Memory docs: all worktrees and subdirectories of one git repository share one auto memory folder."""
    result = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=project,
                            capture_output=True, text=True, check=False)
    return Path(result.stdout.strip()).parent if result.returncode == 0 else project


def _memory(home: Path, project: Path, settings: dict[str, dict]) -> list[Item]:
    # Observed on macOS, Claude Code 2.1.332: the folder name is the path with each non-alphanumeric as '-'.
    # CLAUDE_CODE_PROJECT_DIR_NAME or autoMemoryDirectory can move it; then the index is not found.
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(_memory_root(project)))
    index = home / ".claude" / "projects" / slug / "memory" / "MEMORY.md"
    if not index.is_file():
        return []
    enabled, decided = _merged_scalar(settings, "autoMemoryEnabled")
    if os.environ.get("CLAUDE_CODE_DISABLE_AUTO_MEMORY") == "1":
        enabled, decided = False, "CLAUDE_CODE_DISABLE_AUTO_MEMORY"
    if enabled is False:
        return [Item("memory", "MEMORY.md", "user", "off", decided, 0)]
    lines = read_text(index).splitlines(keepends=True)[:200]  # memory docs: first 200 lines or 25KB load
    return [Item("memory", "MEMORY.md", "user", "on", decided, min(len("".join(lines)), 25_000))]


def inventory(project: Path, home: Path) -> list[Item]:
    settings = _settings(home, project)
    denied = {m.group(1) for r in _merged_list(settings, "permissions", "deny") if (m := AGENT_DENY.match(r))}
    overrides = _merged_map(settings, "skillOverrides")
    return [
        *_plugins(home, settings, denied, _toggled_off(home, project)),
        *_skills(home / ".claude" / "skills", "user", overrides),
        *_skills(project / ".claude" / "skills", "project", overrides),
        *_agents(home / ".claude" / "agents", "user", denied),
        *_agents(project / ".claude" / "agents", "project", denied),
        *_mcp(home, project, settings),
        *_instructions(home, project),
        *_memory(home, project, settings),
    ]


def _print_table(items: list[Item], chars: int) -> None:
    print(f"{'KIND':<13}{'NAME':<40}{'SOURCE':<34}{'STATE':<11}{'DECIDED BY':<24}CHARS")
    for i in items:
        print(f"{i.kind:<13}{i.name[:39]:<40}{i.source[:33]:<34}{i.state:<11}{i.decided_by[:23]:<24}{i.chars}")
    print(f"\nOn for this project: {chars} chars, about {chars // CHARS_PER_TOKEN} tokens (estimate, chars/4).")
    print("MCP tool schemas, the system prompt and built-in tools are not counted; run /context for real numbers.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", nargs="?", default=".", help="project root (default: current directory)")
    parser.add_argument("--home", default=str(Path.home()), help="home directory holding .claude/ and .claude.json")
    parser.add_argument("--json", action="store_true", help="print items and totals as JSON")
    args = parser.parse_args(argv)
    project, home = Path(args.project).resolve(), Path(args.home).expanduser().resolve()
    if not project.is_dir():
        print(f"cannot check: {project} is not a directory", file=sys.stderr)
        return 2
    try:
        items = inventory(project, home)
    except CannotCheck as exc:
        print(f"cannot check: {exc}", file=sys.stderr)
        return 2
    chars = sum(i.chars for i in items if i.state != "off")
    if args.json:
        print(json.dumps({"project": str(project), "items": [asdict(i) for i in items],
                          "totals": {"chars": chars, "tokens_estimate": chars // CHARS_PER_TOKEN}}, indent=2))
    else:
        _print_table(items, chars)
    return 0


if __name__ == "__main__":
    sys.exit(main())
