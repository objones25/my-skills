#!/usr/bin/env python3
"""Lint Claude Code hook configuration and the scripts it points to.

Usage (from the project root):
  check_hooks.py                         .claude/settings.json and .claude/settings.local.json
  check_hooks.py --user                  also ~/.claude/settings.json
  check_hooks.py path/to/hooks.json ...  specific files (settings files or plugin hooks/hooks.json)
  check_hooks.py --project-dir DIR       resolve ${CLAUDE_PROJECT_DIR} against DIR (default: cwd)

Findings are ERROR (the hook won't work as written), WARN (likely wrong or
fails open), or INFO (worth knowing). Exit status is 1 if any ERROR.

Static checks only. Hooks declared in skill or subagent frontmatter are not
parsed. Script heuristics read the hook's script source and can miss things;
confirm behavior with dry_run.py.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hookspec as hs  # noqa: E402

PLACEHOLDERS = ("CLAUDE_PROJECT_DIR", "CLAUDE_PLUGIN_ROOT", "CLAUDE_PLUGIN_DATA")
INTERPRETERS = {"python", "python3", "node", "bash", "sh", "zsh", "uv", "bun", "deno", "ruby", "perl"}


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, where, msg):
        self.items.append((level, where, msg))

    def print(self):
        order = {"ERROR": 0, "WARN": 1, "INFO": 2}
        for level, where, msg in sorted(self.items, key=lambda x: order[x[0]]):
            print(f"{level:5}  {where}\n       {msg}")
        counts = {k: sum(1 for i in self.items if i[0] == k) for k in order}
        print(f"\n{counts['ERROR']} error(s), {counts['WARN']} warning(s), {counts['INFO']} note(s)")
        return counts["ERROR"]


def expand(text, project_dir, plugin_root=None):
    text = text.replace("${CLAUDE_PROJECT_DIR}", str(project_dir)).replace("$CLAUDE_PROJECT_DIR", str(project_dir))
    if plugin_root:
        text = text.replace("${CLAUDE_PLUGIN_ROOT}", str(plugin_root))
    return text


def script_paths(handler, project_dir, plugin_root):
    """Return (path, needs_exec_bit) for files the handler runs that we can locate."""
    found = []
    command = handler.get("command", "")
    args = handler.get("args")
    if isinstance(args, list):
        first = Path(expand(command, project_dir, plugin_root))
        if command and Path(command).name not in INTERPRETERS and ("/" in command or "$" in command):
            found.append((first, True))
        for a in args:
            if isinstance(a, str) and any(p in a for p in PLACEHOLDERS) or (isinstance(a, str) and a.startswith(("./", "/", ".claude/"))):
                p = Path(expand(a, project_dir, plugin_root))
                if p.suffix or p.exists():
                    found.append((p, False))
        return found
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for i, tok in enumerate(tokens):
        if any(p in tok for p in PLACEHOLDERS) or tok.startswith((".claude/", "./.claude/")):
            p = Path(expand(tok, project_dir, plugin_root))
            prev = Path(tokens[i - 1]).name if i else ""
            found.append((p, i == 0 or prev not in INTERPRETERS))
    return found


def check_handler(rep, where, event, group, handler, project_dir, plugin_root, settings_path):
    if not isinstance(handler, dict):
        rep.add("ERROR", where, "handler is not an object")
        return
    htype = handler.get("type")
    if htype not in hs.ALL_TYPES:
        rep.add("ERROR", where, f"unknown handler type {htype!r}; expected one of {sorted(hs.ALL_TYPES)}")
        return
    if htype not in hs.allowed_types(event):
        rep.add("ERROR", where, f"{event} does not support {htype!r} handlers (supports {sorted(hs.allowed_types(event))}); Claude Code skips it")
    if htype == "agent":
        rep.add("INFO", where, "agent hooks are experimental; prefer command hooks for anything that must hold")
    if htype == "mcp_tool" and event == "Setup":
        rep.add("WARN", where, "mcp_tool hooks on Setup are always skipped; use a command hook")
    if htype == "mcp_tool" and event == "SessionStart":
        rep.add("INFO", where, "mcp_tool hooks on SessionStart are skipped at launch (MCP not connected yet); they run after /clear or compaction")

    cond = handler.get("if")
    if cond is not None:
        if event not in hs.TOOL_EVENTS:
            rep.add("ERROR", where, f"`if` only works on tool events; on {event} this handler never runs")
        elif re.search(r"&&|\|\|", cond):
            rep.add("ERROR", where, "`if` holds exactly one permission rule; use one handler per rule")
        else:
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*\(", cond)
            tool = m.group(1) if m else cond.strip()
            matcher = group.get("matcher")
            if tool and not hs.matcher_matches(event, matcher, tool):
                rep.add("WARN", where, f"`if` names tool {tool!r} but the group matcher {matcher!r} never matches it, so this handler never runs")

    on_failure = handler.get("onFailure")
    is_async = handler.get("async") or handler.get("asyncRewake")
    if on_failure is not None:
        if on_failure not in ("continue", "block"):
            rep.add("ERROR", where, f"onFailure must be \"continue\" or \"block\", got {on_failure!r}")
        if event in hs.ONFAILURE_NO_EFFECT or is_async:
            rep.add("WARN", where, f"onFailure has no effect on {event if not is_async else 'async'} hooks")
        if htype not in ("command", "http"):
            rep.add("WARN", where, "onFailure is documented for command and http hooks only")
        rep.add("INFO", where, "onFailure requires Claude Code v2.1.295 or later; check `claude --version`")
    if handler.get("once") is not None:
        rep.add("WARN", where, "`once` is only honored in skill frontmatter; ignored in settings files")
    if is_async and event in (hs.EXIT2_BLOCKS | {"PermissionRequest"}):
        rep.add("INFO", where, "async hooks can't block or decide; output arrives on the next turn")
    if handler.get("suppressOutput") is not None:
        rep.add("INFO", where, "suppressOutput has no effect")

    timeout = handler.get("timeout")
    if timeout is not None and not isinstance(timeout, (int, float)):
        rep.add("ERROR", where, "timeout must be a number of seconds")

    if htype != "command":
        return
    command = handler.get("command")
    if not command:
        rep.add("ERROR", where, "command hook without `command`")
        return
    args = handler.get("args")
    if args is not None:
        if not isinstance(args, list):
            rep.add("ERROR", where, "`args` must be a list")
        elif " " in command and "/" not in command and "\\" not in command:
            rep.add("ERROR", where, f"exec form: `command` must be only the executable, got {command!r}; move the rest into `args`")
        if isinstance(args, list) and "shell" in handler:
            rep.add("INFO", where, "`shell` is ignored when `args` is set")
        exe = Path(command).name
        if "/" not in command and "$" not in command and " " not in command and not shutil.which(command):
            rep.add("WARN", where, f"{exe!r} is not on PATH here; the hook fails to start where it's missing")
    else:
        for ph in PLACEHOLDERS:
            for m in re.finditer(r"\$\{?" + ph + r"\}?", command):
                before = command[max(0, m.start() - 1):m.start()]
                if before not in ('"',):
                    rep.add("WARN", where, f"shell form with unquoted {ph}; a path with spaces breaks. Quote it or switch to exec form (`args`)")
                    break
        if command.lstrip().startswith(".claude/") or command.lstrip().startswith("./"):
            rep.add("WARN", where, "relative script path; handlers run in the current directory, which moves when Claude cd's. Use ${CLAUDE_PROJECT_DIR}")

    sources = [command]
    for path, needs_exec in script_paths(handler, project_dir, plugin_root):
        if "${CLAUDE_PLUGIN_ROOT}" in str(path) or "${CLAUDE_PLUGIN_DATA}" in str(path):
            continue
        if not path.exists():
            rep.add("ERROR", where, f"script not found: {path}. The hook errors on every run; whether that blocks depends on the launcher's exit code (python3 exits 2, sh exits 127, node exits 1)")
            continue
        if needs_exec and not os.access(path, os.X_OK):
            rep.add("ERROR", where, f"script is not executable: {path} (chmod +x, or run it through an interpreter in exec form)")
        try:
            sources.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    check_source(rep, where, event, handler, "\n".join(sources))


def check_source(rep, where, event, handler, src):
    exits_1 = bool(re.search(r"\bexit\s+1\b|sys\.exit\(1\)|process\.exit\(1\)|os\.Exit\(1\)", src))
    guard = exits_1 or bool(re.search(r"exit\s+2|exit\(2\)|[\"']deny\\?[\"']|permissionDecision", src))

    if event in {"PreToolUse", "UserPromptSubmit", "PermissionRequest"} and guard \
            and handler.get("onFailure") != "block" and not handler.get("async"):
        rep.add("WARN", where, "looks like a guard but fails open: if it crashes, can't start, or times out, the action goes ahead. Add \"onFailure\": \"block\"")
    if exits_1 and event in hs.EXIT2_BLOCKS:
        rep.add("WARN", where, "script exits 1 somewhere; exit 1 does NOT block (non-blocking error). Use exit 2 to block")
    if event == "PermissionRequest" and re.search(r"exit\s+2|exit\(2\)", src):
        rep.add("WARN", where, "PermissionRequest ignores exit 2 and discards stderr; deny with hookSpecificOutput.decision.behavior = \"deny\"")
    if event in {"Stop", "SubagentStop"} and re.search(r"exit\s+2|exit\(2\)|[\"']block\\?[\"']", src) \
            and "stop_hook_active" not in src:
        rep.add("WARN", where, "Stop hook can block but never reads stop_hook_active; it may loop until the 8-continuation cap")
    if "/dev/tty" in src:
        rep.add("ERROR", where, "hooks have no controlling terminal; /dev/tty fails. Use systemMessage or terminalSequence")
    if re.search(r"\bjq\b", src) and not shutil.which("jq"):
        rep.add("WARN", where, "uses jq, which isn't on PATH here")
    if event == "PreToolUse" and re.search(r"[\"']decision[\"']\s*:", src) and "permissionDecision" not in src:
        rep.add("WARN", where, "PreToolUse top-level `decision` is deprecated; use hookSpecificOutput.permissionDecision")
    if re.search(r"[\"']additionalContext[\"']", src) and "hookSpecificOutput" not in src:
        rep.add("WARN", where, "additionalContext must be nested inside hookSpecificOutput (with hookEventName); at top level it is silently ignored")


def check_group(rep, where, event, group):
    matcher = group.get("matcher")
    if matcher is not None and not isinstance(matcher, str):
        rep.add("ERROR", where, "matcher must be a string")
        return
    if event in hs.NO_MATCHER and matcher not in (None, "", "*"):
        rep.add("WARN", where, f"{event} has no matcher support; matcher {matcher!r} is silently ignored and the hook fires on every occurrence")
        return
    kind = hs.matcher_kind(event, matcher)
    if event == "FileChanged":
        if kind == "regex" or matcher == "*":
            rep.add("WARN", where, "FileChanged builds its watch list from literal filenames split on |; regex characters or * are watched as literal names")
        return
    if event not in hs.TOOL_EVENTS or kind == "all":
        return
    if kind == "exact":
        for name in hs.exact_names(event, matcher):
            if name.startswith("mcp__") and name.count("__") < 2:
                rep.add("ERROR", where, f"matcher {name!r} is compared as an exact tool name and matches nothing; use {name}__.*")
            elif name not in hs.KNOWN_TOOLS and not name.startswith("mcp__"):
                close = [t for t in hs.KNOWN_TOOLS if t.lower() == name.lower()]
                if close:
                    rep.add("WARN", where, f"{name!r} never matches {close[0]!r}: matchers are case-sensitive")
                else:
                    rep.add("INFO", where, f"{name!r} isn't in this script's list of common tools; check the spelling against the tools reference")
        names = set(hs.exact_names(event, matcher))
        if event == "PostToolUse" and names & {"Edit", "Write"} and not names & {"Bash", "PowerShell"}:
            rep.add("INFO", where, "Edit/Write matcher misses files changed through Bash or outside processes; add a Stop scan or FileChanged if coverage matters")
        if event == "PreToolUse" and "Bash" in names and "PowerShell" not in names:
            rep.add("INFO", where, "matches Bash but not PowerShell; on Windows, shell commands can go through the PowerShell tool and this guard won't see them")
    else:
        try:
            re.compile(matcher)
        except re.error as exc:
            rep.add("ERROR", where, f"matcher is not a valid regex: {exc}")
            return
        if not matcher.startswith("^"):
            hits = sorted(t for t in hs.KNOWN_TOOLS if re.search(matcher, t))
            rep.add("INFO", where, f"regex matcher is unanchored; among built-in tools it matches {hits}")


def check_file(rep, path, project_dir):
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except json.JSONDecodeError as exc:
        rep.add("ERROR", str(path), f"invalid JSON ({exc}); trailing commas and comments aren't allowed. No hooks from this file load")
        return True
    plugin_root = path.parent.parent if path.name == "hooks.json" else None
    hooks = config.get("hooks")
    if hooks is None:
        return True
    if not isinstance(hooks, dict):
        rep.add("ERROR", str(path), "`hooks` must be an object keyed by event name")
        return True
    if config.get("disableAllHooks"):
        rep.add("WARN", str(path), "disableAllHooks is true here; no non-managed hooks run where this file takes precedence")
    for event, gi, group, hi, handler in hs.iter_handlers(config):
        where = f"{path} :: {event}" + (f"[{gi}]" if gi is not None else "") + (f".hooks[{hi}]" if hi is not None else "")
        if event not in hs.EVENTS:
            close = [e for e in hs.EVENTS if e.lower() == event.lower()]
            if close:
                rep.add("ERROR", where, f"unknown event {event!r}; did you mean {close[0]!r}?")
            else:
                rep.add("WARN", where, f"event {event!r} is not in this script's table, transcribed {hs.TRANSCRIBED}. "
                                       "Check the live reference with fetch_reference.py before treating it as a typo.")
            continue
        if gi is None:
            rep.add("ERROR", where, "event value must be a list of matcher groups")
            continue
        if hi is None:
            rep.add("ERROR", where, "matcher group needs a `hooks` list of handlers")
            continue
        if hi == 0:
            check_group(rep, f"{path} :: {event}[{gi}]", event, group)
        check_handler(rep, where, event, group, handler, project_dir, plugin_root, path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--user", action="store_true", help="also check ~/.claude/settings.json")
    ap.add_argument("--project-dir", type=Path, default=Path.cwd())
    args = ap.parse_args()

    project_dir = args.project_dir.resolve()
    files = args.files or [project_dir / ".claude/settings.json", project_dir / ".claude/settings.local.json"]
    if args.user:
        files.append(Path.home() / ".claude/settings.json")

    rep = Report()
    checked = [f for f in files if check_file(rep, f, project_dir)]
    if not checked:
        print("no settings files found: " + ", ".join(str(f) for f in files))
        return 0
    print("checked: " + ", ".join(str(f) for f in checked) + "\n")
    skills = list((project_dir / ".claude/skills").glob("*/SKILL.md")) + list((project_dir / ".claude/agents").glob("*.md"))
    fm_hooks = [s for s in skills if re.search(r"^hooks:", s.read_text(encoding="utf-8", errors="replace").split("\n---", 1)[0], re.M)]
    for s in fm_hooks:
        rep.add("INFO", str(s), "declares frontmatter hooks, which this script doesn't parse; check them with /hooks after invoking it")
    errors = rep.print()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
