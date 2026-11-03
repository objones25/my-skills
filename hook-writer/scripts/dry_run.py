#!/usr/bin/env python3
"""Run configured hook handlers against a sample event and say what Claude Code would do.

Examples (from the project root):
  dry_run.py --event PreToolUse --tool Bash --command "rm -rf build"
  dry_run.py --event PostToolUse --tool Edit --file src/app.py
  dry_run.py --event Stop --set stop_hook_active=true
  dry_run.py --event SessionStart --source compact
  dry_run.py --event UserPromptSubmit --prompt "deploy to prod"
  dry_run.py --event PreToolUse --tool Bash --command ls --hook "python3 .claude/hooks/guard.py"
  dry_run.py --event PreToolUse --tool Write --input sample.json --show-input

Reads hooks from .claude/settings.json and .claude/settings.local.json (or
--settings FILE ...), picks the groups whose matcher matches and the handlers
whose `if` matches, runs the command handlers with the event JSON on stdin and
CLAUDE_PROJECT_DIR set, then interprets exit code, stdout, and stderr using
the rules in the hooks reference. http, mcp_tool, prompt, and agent handlers
are listed but not run.

This runs the real scripts, so side effects (formatting a file, running
tests) happen for real. Matching `if` rules is approximate; when unsure it
runs the handler, as Claude Code does.
"""
import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hookspec as hs  # noqa: E402

CONTEXT_CAP = 10_000


# ---------- building the event input ----------

def build_input(args, project_dir):
    ev = {
        "session_id": "dry-run-session",
        "transcript_path": str(project_dir / ".claude" / "dry-run-transcript.jsonl"),
        "cwd": str(project_dir),
        "permission_mode": "default",
        "hook_event_name": args.event,
    }
    if args.event in hs.TOOL_EVENTS:
        if not args.tool:
            sys.exit(f"--tool is required for {args.event}")
        ev["tool_name"] = args.tool
        tool_input = {}
        if args.command is not None:
            tool_input = {"command": args.command, "description": "dry run"}
        elif args.file is not None:
            path = Path(args.file)
            path = path if path.is_absolute() else project_dir / path
            tool_input = {"file_path": str(path)}
            if args.tool == "Edit":
                tool_input.update(old_string="old", new_string="new", replace_all=False)
            elif args.tool == "Write":
                tool_input["content"] = path.read_text() if path.is_file() else "dry run content\n"
        ev["tool_input"] = tool_input
        if args.event != "PermissionRequest":
            ev["tool_use_id"] = "toolu_dryrun"
        if args.event == "PostToolUse":
            ev["tool_response"] = ({"stdout": "", "stderr": "", "interrupted": False, "isImage": False}
                                   if args.tool in ("Bash", "PowerShell") else
                                   {"filePath": tool_input.get("file_path", ""), "type": "update"})
    elif args.event in ("Stop", "SubagentStop"):
        ev.update(stop_hook_active=False, last_assistant_message="Done.", background_tasks=[], session_crons=[])
    elif args.event == "SessionStart":
        ev["source"] = args.source or "startup"
    elif args.event == "SessionEnd":
        ev["reason"] = args.source or "other"
    elif args.event in ("PreCompact", "PostCompact"):
        ev["trigger"] = args.source or "auto"
    elif args.event == "UserPromptSubmit":
        ev["prompt"] = args.prompt or "dry run prompt"
    elif args.event == "FileChanged":
        path = Path(args.file or ".env")
        ev.update(file_path=str(path if path.is_absolute() else project_dir / path), event="change")
    elif args.event == "Notification":
        ev.update(message="Claude needs your permission", notification_type=args.source or "permission_prompt")

    if args.input:
        ev.update(json.loads(Path(args.input).read_text()))
    for item in args.set or []:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = ev
        parts = key.split(".")
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    return ev


# ---------- selecting handlers ----------

def glob_to_regex(pattern):
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile("^" + out + "$")


SUBCOMMAND_SPLIT = re.compile(r"&&|\|\||;|\||\n")


def subcommands(command):
    inner = re.findall(r"\$\(([^)]*)\)|`([^`]*)`", command)
    parts = SUBCOMMAND_SPLIT.split(command) + [a or b for a, b in inner]
    for part in parts:
        part = re.sub(r"\$\([^)]*\)|`[^`]*`", "", part).strip()
        while re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\S*\s+", part):
            part = re.sub(r"^[A-Za-z_][A-Za-z0-9_]*=\S*\s+", "", part)
        if part:
            yield part


def if_matches(rule, ev, project_dir):
    """Approximate the permission-rule `if` filter. Returns (runs, note)."""
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*(?:\((.*)\))?\s*$", rule)
    if not m:
        return True, f"couldn't parse if {rule!r}; assuming it runs"
    tool, pattern = m.group(1), m.group(2)
    if tool != ev.get("tool_name"):
        return False, None
    if pattern is None:
        return True, None
    tool_input = ev.get("tool_input") or {}
    if tool in ("Bash", "PowerShell"):
        command = tool_input.get("command", "")
        has_subst = any(s in command for s in ("$(", "`", "$"))
        if has_subst and len(pattern.split()) > 1 and pattern.split()[1] != "*":
            return True, "pattern names more than the command and the command has $()/`/$VAR: Claude Code runs the hook anyway"
        hits = [s for s in subcommands(command) if fnmatch.fnmatchcase(s, pattern) or s == pattern.rstrip(" *")]
        return bool(hits), None
    file_path = tool_input.get("file_path")
    if file_path:
        path = file_path.replace("\\", "/")
        root = str(project_dir).rstrip("/") + "/"
        rel = path[len(root):] if path.startswith(root) else path
        if "/" not in pattern:
            return fnmatch.fnmatchcase(Path(path).name, pattern), "approximate: pattern without / matched against the file name"
        return bool(glob_to_regex(pattern.lstrip("./")).match(rel)), "approximate: path rule matched relative to the project root"
    return True, "no file_path or command in tool_input; assuming it runs"


def select_handlers(configs, ev, project_dir):
    event = ev["hook_event_name"]
    field = hs.MATCHER_FIELD.get(event)
    value = ev.get(field) if field else None
    if event == "FileChanged" and value:
        value = Path(value).name
    for source, config in configs:
        for evt, gi, group, hi, handler in hs.iter_handlers(config):
            if evt != event or hi is None:
                continue
            matcher = None if event in hs.NO_MATCHER else group.get("matcher")
            if not hs.matcher_matches(event, matcher, value):
                continue
            note = None
            if event in ("PreModelSwitch", "PostModelSwitch") and matcher not in (None, "", "*"):
                note = "approximate: matched against the raw to_model; Claude Code uses the canonical model name"
            if "if" in handler:
                if event not in hs.TOOL_EVENTS:
                    yield source, gi, hi, handler, False, "`if` on a non-tool event: Claude Code never runs this handler"
                    continue
                runs, note = if_matches(handler["if"], ev, project_dir)
                if not runs:
                    yield source, gi, hi, handler, False, f"skipped: if {handler['if']!r} doesn't match"
                    continue
            yield source, gi, hi, handler, True, note


# ---------- running ----------

def expand(text, project_dir):
    return text.replace("${CLAUDE_PROJECT_DIR}", str(project_dir)).replace("$CLAUDE_PROJECT_DIR", str(project_dir))


def run_handler(handler, ev, project_dir):
    event = ev["hook_event_name"]
    timeout = handler.get("timeout") or hs.default_timeout(event, handler.get("type", "command"))
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(project_dir))
    if isinstance(handler.get("args"), list):
        argv = [expand(handler["command"], project_dir)] + [expand(a, project_dir) for a in handler["args"]]
    else:
        argv = ["sh", "-c", handler["command"]]
    try:
        proc = subprocess.run(argv, input=json.dumps(ev), capture_output=True, text=True,
                              cwd=project_dir, env=env, timeout=timeout)
        return {"rc": proc.returncode, "out": proc.stdout, "err": proc.stderr, "timed_out": False, "start_failed": False}
    except subprocess.TimeoutExpired as exc:
        return {"rc": None, "out": exc.stdout or "", "err": exc.stderr or "", "timed_out": True, "start_failed": False, "timeout": timeout}
    except OSError as exc:
        return {"rc": None, "out": "", "err": str(exc), "timed_out": False, "start_failed": True}


# ---------- interpreting ----------

def classify_stdout(out):
    s = out.strip()
    if not s:
        return "empty", None
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return ("json", obj) if isinstance(obj, dict) else ("text", None)
        except json.JSONDecodeError:
            return "parse_error", None
    return "text", None


def first_line(text):
    return text.strip().splitlines()[0] if text.strip() else ""


def json_verdict(event, obj, notes):
    """Return a decision tag and a description for a parsed JSON output."""
    unknown = set(obj) - hs.UNIVERSAL_KEYS - {"decision", "reason", "hookSpecificOutput"}
    misplaced = unknown & hs.HSO_ONLY_KEYS
    if misplaced:
        notes.append(f"{sorted(misplaced)} at top level are IGNORED; they belong inside hookSpecificOutput")
    if unknown - misplaced:
        notes.append(f"unrecognized top-level keys ignored: {sorted(unknown - misplaced)}")
    hso = obj.get("hookSpecificOutput") or {}
    if hso and hso.get("hookEventName") != event:
        notes.append(f"hookSpecificOutput.hookEventName is {hso.get('hookEventName')!r}, expected {event!r}")
    for key in ("additionalContext", "systemMessage", "initialUserMessage"):
        val = hso.get(key) if key != "systemMessage" else obj.get(key)
        if isinstance(val, str) and len(val) > CONTEXT_CAP:
            notes.append(f"{key} is {len(val)} chars; over {CONTEXT_CAP} Claude gets a file path and a 2,000-char preview")
    if obj.get("systemMessage"):
        notes.append(f"user sees systemMessage: {first_line(obj['systemMessage'])}")
    if isinstance(hso.get("additionalContext"), str):
        notes.append(f"Claude gets additionalContext ({len(hso['additionalContext'])} chars)")
    if obj.get("continue") is False:
        return "stop", f"Claude stops entirely (continue: false). User sees: {obj.get('stopReason', '')}"

    if event == "PreToolUse":
        decision = hso.get("permissionDecision")
        if decision is None and obj.get("decision") in ("approve", "block"):
            notes.append("top-level decision is deprecated for PreToolUse; use hookSpecificOutput.permissionDecision")
            decision = {"approve": "allow", "block": "deny"}[obj["decision"]]
        reason = hso.get("permissionDecisionReason") or obj.get("reason") or ""
        if "updatedInput" in hso:
            notes.append("updatedInput replaces the whole tool input; include unchanged fields")
        return {
            "deny": ("deny", f"DENIED. Claude reads: {first_line(reason)}"),
            "ask": ("ask", f"user gets a permission prompt: {first_line(reason)}"),
            "allow": ("allow", "allowed without a prompt (deny/ask rules from settings still apply)"),
            "defer": ("defer", "deferred (only works in -p mode)"),
            None: ("none", "no decision; normal permission flow applies"),
        }.get(decision, ("none", f"unknown permissionDecision {decision!r}; ignored"))
    if event == "PermissionRequest":
        d = hso.get("decision") or {}
        behavior = d.get("behavior")
        if behavior == "allow":
            return "allow", "permission granted on the user's behalf (deny rules still apply)"
        if behavior == "deny":
            return "deny", f"permission denied. Claude reads: {d.get('message', '')}"
        return "none", "no decision; the user sees the normal prompt"
    if obj.get("decision") == "block":
        reason = obj.get("reason")
        if event in ("Stop", "SubagentStop"):
            if not reason:
                notes.append("decision block without reason: reason is required for Stop")
            return "block", f"Claude keeps working. It reads: {first_line(reason or '')}"
        if event in ("PostToolUse", "PostToolUseFailure"):
            return "feedback", f"reason shown to Claude next to the tool result (tool already ran): {first_line(reason or '')}"
        return "block", f"blocked: {first_line(reason or '')}"
    if obj.get("decision") is not None:
        notes.append(f"decision {obj['decision']!r} is not a recognized value; only \"block\" is (omit it to allow)")
    if event in ("Stop", "SubagentStop") and isinstance(hso.get("additionalContext"), str):
        return "block", "Claude keeps working on the additionalContext feedback"
    return "none", "no blocking decision"


def interpret(event, handler, res):
    notes = []
    on_fail_block = (handler.get("onFailure") == "block" and event not in hs.ONFAILURE_NO_EFFECT
                     and not handler.get("async") and not handler.get("asyncRewake"))
    exit2_effect = ("block" if event in hs.EXIT2_BLOCKS else
                    "feedback" if event in hs.EXIT2_FEEDBACK_ONLY else "none")

    def failure(desc):
        if on_fail_block:
            tag = "deny" if event == "PermissionRequest" else exit2_effect
            return tag, f"{desc}; onFailure is \"block\", so it acts like exit 2: {tag}"
        if event == "PreModelSwitch" and res["timed_out"]:
            return "block", f"{desc}; a timed-out PreModelSwitch hook blocks the switch"
        return "none", f"{desc}; NON-BLOCKING ERROR, the action goes ahead"

    if res["start_failed"]:
        tag, desc = failure(f"hook failed to start ({res['err']})")
        return tag, desc, notes
    if res["timed_out"]:
        tag, desc = failure(f"timed out after {res['timeout']} s, output discarded")
        return tag, desc, notes

    rc = res["rc"]
    kind, obj = classify_stdout(res["out"])
    if handler.get("async"):
        notes.append("async: Claude Code doesn't wait; decision fields are ignored and additionalContext/systemMessage arrive next turn")

    if rc == 2 and kind == "empty" and event in ("Stop", "SubagentStop", "TaskCompleted") \
            and "No such file" in res["err"]:
        return "none", "exit 2 with a missing-file error on Stop/SubagentStop/TaskCompleted: treated as a NON-BLOCKING ERROR", notes
    if rc == 2:
        reason = first_line(res["err"])
        if "can't open file" in res["err"] or "No such file" in res["err"]:
            notes.append("this exit 2 looks like the interpreter failing to find the script, not a deliberate block")
        if kind == "json":
            json_verdict(event, obj, notes)
            notes.append("exit 2 with JSON: still blocks; JSON can't override the block")
        if event == "PermissionRequest":
            return "none", "exit 2 is IGNORED on PermissionRequest and stderr is discarded; use decision.behavior", notes
        if exit2_effect == "block":
            if event in ("Stop", "SubagentStop", "TeammateIdle", "TaskCompleted"):
                return "block", f"exit 2: Claude keeps working instead of stopping. It reads stderr: {reason}", notes
            return "block" if event != "PreToolUse" else "deny", f"BLOCKED by exit 2. Reason from stderr: {reason}", notes
        if exit2_effect == "feedback":
            return "feedback", f"stderr shown to Claude, but the tool already ran: {reason}", notes
        if event in hs.EXIT2_USER_ONLY:
            return "none", f"exit 2 can't block {event}; stderr shown to the user only", notes
        return "none", f"exit 2 is ignored on {event}", notes

    if rc == 0:
        if res["err"].strip():
            notes.append("stderr on exit 0 goes to the debug log; Claude never sees it")
        if kind == "json":
            tag, desc = json_verdict(event, obj, notes)
            return tag, desc, notes
        if kind == "parse_error":
            return "none", "stdout looks like JSON but doesn't parse: NON-BLOCKING ERROR, fields ignored", notes
        if kind == "text":
            if event in hs.PLAIN_STDOUT_TO_CONTEXT:
                return "context", f"plain stdout added to Claude's context ({len(res['out'].strip())} chars)", notes
            if res["out"].lstrip().startswith("{") or "{" in res["out"]:
                notes.append("stdout has text around the JSON, so Claude Code treats it as plain text and ignores the fields (shell profile echo?)")
            return "none", "plain stdout goes to the debug log only; no decision", notes
        return "none", "no output, exit 0: no decision", notes

    if kind == "json" and not on_fail_block:
        tag, desc = json_verdict(event, obj, notes)
        return tag, f"exit {rc} ignored because stdout is valid JSON; {desc}", notes
    tag, desc = failure(f"exit {rc}" + (f", stderr: {first_line(res['err'])}" if res["err"].strip() else ""))
    if rc == 1:
        notes.append("exit 1 never blocks; use exit 2 for a policy hook")
    return tag, desc, notes


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", required=True, choices=hs.EVENTS)
    ap.add_argument("--tool", help="tool name for tool events, e.g. Bash, Edit, Write, mcp__github__create_issue")
    ap.add_argument("--command", help="shell command for Bash/PowerShell")
    ap.add_argument("--file", help="file path for Edit/Write/Read or FileChanged (relative to project ok)")
    ap.add_argument("--prompt", help="prompt text for UserPromptSubmit")
    ap.add_argument("--source", help="matcher value for SessionStart/SessionEnd/PreCompact/Notification")
    ap.add_argument("--set", action="append", metavar="KEY=JSON", help="override an input field, dotted keys allowed")
    ap.add_argument("--input", help="JSON file merged into the generated input")
    ap.add_argument("--settings", nargs="+", type=Path, help="settings or hooks.json files to read")
    ap.add_argument("--hook", help="test this shell-form command instead of reading settings")
    ap.add_argument("--project-dir", type=Path, default=Path.cwd())
    ap.add_argument("--show-input", action="store_true")
    args = ap.parse_args()

    project_dir = args.project_dir.resolve()
    ev = build_input(args, project_dir)
    if args.show_input:
        print("input:\n" + json.dumps(ev, indent=2) + "\n")

    if args.hook:
        configs = [("--hook", {"hooks": {args.event: [{"hooks": [{"type": "command", "command": args.hook}]}]}})]
    else:
        files = args.settings or [project_dir / ".claude/settings.json", project_dir / ".claude/settings.local.json"]
        configs = []
        for f in files:
            if f.exists():
                configs.append((str(f), json.loads(f.read_text())))

    results = []
    any_handler = False
    for source, gi, hi, handler, runs, note in select_handlers(configs, ev, project_dir):
        any_handler = True
        label = f"{source} :: {args.event}[{gi}].hooks[{hi}]"
        shown = handler.get("command", handler.get("url", handler.get("prompt", "")))
        if handler.get("args"):
            shown += " " + " ".join(handler["args"])
        print(f"- {label}\n  {handler.get('type')}: {shown}")
        if not runs:
            print(f"  {note}\n")
            continue
        if note:
            print(f"  note: {note}")
        if handler.get("type") != "command":
            print(f"  not run: dry_run only executes command handlers\n")
            continue
        res = run_handler(handler, ev, project_dir)
        tag, desc, notes = interpret(args.event, handler, res)
        if args.event in ("Stop", "SubagentStop") and tag == "block":
            if ev.get("stop_hook_active"):
                notes.append("still blocks with stop_hook_active=true: this loops until the 8-continuation cap")
            else:
                notes.append("rerun with --set stop_hook_active=true to check the hook lets Claude stop eventually")
        results.append(tag)
        rc = "start failed" if res["start_failed"] else "timeout" if res["timed_out"] else res["rc"]
        kind, _ = classify_stdout(res["out"])
        print(f"  exit: {rc}   stdout: {kind}" + (f"   stderr: {first_line(res['err'])[:120]}" if res["err"].strip() else ""))
        print(f"  => {desc}")
        for n in notes:
            print(f"     . {n}")
        print()

    if not any_handler:
        print(f"no handler matches this {args.event} input. Check the event name, matcher (case-sensitive), and `if`.")
        return 0
    if args.event == "PreToolUse" and results:
        for tag in ("deny", "defer", "ask", "allow"):
            if tag in results:
                print(f"combined PreToolUse outcome: {tag} (precedence deny > defer > ask > allow)")
                break
        else:
            print("combined PreToolUse outcome: no decision; normal permission flow")
    elif results:
        blocking = [t for t in results if t in ("block", "deny", "stop")]
        print(f"combined outcome: {'blocked/continued by a hook' if blocking else 'nothing blocks'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
