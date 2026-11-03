"""Event tables and matcher semantics shared by check_hooks.py and dry_run.py.

Transcribed from the Claude Code hooks reference on 2026-10-25
(https://code.claude.com/docs/en/hooks.md). If the reference adds events or
changes these tables, update this file; fetch_reference.py --section
"Exit code 2 behavior per event" and --section "Prompt-based hooks" are
the sources.
"""
import re

EVENTS = [
    "SessionStart", "Setup", "UserPromptSubmit", "UserPromptExpansion", "PreToolUse",
    "PermissionRequest", "PermissionDenied", "PostToolUse", "PostToolUseFailure",
    "PostToolBatch", "Notification", "MessageDisplay", "SubagentStart", "SubagentStop",
    "TaskCreated", "TaskCompleted", "Stop", "StopFailure", "TeammateIdle",
    "InstructionsLoaded", "ConfigChange", "CwdChanged", "DirectoryAdded", "FileChanged",
    "WorktreeCreate", "WorktreeRemove", "PreCompact", "PostCompact", "PreModelSwitch",
    "PostModelSwitch", "Elicitation", "ElicitationResult", "SessionEnd",
]

TOOL_EVENTS = {"PreToolUse", "PostToolUse", "PostToolUseFailure", "PermissionRequest", "PermissionDenied"}

NO_MATCHER = {
    "UserPromptSubmit", "PostToolBatch", "Stop", "TeammateIdle", "TaskCreated", "TaskCompleted",
    "WorktreeCreate", "WorktreeRemove", "MessageDisplay", "CwdChanged",
}

# input field the matcher is tested against, for events with matcher support
MATCHER_FIELD = {e: "tool_name" for e in TOOL_EVENTS}
MATCHER_FIELD.update({
    "SessionStart": "source", "Setup": "trigger", "SessionEnd": "reason",
    "Notification": "notification_type", "SubagentStart": "agent_type", "SubagentStop": "agent_type",
    "PreCompact": "trigger", "PostCompact": "trigger", "ConfigChange": "source",
    "StopFailure": "error", "InstructionsLoaded": "load_reason", "FileChanged": "file_path",
    "Elicitation": "mcp_server_name", "ElicitationResult": "mcp_server_name",
    "UserPromptExpansion": "command_name", "DirectoryAdded": "source",
    "PreModelSwitch": "to_model", "PostModelSwitch": "to_model",
})

ALL_TYPES = {"command", "http", "mcp_tool", "prompt", "agent"}
ALL_FIVE = {
    "PermissionDenied", "PostToolBatch", "PostToolUse", "PostToolUseFailure", "PreToolUse", "Stop",
    "SubagentStop", "TaskCompleted", "TaskCreated", "TeammateIdle", "UserPromptExpansion", "UserPromptSubmit",
}


def allowed_types(event):
    if event in ALL_FIVE:
        return ALL_TYPES
    if event == "PermissionRequest":
        return {"command", "http", "mcp_tool", "prompt"}
    if event in {"SessionStart", "Setup"}:
        return {"command", "mcp_tool"}
    return {"command", "http", "mcp_tool"}


# What exit code 2 does, per event ("Exit code 2 behavior per event").
EXIT2_BLOCKS = {
    "PreToolUse", "UserPromptSubmit", "UserPromptExpansion", "Stop", "SubagentStop", "TeammateIdle",
    "TaskCreated", "TaskCompleted", "ConfigChange", "PostToolBatch", "PreCompact", "PreModelSwitch",
    "Elicitation", "ElicitationResult", "WorktreeCreate", "WorktreeRemove",
}
EXIT2_FEEDBACK_ONLY = {"PostToolUse", "PostToolUseFailure"}  # stderr shown to Claude, nothing blocked
EXIT2_USER_ONLY = {"SubagentStart", "SessionStart", "SessionEnd", "CwdChanged", "FileChanged", "PostCompact", "PostModelSwitch"}
EXIT2_IGNORED = {"PermissionRequest", "StopFailure", "PermissionDenied", "Notification", "Setup",
                 "InstructionsLoaded", "MessageDisplay", "DirectoryAdded"}

# Plain-text stdout on exit 0 is added to Claude's context only on these.
PLAIN_STDOUT_TO_CONTEXT = {"UserPromptSubmit", "UserPromptExpansion", "SessionStart", "PostModelSwitch"}

# onFailure is ignored on these (and on async hooks).
ONFAILURE_NO_EFFECT = {"Stop", "SubagentStop", "TaskCompleted", "TeammateIdle"}

# Events whose decision uses top-level "decision": "block" + "reason".
TOP_LEVEL_DECISION = {
    "UserPromptSubmit", "UserPromptExpansion", "PostToolUse", "PostToolUseFailure", "PostToolBatch",
    "Stop", "SubagentStop", "ConfigChange", "PreCompact", "TaskCreated", "PreModelSwitch",
    "Elicitation", "ElicitationResult",
}

UNIVERSAL_KEYS = {"continue", "stopReason", "suppressOutput", "systemMessage", "terminalSequence"}
HSO_ONLY_KEYS = {"permissionDecision", "permissionDecisionReason", "updatedInput", "additionalContext",
                 "updatedToolOutput", "updatedMCPToolOutput", "watchPaths", "sessionTitle",
                 "initialUserMessage", "reloadSkills", "retry", "displayContent", "classifierContext"}


def default_timeout(event, handler_type):
    if event == "SessionEnd":
        return 1.5
    if handler_type == "prompt":
        return 30
    if handler_type == "agent":
        return 60
    if event in {"UserPromptSubmit", "PreModelSwitch", "PostModelSwitch"}:
        return 30
    if event == "MessageDisplay":
        return 10
    return 600


_EXACT_CHARS = re.compile(r"^[A-Za-z0-9_\-, |]*$")
_EXACT_CHARS_NARROW = re.compile(r"^[A-Za-z0-9_|]*$")


def matcher_kind(event, matcher):
    """Return 'all', 'exact', or 'regex' following the reference's Matcher patterns table."""
    if matcher in (None, "", "*"):
        return "all"
    pattern = _EXACT_CHARS_NARROW if event in {"FileChanged", "StopFailure"} else _EXACT_CHARS
    return "exact" if pattern.match(matcher) else "regex"


def exact_names(event, matcher):
    seps = r"\|" if event in {"FileChanged", "StopFailure"} else r"[|,]"
    return [name.strip() for name in re.split(seps, matcher) if name.strip()]


def matcher_matches(event, matcher, value):
    kind = matcher_kind(event, matcher)
    if kind == "all":
        return True
    if value is None:
        return False
    if kind == "exact":
        return value in exact_names(event, matcher)
    try:
        return re.search(matcher, value) is not None  # unanchored, like JS RegExp.test
    except re.error:
        return False


# Tool names the reference mentions as hook targets. Not exhaustive: the tools
# reference lists more, so an unknown name is only a spelling hint.
KNOWN_TOOLS = {"Bash", "PowerShell", "Edit", "Write", "Read", "Glob", "Grep", "Agent", "Workflow",
               "WebFetch", "WebSearch", "AskUserQuestion", "ExitPlanMode", "NotebookEdit", "Skill",
               "SubagentHandback", "EnterWorktree", "ExitWorktree"}


def iter_handlers(config):
    """Yield (event, group_index, group, handler_index, handler) from a settings-style dict."""
    hooks = config.get("hooks") if isinstance(config, dict) else None
    if not isinstance(hooks, dict):
        return
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            yield event, None, groups, None, None
            continue
        for gi, group in enumerate(groups):
            handlers = group.get("hooks") if isinstance(group, dict) else None
            if not isinstance(handlers, list):
                yield event, gi, group, None, None
                continue
            for hi, handler in enumerate(handlers):
                yield event, gi, group, hi, handler
