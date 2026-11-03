#!/usr/bin/env python3
"""Audit bash scripts for problems ShellCheck does not report.

ShellCheck catches quoting, word splitting and most syntax pitfalls. This audit adds:
  B1xx  error-handling traps that were verified by running bash (errexit, ERR, pipefail, wait)
  B2xx  portability: features newer than the target bash version, GNU-only command flags
  B3xx  security: eval, curl|sh, secrets in argv, predictable temp files, unguarded rm -rf
  B4xx  structure: shebang, main(), script length

Usage:
  audit_bash.py [--target 4.4] [--platform any|linux|macos] [--json] [--strict] FILE...
  audit_bash.py --rules

Exit status: 0 no errors (and no warnings with --strict), 1 findings, 2 usage error.

Suppress a finding with a comment on the same line or the line above:
  # audit-bash disable=B301,B104
or for the whole file (anywhere in it):
  # audit-bash disable-file=B402
Set the target version inside a script with:  # bash-target: 3.2

Heuristic, line-based analysis: it reads code with comments, single-quoted strings and
heredoc bodies masked, so it can miss constructs split across lines and can flag code
that is correct for reasons it cannot see. Treat findings as review prompts, then fix
or suppress with a reason.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Rule catalogue
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str  # error | warning | info
    title: str
    fix: str


RULES = {
    r.id: r
    for r in [
        Rule("B101", "warning", "no errexit: failures are silently ignored",
             "Add `set -Eeuo pipefail` near the top (and still check important commands explicitly)."),
        Rule("B102", "warning", "ERR trap without errtrace (set -E): it never fires for failures inside functions",
             "Use `set -E` (or `set -o errtrace`) together with the ERR trap."),
        Rule("B103", "warning", "((x++)), ((x--)), ((--x)) or let as a statement under set -e: the expression evaluates to 0 when crossing zero, the command returns 1, and bash 4.1+ exits",
             "Write `x=$((x + 1))` (always status 0)."),
        Rule("B104", "warning", "pipefail with a reader that exits early (head, grep -q, grep -m): the writer gets SIGPIPE and the pipeline fails with 141",
             "Let the reader consume all input (`awk 'NR==1'`, `sed -n 1p`), use `< <(cmd)` or a here-string, or check PIPESTATUS and accept 141 from the writer."),
        Rule("B105", "warning", "bare `wait` after background jobs always returns 0, so job failures are lost",
             "Save each `$!` in an array and `wait \"$pid\" || failed=1` for each one."),
        Rule("B106", "warning", "function ends with `cond && cmd`: it returns 1 when cond is false, which kills the caller under set -e",
             "Use `if cond; then cmd; fi` as the last statement, or end the function with `return 0`."),
        Rule("B107", "warning", "mktemp without an EXIT trap: temp files are left behind on errors and Ctrl-C",
             "Set `trap cleanup EXIT` before creating temp files and remove them in cleanup()."),
        Rule("B108", "info", "errexit is on but command substitutions do not inherit it",
             "Add `shopt -s inherit_errexit` (bash 4.4+; use `2>/dev/null || true` if the script must also run on older bash)."),
        Rule("B109", "warning", "`local`/`declare`/`export`/`readonly` with a command substitution hides the command's exit status",
             "Declare first, assign on the next line: `local out; out=$(cmd)`."),
        Rule("B201", "error", "feature needs a newer bash than the target",
             "Rewrite with an older equivalent, raise --target, or add a version guard that exits with a clear message."),
        Rule("B202", "warning", "with set -u, \"${arr[@]}\" on an empty array is an 'unbound variable' error before bash 4.4 (stock macOS bash is 3.2)",
             "Use ${arr[@]+\"${arr[@]}\"} or require bash 4.4+."),
        Rule("B203", "warning", "#!/bin/bash runs the system bash; on macOS that is 3.2 even when a newer bash is installed",
             "Use `#!/usr/bin/env bash` so the first bash on PATH (e.g. Homebrew's) runs it, and keep a version guard."),
        Rule("B204", "warning", "GNU-only command or flag: fails or behaves differently on macOS/BSD",
             "See references/portability.md for the portable form, or set --platform linux if the script never runs on macOS."),
        Rule("B205", "warning", "in ${var/pat/rep}, bash 5.2+ treats & in rep as the matched text (patsub_replacement is on by default)",
             "Quote the replacement: ${var/pat/\"$rep\"} (bash 4.3+), escape a literal \\&, or `shopt -u patsub_replacement 2>/dev/null || true` for 3.2-5.x compatibility."),
        Rule("B301", "warning", "eval runs data as code",
             "Use arrays for commands, `printf -v` for indirect assignment, or `declare -n` after validating the name."),
        Rule("B302", "error", "piping a download straight into a shell",
             "Download to a file, verify a checksum or signature, inspect, then run."),
        Rule("B303", "warning", "secret passed on the command line is visible to other users via ps and /proc",
             "Pass it via stdin, an environment variable, or a 0600 file (`--password-file`, `--password-stdin`)."),
        Rule("B304", "warning", "predictable path in /tmp can be pre-created or symlinked by another user",
             "Use `mktemp` / `mktemp -d` and remove it in an EXIT trap."),
        Rule("B305", "warning", "world-writable permissions",
             "Grant the narrowest mode that works (e.g. 755 for dirs, 644 for files, 600 for secrets)."),
        Rule("B306", "warning", "rm -rf on a path built from a variable with no empty-value guard",
             "Use \"${dir:?}\" so an empty or unset variable aborts instead of expanding to / or the cwd."),
        Rule("B307", "info", "untrusted-looking value used in arithmetic: $((...)), ((...)) and -eq evaluate array subscripts, so 'a[$(cmd)]' runs cmd",
             "Validate first with a pattern that never evaluates the input: `[[ -n $n && $n != *[!0123456789]* ]]`, then use 10#$n."),
        Rule("B308", "warning", "sourcing a relative path depends on the caller's working directory",
             "Resolve the script directory first and source \"$script_dir/lib.sh\"."),
        Rule("B309", "info", "global IFS change affects every later unquoted expansion and read",
             "Scope it: `IFS=, read -r -a parts <<<\"$line\"` or `local IFS=...` inside a function."),
        Rule("B401", "error", "missing shebang",
             "Start the file with `#!/usr/bin/env bash`."),
        Rule("B402", "info", "long shell script: Google's style guide says rewrite in a structured language past ~100 lines or complex control flow",
             "Consider Python for data structures, parsing, or heavy logic; keep bash for gluing commands."),
        Rule("B403", "info", "script defines functions but has no main(): top-level code between functions is hard to follow and to test",
             "Put top-level logic in main() and end with a guarded `main \"$@\"`."),
        Rule("B404", "info", "echo with options or escapes is not portable and misprints data starting with -",
             "Use printf: `printf '%s\\n' \"$x\"`."),
    ]
}

# Bash features by version, from the bash NEWS file and BashFAQ/061, confirmed on
# bash 3.2.57, 4.3.30, 4.4.18, 5.2.21 and 5.3.20 where those versions bracket the change.
# (regex, minimum version, description). Regexes run on masked code (no comments,
# single-quoted text blanked) unless the description starts with "raw:".
FEATURES: list[tuple[str, tuple[int, int], str]] = [
    (r"\b(?:declare|local|typeset|readonly)\s+(?:-\w*\s+)*-\w*A", (4, 0), "associative arrays (declare -A)"),
    (r"\b(?:declare|local|typeset)\s+(?:-\w*\s+)*-\w*[lu]\b", (4, 0), "declare -l/-u case attributes"),
    (r"\b(?:mapfile|readarray)\b", (4, 0), "mapfile/readarray"),
    (r"\b(?:mapfile|readarray)\b[^;|&]*\s-d\b", (4, 4), "mapfile -d"),
    (r"\$\{[A-Za-z_][A-Za-z0-9_]*(?:\[[^]]*\])?(?:,,?|\^\^?)[^}]*\}", (4, 0), "${var,,} / ${var^^} case conversion"),
    (r"(?<![;&]);&\s*$|;;&", (4, 0), "case fall-through ;& or ;;&"),
    (r"(?:^|[;&|(]\s*)coproc\b", (4, 0), "coproc"),
    (r"(?<!\|)\|&", (4, 0), "|& (pipe stderr too)"),
    (r"&>>", (4, 0), "&>> (append stdout and stderr)"),
    (r"\bshopt\s+-s\s+(?:\w+\s+)*globstar\b", (4, 0), "globstar (**)"),
    (r"\bBASHPID\b", (4, 0), "$BASHPID"),
    (r"\bread\b[^;|&]*\s-t\s*[0-9]*\.[0-9]", (4, 0), "read -t with a fractional timeout"),
    (r"\{[0-9]+\.\.[0-9]+\.\.[0-9]+\}|\{0[0-9]+\.\.[0-9]+\}", (4, 0), "brace ranges with step or zero padding"),
    (r"\{[A-Za-z_][A-Za-z0-9_]*\}[<>]", (4, 1), "{fd}> automatic file descriptor allocation"),
    (r"\bBASH_XTRACEFD\b", (4, 1), "BASH_XTRACEFD"),
    (r"\bread\b[^;|&]*\s-N\b", (4, 1), "read -N"),
    (r"(?:\[\[|\[|\btest)\s+-v\s", (4, 2), "test -v (is variable set)"),
    (r"raw:printf\b.*%(?:-?[0-9]*)?\([^)]*\)T", (4, 2), "printf %(fmt)T timestamps"),
    (r"\$\{[A-Za-z_][A-Za-z0-9_]*\[\s*-[0-9]+\s*\]\}", (4, 2), "negative array index ${a[-1]}"),
    (r"\b[A-Za-z_][A-Za-z0-9_]*\[-[0-9]+\]=|\bunset\s+(?:-v\s+)?['\"]?[A-Za-z_][A-Za-z0-9_]*\[-[0-9]+\]", (4, 3), "negative index in assignment/unset a[-1]=x"),
    (r"\$\{[A-Za-z_][A-Za-z0-9_]*:[^}:]*:\s*-[0-9]", (4, 2), "${var:start:-len} negative length"),
    (r"\bshopt\s+-s\s+(?:\w+\s+)*lastpipe\b", (4, 2), "lastpipe"),
    (r"\b(?:declare|typeset)\s+(?:-\w*\s+)*-\w*g", (4, 2), "declare -g"),
    (r"\b(?:declare|local|typeset)\s+(?:-\w*\s+)*-\w*n\b", (4, 3), "namerefs (declare -n)"),
    (r"\bwait\s+(?:-\w+\s+)*-\w*n", (4, 3), "wait -n"),
    (r"\$\{[^}]*@[QEPAa]\}", (4, 4), "${var@Q} style transformations"),
    (r"\bshopt\s+-s\s+(?:\w+\s+)*inherit_errexit\b", (4, 4), "inherit_errexit"),
    (r"\blocal\s+-(?:\s|;|$)", (4, 4), "local - (scoped set options)"),
    (r"\b(?:EPOCHSECONDS|EPOCHREALTIME|BASH_ARGV0)\b", (5, 0), "EPOCHSECONDS/EPOCHREALTIME/BASH_ARGV0"),
    (r"\bSRANDOM\b", (5, 1), "SRANDOM"),
    (r"\bwait\s+(?:-\w+\s+)*-\w*p\s", (5, 1), "wait -p"),
    (r"\$\{[^}]*@[ULuK]\}", (5, 1), "${var@U} ${var@L} ${var@u} ${var@K}"),
    (r"\$\{[^}]*@k\}", (5, 2), "${var@k}"),
    (r"\bshopt\s+-s\s+(?:\w+\s+)*patsub_replacement\b", (5, 2), "patsub_replacement"),
    (r"\$\{[ \t]|\$\{\|", (5, 3), "${ cmd; } / ${| cmd; } command substitution without a fork"),
    (r"\bGLOBSORT\b", (5, 3), "GLOBSORT"),
    (r"\bsource\s+-p\b", (5, 3), "source -p"),
    (r"\bshopt\s+-s\s+(?:\w+\s+)*array_expand_once\b", (5, 3), "array_expand_once"),
]

# GNU-only commands/flags. Each entry: (regex on masked code, message).
GNU_ONLY: list[tuple[str, str]] = [
    (r"\bsed\s+(?:-[a-zA-Z]*\s+)*-[a-zA-Z]*i(?:\s+-e|\s+['\"]|\s+s[/|#,]|\s*$)",
     "sed -i with no backup suffix: BSD sed reads the next word as the suffix. Portable: write to a temp file and mv, or `sed -i.bak ... && rm file.bak`."),
    (r"\bsed\s+[^|;]*--in-place\b", "sed --in-place is GNU-only."),
    (r"\bdate\b[^|;]*\s(?:-d\b|--date\b)", "date -d is GNU-only (BSD: date -j -f FMT STR). Prefer printf '%(...)T' (bash 4.2+) for now, or EPOCHSECONDS (5.0+)."),
    (r"\bstat\s+(?:-\w+\s+)*-c\b|\bstat\s+--format", "stat -c is GNU-only (BSD: stat -f with different format letters)."),
    (r"\bgrep\s+(?:-\w+\s+)*-\w*P", "grep -P (PCRE) is GNU-only; use grep -E."),
    (r"\bxargs\b[^|;]*--no-run-if-empty", "xargs --no-run-if-empty is GNU-only (macOS xargs accepts the short -r and skips empty input by default)."),
    (r"\bfind\b[^|;]*\s-printf\b", "find -printf is GNU-only; use -exec printf or stat."),
    (r"(?:^|[;&|(]\s*|\bthen\s+|\bdo\s+)timeout\s+", "timeout is not installed on macOS by default (Homebrew coreutils provides gtimeout)."),
    (r"(?:^|[;&|(]\s*|\bthen\s+|\bdo\s+)flock\s+", "flock is util-linux only; macOS needs `brew install flock` or a mkdir lock."),
    (r"\bgetopt\s+(?:-\w+\s+)*(?:-o|--options|-l|--long)\b", "enhanced getopt (-o/--long) is util-linux only; macOS getopt is the old BSD one. Parse with a while/case loop instead."),
    (r"\breadlink\s+(?:-\w+\s+)*-\w*[em]\b", "readlink -e/-m are GNU-only (macOS 12.3+ has only -f and -n); use `cd -P -- dir && pwd -P` for directories."),
    (r"\brealpath\s+(?:-\w+\s+)*-(?:[^q\s-]|-relative)", "realpath options other than -q are GNU-only (macOS realpath, on recent releases only, has just -q)."),
    (r"\bsed\s+(?:-[a-zA-Z]*\s+)*-i\s+(?:''|\"\")", "sed -i '' is the BSD form; GNU sed reads '' as the script. Portable: `sed -i.bak ... && rm file.bak`, or a temp file and mv."),
    (r"\b(?:cp|mv|ln|rm|mkdir)\s+[^|;]*--(?:parents|no-target-directory|target-directory|relative|verbose)\b", "long GNU coreutils options are not accepted by BSD tools."),
    (r"\bmktemp\s+(?:-\w+\s+)*-\w*t\b", "mktemp -t means different things in GNU (deprecated) and BSD; use `mktemp` or `mktemp -d \"${TMPDIR:-/tmp}/name.XXXXXX\"`."),
    (r"\bbase64\s+(?:-\w+\s+)*-w\b", "base64 -w is GNU-only."),
]

SECRET_FLAG = re.compile(
    r"(?:--(?:password|passwd|pass|token|secret|api[-_]?key|access[-_]?key|client[-_]secret)(?:=|\s+)[\"']?\$"
    r"|\bsshpass\s+-p\s*[\"']?\$"
    r"|\bmysql(?:dump|admin)?\b[^|;]*\s-p[\"']?\$"
    r"|\bcurl\b[^|;]*\s(?:-u|--user)\s+[\"']?[^\s\"']*:\$"
    r"|\bcurl\b[^|;]*-H\s+[\"']?Authorization:[^\"']*\$"
    r"|\bdocker\s+login\b[^|;]*\s(?:-p|--password)(?:=|\s+)[\"']?\$"
    r"|://[^/\s:@\"']*:\$\{?[A-Za-z_][A-Za-z0-9_]*\}?@)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------------------
# Lexing: mask comments, single-quoted text, and heredoc bodies; join continuations
# --------------------------------------------------------------------------------------

HEREDOC_RE = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")


@dataclass
class Line:
    no: int  # 1-based number of the first physical line
    raw: str  # original text (joined continuations)
    code: str  # masked text
    comment: str = ""  # trailing comment text, for directives


def mask_line(text: str, state: dict) -> tuple[str, str]:
    """Blank single-quoted content and strip comments. Tracks open quotes across lines."""
    out = []
    comment = ""
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if state["sq"]:
            if state.get("ansi") and c == "\\" and i + 1 < n:
                out.append("  ")  # escaped char inside $'...', e.g. \'
                i += 2
                continue
            if c == "'":
                state["sq"] = False
                state["ansi"] = False
                out.append("'")
            else:
                out.append(" ")
            i += 1
            continue
        if state["dq"]:
            if c == "\\" and i + 1 < n:
                out.append(text[i : i + 2])
                i += 2
                continue
            if c == '"':
                state["dq"] = False
            out.append(c)
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            out.append(text[i : i + 2])
            i += 2
            continue
        if c == "'":
            # '...' and $'...' are both masked; only $'...' interprets backslashes
            state["sq"] = True
            state["ansi"] = i > 0 and text[i - 1] == "$"
            out.append("'")
            i += 1
            continue
        if c == '"':
            state["dq"] = True
            out.append(c)
            i += 1
            continue
        if c == "#" and (i == 0 or text[i - 1] in " \t;&|()"):
            # ${#var} and $# are preceded by { or $, so they never reach here
            comment = text[i:]
            break
        out.append(c)
        i += 1
    return "".join(out), comment


def lex(src: str) -> list[Line]:
    physical = src.splitlines()
    lines: list[Line] = []
    state = {"sq": False, "dq": False, "ansi": False}
    pending_heredocs: list[tuple[str, bool]] = []
    i = 0
    while i < len(physical):
        if pending_heredocs:
            term, strip_tabs = pending_heredocs[0]
            body_line = physical[i].lstrip("\t") if strip_tabs else physical[i]
            if body_line == term:
                pending_heredocs.pop(0)
            i += 1
            continue
        start = i
        raw = physical[i]
        # join backslash continuations (not inside single quotes)
        while raw.endswith("\\") and not raw.endswith("\\\\") and i + 1 < len(physical) and not state["sq"]:
            i += 1
            raw = raw[:-1] + " " + physical[i].lstrip()
        code, comment = mask_line(raw, state)
        for m in HEREDOC_RE.finditer(raw[: len(code)]):
            pending_heredocs.append((m.group(3), m.group(1) == "-"))
        lines.append(Line(no=start + 1, raw=raw, code=code, comment=comment))
        i += 1
    return lines


# --------------------------------------------------------------------------------------
# Findings and suppression
# --------------------------------------------------------------------------------------


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    message: str
    severity: str = field(init=False)

    def __post_init__(self) -> None:
        self.severity = RULES[self.rule].severity


DIRECTIVE_RE = re.compile(r"audit-bash\s+disable=([A-Z0-9,\s]+)")
FILE_DIRECTIVE_RE = re.compile(r"audit-bash\s+disable-file=([A-Z0-9,\s]+)")
TARGET_RE = re.compile(r"bash-target:\s*([0-9]+)\.([0-9]+)")


def parse_ids(s: str) -> set[str]:
    return {x.strip() for x in s.split(",") if x.strip()}


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def version_str(v: tuple[int, int]) -> str:
    return f"{v[0]}.{v[1]}"


def audit(path: str, src: str, target: tuple[int, int], platform: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = lex(src)
    physical = src.splitlines()

    def add(line_no: int, rule: str, msg: str = "") -> None:
        findings.append(Finding(path, line_no, rule, msg or RULES[rule].title))

    # in-file target override
    for ln in physical[:30]:
        m = TARGET_RE.search(ln)
        if m:
            target = (int(m.group(1)), int(m.group(2)))
            break

    shebang = physical[0] if physical else ""
    if not shebang.startswith("#!"):
        add(1, "B401")
    elif not re.search(r"\bbash\b", shebang):
        # Not a bash script; only the shebang check applies.
        return findings

    code_lines = [ln for ln in lines if ln.code.strip()]
    all_code = "\n".join(ln.code for ln in lines)
    raw_code = "\n".join(ln.raw[: len(ln.code)] for ln in lines)  # quotes kept, comments dropped

    errexit = bool(re.search(r"\bset\s+-[a-zA-Z]*e|\bset\s+(?:-o\s+\w+\s+)*-o\s+errexit\b", all_code))
    errtrace = bool(re.search(r"\bset\s+-[a-zA-Z]*E|\bset\s+(?:-o\s+\w+\s+)*-o\s+errtrace\b", all_code))
    nounset = bool(re.search(r"\bset\s+-[a-zA-Z]*u|\bset\s+(?:-o\s+\w+\s+)*-o\s+nounset\b", all_code))
    pipefail = bool(re.search(r"\bset\s+(?:-\w+\s+)*-o\s+pipefail\b|\bset\s+-[a-zA-Z]*o\s+pipefail\b", all_code))
    inherit = bool(re.search(r"\binherit_errexit\b", all_code))
    has_exit_trap = bool(re.search(r"\btrap\b[^\n]*\b(?:EXIT|0)\b", all_code))

    # ---- B1xx error handling ----
    if not errexit:
        add(code_lines[0].no if code_lines else 1, "B101")
    if errexit and not inherit and target >= (4, 4):
        m = re.search(r"\bset\s+-[a-zA-Z]*e|\bset\s+(?:-o\s+\w+\s+)*-o\s+errexit\b", all_code)
        line_no = all_code[: m.start()].count("\n") + 1 if m else 1
        add(lines[line_no - 1].no if line_no - 1 < len(lines) else 1, "B108")

    bg_jobs = False
    for ln in lines:
        c = ln.code.strip()  # strip indentation so ^-anchored rules see code in functions
        if re.search(r"\btrap\b.*\bERR\b", c) and not errtrace:
            add(ln.no, "B102")
        # x++, x-- and --x evaluate to 0 when they cross zero; ++x only when x was -1.
        if errexit and re.search(r"(?:^|[;&|{]\s*|\bthen\s+|\bdo\s+|\belse\s+)(?:\(\(\s*(?:[A-Za-z_][A-Za-z0-9_]*(?:\[[^]]*\])?\s*(?:\+\+|--)|--\s*[A-Za-z_][A-Za-z0-9_]*)\s*\)\)|let\s+[\"']?(?:[A-Za-z_][A-Za-z0-9_]*(?:\+\+|--)|--[A-Za-z_]))\s*(?:;|$|\bfi\b|\bdone\b|\})", c):
            add(ln.no, "B103")
        if pipefail and re.search(r"\|\s*(?:head\b|grep\s+(?:-\w+\s+)*-\w*[qm])", c):
            add(ln.no, "B104")
        if re.search(r"(?<![&>|])&\s*(?:$|;|\bdone\b)", c) or re.search(r"(?<![&>|])&\s+\w+=\$!", c):
            bg_jobs = True
        if bg_jobs and re.search(r"(?:^|[;&|]\s*|\bthen\s+|\bdo\s+)wait\s*(?:;|$|\|\||&&)", c):
            add(ln.no, "B105")
        if re.search(r"\b(?:local|declare|typeset|export|readonly)\s+(?:-\w+\s+)*[A-Za-z_][A-Za-z0-9_]*=\"?\$\(", c):
            add(ln.no, "B109")
        if re.search(r"\bmktemp\b", c) and not has_exit_trap:
            add(ln.no, "B107")

    # B106: last statement of a function body is `cond && cmd`
    def and_list_outside_tests(code: str) -> bool:
        """True if code has a top-level `&&` (not inside [[ ]], (( )) or double quotes)."""
        stripped = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
        stripped = re.sub(r"\[\[.*?\]\]", "[[ ]]", stripped)
        stripped = re.sub(r"\(\(.*?\)\)", "(( ))", stripped)
        return "&&" in stripped and not re.search(r"\|\||\breturn\b|\bexit\b|\btrue\b", stripped)

    if errexit:
        i = 0
        while i < len(lines):
            code_i = lines[i].code
            one_line = re.match(r"\s*(?:function\s+)?[A-Za-z_][A-Za-z0-9_:.-]*\s*\(\s*\)\s*\{(.*);\s*\}\s*$", code_i)
            if one_line:
                last_stmt = one_line.group(1).split(";")[-1]
                if and_list_outside_tests(last_stmt):
                    add(lines[i].no, "B106")
                i += 1
                continue
            m = re.match(r"\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_:.-]*)\s*\(\s*\)\s*\{\s*$", code_i)
            if m:
                indent = len(code_i) - len(code_i.lstrip())
                j = i + 1
                last = None
                while j < len(lines):
                    cj = lines[j].code
                    if cj.strip() == "}" and len(cj) - len(cj.lstrip()) == indent:
                        break
                    if cj.strip():
                        last = lines[j]
                    j += 1
                if last is not None and and_list_outside_tests(last.code.split(";")[-1] or last.code):
                    add(last.no, "B106")
                i = j
            i += 1

    # ---- B2xx portability ----
    # Arrays that can be empty: initialized to (), declared -a without a value, or filled
    # by mapfile/readarray/read -a. Arrays only ever assigned non-empty literals are skipped.
    maybe_empty: set[str] = set()
    maybe_empty |= set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)=\(\s*\)", all_code))
    maybe_empty |= set(re.findall(r"\b(?:local|declare|typeset)\s+(?:-\w+\s+)*-\w*a\w*\s+([A-Za-z_][A-Za-z0-9_]*)(?![=\w])", all_code))
    maybe_empty |= set(re.findall(r"\b(?:mapfile|readarray)\b[^;|&<\n]*?\s([A-Za-z_][A-Za-z0-9_]*)\s*(?:<|$|;)", all_code, re.M))
    maybe_empty |= set(re.findall(r"\bread\b[^;|&\n]*-\w*a\s+([A-Za-z_][A-Za-z0-9_]*)", all_code))
    # copies of possibly-empty lists: args=("$@"), b=("${a[@]}")
    maybe_empty |= set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)=\(\s*\"\$(?:@|\{[A-Za-z_][A-Za-z0-9_]*\[@\]\})\"\s*\)", all_code))
    for ln in lines:
        for pattern, ver, desc in FEATURES:
            text = ln.raw[: len(ln.code)] if pattern.startswith("raw:") else ln.code
            pat = pattern[4:] if pattern.startswith("raw:") else pattern
            if ver > target and re.search(pat, text):
                # guarded optional enablement is fine: shopt -s x 2>/dev/null || true
                if "shopt" in pat and "||" in ln.code:
                    continue
                add(ln.no, "B201", f"{desc} needs bash {version_str(ver)}+, target is {version_str(target)}")
        if nounset and target < (4, 3) and re.search(r"\$\{[@*]\}", ln.code):
            add(ln.no, "B202", "\"${@}\" / \"${*}\" abort under set -u with no arguments on bash 3.2; write \"$@\" / \"$*\"")
        if nounset and target < (4, 4):
            for m in re.finditer(r"(?<!\+)\"\$\{([A-Za-z_][A-Za-z0-9_]*)\[@\]\}\"", ln.code):
                if m.group(1) not in maybe_empty:
                    continue
                prefix = ln.code[: m.start()]
                if prefix.endswith(f"${{{m.group(1)}[@]+"):
                    continue
                add(ln.no, "B202", f"\"${{{m.group(1)}[@]}}\" errors under set -u when empty on bash < 4.4")

    if not re.search(r"shopt\s+-u\s+(?:\w+\s+)*patsub_replacement", all_code):
        for ln in lines:
            for m in re.finditer(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?:\[[^]]*\])?//?[^/}]*/([^}]*)\}", ln.code):
                rep = m.group(1)
                if re.search(r"(?<!\\)&", rep) or re.match(r"\$\{?[A-Za-z_]", rep):
                    add(ln.no, "B205")

    newest_needed = max((v for p, v, _ in FEATURES
                         if re.search(p[4:] if p.startswith("raw:") else p, raw_code if p.startswith("raw:") else all_code)),
                        default=(3, 2))
    if re.match(r"#!\s*/bin/bash\b", shebang) and newest_needed > (3, 2) and platform != "linux":
        add(1, "B203", f"{RULES['B203'].title} (this script uses bash {version_str(newest_needed)} features)")

    if platform != "linux":
        for ln in lines:
            for pattern, msg in GNU_ONLY:
                if re.search(pattern, ln.code.strip()):
                    add(ln.no, "B204", msg)

    # ---- B3xx security ----
    for ln in lines:
        c = ln.code.strip()  # strip indentation so ^-anchored rules see code in functions
        if re.search(r"(?:^|[;&|(]\s*|\bthen\s+|\bdo\s+)eval\s", c):
            add(ln.no, "B301")
        if re.search(r"\b(?:curl|wget|fetch)\b[^|]*\|\s*(?:sudo\s+(?:-\w+\s+)*)?(?:ba|z|da|k)?sh\b", c):
            add(ln.no, "B302")
        if SECRET_FLAG.search(c):
            add(ln.no, "B303")
        if re.search(r"(?:>>?|=|\btee\s+(?:-a\s+)?)\s*[\"']?/tmp/[A-Za-z0-9_.${}-]+", c) and "mktemp" not in c and "XXX" not in c:
            add(ln.no, "B304")
        if re.search(r"\bchmod\s+(?:-\w+\s+)*(?:0?777|[0-7]?[0-7][0-7][2367]\b|a\+w|o\+w)", c):
            add(ln.no, "B305")
        for m in re.finditer(r"\brm\s+((?:-[A-Za-z-]+\s+)+)([^;&|]*)", c):
            if not re.search(r"-\w*[rR]|--recursive", m.group(1)):
                continue
            for tok in m.group(2).split():
                if tok == "--":
                    continue
                v = re.match(r"[\"']?\$\{?([A-Za-z_][A-Za-z0-9_]*)", tok)
                if v and ":?" not in tok:
                    add(ln.no, "B306", f"rm -r on ${v.group(1)} with no ${{{v.group(1)}:?}} guard")
        if re.search(r"(?:\$\(\(|\(\(|\[\[)[^)\]]*\$(?:[1-9]|\{[1-9]\}|OPTARG|\{OPTARG\}|REPLY)\b", c) and \
                re.search(r"\$\(\(|\(\(|-(?:eq|ne|lt|le|gt|ge)\b", c):
            add(ln.no, "B307")
        if re.search(r"(?:^|[;&|]\s*|\bthen\s+)(?:source|\.)\s+[\"']?(?!\$|/|~)[A-Za-z0-9_./-]+\.(?:sh|bash)\b", c):
            add(ln.no, "B308")
        if re.match(r"\s*IFS=(?:\$'[^']*'|'[^']*'|\"[^\"]*\"|[^\s;]*)\s*(?:;|$)", c):
            add(ln.no, "B309")
        if re.search(r"(?:^|[;&|]\s*|\bthen\s+|\bdo\s+)echo\s+-[neE]+\b", c):
            add(ln.no, "B404")

    # ---- B4xx structure ----
    funcs = [ln for ln in lines if re.match(r"\s*(?:function\s+)?[A-Za-z_][A-Za-z0-9_:.-]*\s*\(\s*\)\s*\{?", ln.code)]
    has_main = any(re.match(r"\s*(?:function\s+)?main\s*\(\s*\)", ln.code) for ln in lines)
    if len(funcs) >= 2 and not has_main:
        add(funcs[0].no, "B403")
    if len(code_lines) > 150:
        add(1, "B402", f"{RULES['B402'].title} (this one has {len(code_lines)} code lines)")

    # ---- suppression ----
    file_disabled: set[str] = set()
    for ln in physical:
        m = FILE_DIRECTIVE_RE.search(ln)
        if m:
            file_disabled |= parse_ids(m.group(1))

    def suppressed(f: Finding) -> bool:
        if f.rule in file_disabled:
            return True
        for idx in (f.line - 1, f.line - 2):
            if 0 <= idx < len(physical):
                m = DIRECTIVE_RE.search(physical[idx])
                if m and f.rule in parse_ids(m.group(1)):
                    return True
        return False

    findings = [f for f in findings if not suppressed(f)]
    # de-duplicate same rule on same line
    seen = set()
    unique = []
    for f in findings:
        key = (f.line, f.rule, f.message)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return sorted(unique, key=lambda f: (f.line, f.rule))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_target(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"([0-9]+)\.([0-9]+)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"expected MAJOR.MINOR, got {s!r}")
    return int(m.group(1)), int(m.group(2))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--target", type=parse_target, default=(4, 4),
                    help="oldest bash the script must run on (default 4.4; use 3.2 for stock macOS)")
    ap.add_argument("--platform", choices=["any", "linux", "macos"], default="any",
                    help="'linux' skips GNU-vs-BSD checks (default: any)")
    ap.add_argument("--json", action="store_true", help="print findings as JSON")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--rules", action="store_true", help="list rules and exit")
    args = ap.parse_args(argv)

    if args.rules:
        for r in RULES.values():
            print(f"{r.id}  {r.severity:<7}  {r.title}\n        fix: {r.fix}")
        return 0
    if not args.files:
        ap.error("no files given")

    all_findings: list[Finding] = []
    for path in args.files:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
        except OSError as e:
            print(f"audit_bash: {path}: {e.strerror}", file=sys.stderr)
            return 2
        all_findings += audit(path, src, args.target, args.platform)

    if args.json:
        print(json.dumps([f.__dict__ for f in all_findings], indent=2))
    else:
        for f in all_findings:
            print(f"{f.path}:{f.line}: {f.severity}: {f.rule} {f.message}")
            print(f"    fix: {RULES[f.rule].fix}")
        counts = {s: sum(f.severity == s for f in all_findings) for s in ("error", "warning", "info")}
        print(f"audit_bash: {counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} info", file=sys.stderr)

    if any(f.severity == "error" for f in all_findings):
        return 1
    if args.strict and any(f.severity == "warning" for f in all_findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
