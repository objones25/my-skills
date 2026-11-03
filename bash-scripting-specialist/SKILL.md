---
name: bash-scripting-specialist
description: Write, review, fix, test, and port bash scripts to production standards (strict error handling that actually catches failures, safe quoting, cleanup traps, argument parsing, portability between macOS bash 3.2 and Linux bash 5.x, GNU vs BSD tools, security). Ships a tested script template and bats suite, a ShellCheck config, an audit script that catches what ShellCheck misses (bash version requirements, GNU-only flags, set -e and pipefail traps), and a multi-version test runner. Use whenever the user asks to write, review, debug, harden, speed up, or explain a shell script or one-liner, mentions bash, sh, .sh files, set -euo pipefail, ShellCheck, shfmt, bats, cron jobs, or "automate this in the terminal", or a script works on Linux but not macOS (or the reverse). For Dockerfiles use dockerfile-expert.
---

# Bash scripting specialist

Goal: every script fails loudly instead of silently, never runs a destructive command on an
empty or wrong path, handles any filename, cleans up after itself on errors and Ctrl-C,
runs on the oldest bash it targets, and has been checked by running tools, not by
reading alone.

## Workflow

1. **Settle the target before writing.** Find out, or decide and state:
   - where it runs: macOS (stock `/bin/bash` is 3.2.57), which Linux, containers (Alpine
     has no bash by default), cron/CI (minimal PATH, no terminal);
   - the oldest bash version it must support (`references/portability.md` has the table);
   - whether it needs to be bash at all. Past ~100 lines of real logic, or when it needs
     JSON, nested data, floating point, or unit-testable internals, say that Python fits
     better (Google Shell Style Guide's threshold) and write bash only if the user still
     wants it.
   Done when the target version and platforms are written in the script header and in a
   `# bash-target: X.Y` line.
2. **Start from `assets/template.sh`.** It has the strict-mode header, logging to stderr,
   usage and exit codes, long and short option parsing, `--dry-run`, a cleanup trap, a
   temp dir, and the source guard, and it runs on bash 3.2 through 5.3. Delete what the
   script does not need; keep the structure. Look up patterns (parallel jobs, locking,
   retries, reading files safely) in `references/patterns.md`.
3. **Write the logic** following the rules below. When unsure about a construct, check
   `references/pitfalls.md` (wrong → right → why tables) and, for anything that handles
   outside input, secrets, or privileges, `references/security.md`.
4. **Check.** Run and show the real output:
   ```
   scripts/check.sh --target 3.2 --platform macos script.sh
   ```
   It runs `bash -n`, ShellCheck with `assets/shellcheckrc`, `shfmt -d`, and
   `scripts/audit_bash.py`, and reports any tool that is missing instead of skipping it
   silently. Done when every finding is fixed or suppressed with a written reason
   (`# shellcheck disable=SCxxxx # reason`, `# audit-bash disable=Bxxx # reason`). If a
   tool is unavailable, say which checks did not run.
5. **Test by running it.** For anything beyond a one-off: write a bats suite starting from
   `assets/template.bats` (help, a usage error, the main path, one failure path, a
   filename with spaces), and run it on the oldest target bash: `/bin/bash` on a Mac,
   `PATH=/bin:$PATH bats test/`, or `scripts/test_matrix.sh -- script.sh args`.
   Done when the tests pass on the oldest and newest target versions, or the reply says
   which versions were not tested.
6. **Deliver** the script (executable), its tests, and a short note: target version and
   platforms, required commands, exit codes, what was verified and on which versions,
   and anything not verified. Then explain, in a line each, the non-obvious constructs the
   script uses (for example `${arr[@]+"${arr[@]}"}`, `< <(...)`, `${var:?}`), so the user
   can maintain it without asking again.

When reviewing an existing script, run step 4, then walk `references/review-checklist.md`
and report findings by impact: destructive operations, then failure handling, then
security, then portability, then style. Give each one a concrete failing input and an
exact fix.

## Rules and why they matter

**Failure handling** (details and tested exemptions: `references/errors-and-traps.md`)
- Header: `set -Eeuo pipefail` plus `shopt -s inherit_errexit 2>/dev/null || true`. This
  is a backstop, not the strategy: `set -e` is silently off inside any function called
  from `if`, `&&`, `||` or `!`, and its value is masked by `local x=$(cmd)` and by
  `cmd "$(failing)"`. Check every command whose failure matters explicitly
  (`cmd || die "..."`, `if ! cmd; then`).
- Predicate functions (called from `if`) handle their own errors with `|| return 1`.
- Increment with `i=$((i + 1))`, never a bare `((i++))`, and never end a function with
  `cond && cmd`: both return 1 in normal operation and kill the script under `set -e`.
- With pipefail, `cmd | head` and `cmd | grep -q` can fail with 141 depending on timing;
  use a reader that consumes all input (`sed -n 1p`, `awk 'NR==1'`) or no pipe.
- `wait` with no arguments always returns 0; wait on each saved PID.
- One `cleanup` function on `trap cleanup EXIT`, failure-tolerant (`|| true` per step).
  Run long children as `cmd & child=$!; wait "$child"` and kill them in cleanup, or a
  `kill` of the script leaves them running.
- Exit 0 success, 1 failure, 2 usage error; messages to stderr, data to stdout.

**Quoting and data**
- Quote every expansion: `"$var"`, `"$(cmd)"`, `"${arr[@]}"`. Commands with variable parts
  live in arrays, never in strings; no `eval` on anything containing outside data.
- Read lines with `while IFS= read -r line || [[ -n $line ]]`; filenames with
  `find ... -print0` and `read -r -d ''`; loop with `< <(cmd)`, not `cmd | while`.
- `--` before user-supplied paths; `rm -rf -- "${dir:?}/..."` so an empty variable aborts.
- Validate numbers from outside with a pattern before any arithmetic or `-eq`: bash
  evaluates `a[$(cmd)]` inside arithmetic, so unvalidated input runs commands.
- `printf '%s\n' "$x"`, not `echo`, for anything that is not a fixed string.

**Portability** (`references/portability.md`)
- Shebang `#!/usr/bin/env bash`. `#!/bin/bash` pins macOS to 3.2 even when Homebrew bash
  is installed, and the official Docker bash images have no `/bin/bash`.
- Use only features at or below the target version; above 3.2, add the version guard from
  portability.md so old bash fails with a clear message instead of halfway through.
- Targeting 3.2 with `set -u`: expand possibly-empty arrays as `${arr[@]+"${arr[@]}"}`,
  and write `"$@"`, never `"${@}"` (the braced form aborts there when there are no
  arguments).
- No GNU-only flags in scripts that can run on macOS: `sed -i` (use `sed -i.bak`),
  `date -d`, `stat -c`, `grep -P`, `find -printf`, `timeout`, `flock`, `getopt --long`.
- Shell snippets meant to be pasted into a terminal contain no comment lines: macOS's
  default shell is zsh, where a pasted `# ...` line fails with "command not found: #".

**Structure** (Google Shell Style Guide)
- Constants and `set` options at the top, then functions, then `main`, then
  `if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi` so tests can source it.
- `local` for every function variable; `readonly` for constants; lower_snake_case names.
- `[[ ]]` over `[ ]`, `$(...)` over backticks, `(( ))` over `let`/`expr`.
- Format with `shfmt -i 2 -ci -bn` unless the repo has its own `.editorconfig`.

## Gotchas confirmed by testing

Tested October 2026 on bash 3.2.57, 4.3.30, 4.4.18, 5.2.21 and 5.3.20 (built from GNU
sources), ShellCheck 0.11.0, shfmt 3.14.1, bats 1.13.0 and 1.14.0, zsh 5.9.

- `set -e` without `-E`: an ERR trap prints nothing for a failure inside a function; the
  script just exits.
- `set -u` with an empty array: "unbound variable" on 3.2 and 4.3, fine from 4.4.
- `((i++))` from 0 under `set -e` exits on 4.3+ but not on 3.2, because `((` and `[[`
  only became subject to `set -e` in 4.1. Scripts tested only on a Mac can die on Linux.
- Bash 3.2 interrupted by Ctrl-C while in `wait` runs the EXIT trap but exits 129 as a
  normal exit; add the INT/TERM re-raise traps from errors-and-traps.md for 3.2 targets.
- `seq 1 100000 | head -n1` under pipefail returns 141 on every version.
- SIGTERM to a script running a foreground `sleep`: the EXIT trap runs and the script
  exits 143, but the `sleep` keeps running. Ctrl-C (SIGINT to the process group) kills
  both. The `& wait` pattern leaves nothing behind in both cases.
- bash 5.2 turned on `patsub_replacement`: `r='X&Y'; echo "${x/=/$r}"` prints `aX=Yb` on
  5.2+ and `aX&Yb` before. Quote the replacement (`"$r"`, 4.3+) or turn the option off.
- A comment starting with the word `shellcheck` is parsed as a directive and errors.
- Sourcing a script into a bats test imports its `set -e` and traps, and a script
  function named `run` silently replaces bats' `run`. Source in a child `bash -c`.

## Files

- `assets/template.sh`: production skeleton (bash 3.2+); `assets/template.bats`: its
  10-test suite, a model for new suites; `assets/shellcheckrc`: copy to the repo root as
  `.shellcheckrc`.
- `scripts/check.sh FILE... [--target X.Y] [--platform any|linux|macos] [--fix]`: all
  static checks with a summary; `--fix` applies shfmt.
- `scripts/audit_bash.py`: dependency-free checks ShellCheck lacks. `--rules` lists them
  (B1xx errexit/pipefail/wait/trap, B2xx version and GNU/BSD, B3xx security, B4xx
  structure), `--json` for CI, `--strict` fails on warnings. Heuristic and line-based:
  treat findings as review prompts. Tests: `python3 scripts/tests/test_audit_bash.py`.
- `scripts/test_matrix.sh [--versions "3.2 4.4 5.3"] [--bash PATH]... -- SCRIPT ARGS`:
  runs a command under several bash versions (Docker `bash:X.Y` images or local
  binaries) and flags differences in exit code or output.
- `references/errors-and-traps.md`: errexit exemptions, pipefail, nounset, ERR stack
  traces, signals, cleanup, temp files, exit codes, and the strict-mode debate.
- `references/portability.md`: choosing a target, feature-by-version table, macOS,
  GNU vs BSD commands, POSIX sh, testing on old bash.
- `references/pitfalls.md`: wrong → right → why tables with the ShellCheck code for each.
- `references/security.md`: eval, arithmetic injection, ssh quoting, secrets, temp-file
  races, privileged scripts, downloads.
- `references/patterns.md`: argument parsing, logging, reading files, script dir,
  parallel jobs, locking, retries, dry run, string operations.
- `references/tooling.md`: ShellCheck config and codes, shfmt, bats, debugging with PS4.
- `references/review-checklist.md`: ordered checklist for reviewing a script.

Versions and distro facts were current on 2026-10-22 (latest bash 5.3 with patches
through 5.3.20; ShellCheck 0.11.0; shfmt 3.14.1; bats-core 1.14.0). Check for newer
releases before stating them as current.
