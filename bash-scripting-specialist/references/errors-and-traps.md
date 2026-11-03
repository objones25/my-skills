# Error handling, traps, signals, cleanup

Read this when a script must fail safely: anything that deletes, deploys, writes shared
state, or runs unattended. Every behavior marked *tested* was run on bash 3.2.57, 4.3.30,
4.4.18, 5.2.21 and 5.3.20 (October 2026) unless a narrower set is named.

## The recommended header, and what it does not do

```bash
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true   # bash 4.4+; harmless no-op on older bash
```

- `-e` (errexit): exit when a command fails, with the exemptions below.
- `-E` (errtrace): the ERR trap also fires inside functions, subshells and `$(...)`.
- `-u` (nounset): expanding an unset variable is an error.
- `-o pipefail`: a pipeline fails if any stage fails, not only the last.
- `inherit_errexit`: `$(...)` keeps `-e` (otherwise bash turns it off inside).

Treat this as a backstop, not the error-handling strategy. Every source agrees `-e` is
incomplete; they disagree on whether to use it at all:

| Position | Source |
|---|---|
| Use `set -euo pipefail` plus `IFS=$'\n\t'` | Aaron Maxwell, "Unofficial Bash Strict Mode" (redsymbol.net) |
| Don't use `set -e`; check errors yourself | Greg's Wiki BashFAQ/105 (GreyCat) |
| `pipefail` should not be on globally (early-exit readers) | Greg's Wiki BashPitfalls #60 |
| `-u` is controversial; don't bolt it on without auditing | Greg's Wiki BashFAQ/112 |
| Strict mode gives false security; `echo "$(false)"` passes | Jane Street, "When Bash Scripts Bite" |
| Silent on strict mode; check return values, use PIPESTATUS | Google Shell Style Guide |

This skill's position: use the header (it turns many silent failures into loud ones),
skip the global `IFS` change (quote correctly instead), and still check every command
whose failure matters with `if ! cmd; then ...; fi` or `cmd || die "..."`. The cases
below are where the header silently does nothing; ShellCheck's optional
`check-set-e-suppressed` (SC2310, SC2311) and `check-extra-masked-returns` (SC2312)
flag most of them, which is why `assets/shellcheckrc` enables both.

## Where `set -e` does not exit

From the bash manual (The Set Builtin), all *tested*:

| Construct | Why it does not exit | Fix |
|---|---|---|
| `if cmd`, `while cmd`, `until cmd` | condition context | intended; that is how you test |
| `a && b`, `a \|\| b` | every command except the last in the list | intended |
| `! cmd` | status inverted | intended |
| `f` called from `if f`, `f \|\| x`, `f && x`, `! f` | **errexit is off for f's whole body**, even a `set -e` inside f | make f check its own errors (`cmd \|\| return 1`), or call f on its own line |
| `x=$(false; echo hi)` | `$(...)` drops `-e` unless `inherit_errexit` (4.4+) | `shopt -s inherit_errexit` |
| `echo "$(false)"` | the outer command's status is what counts | assign first: `out=$(cmd)`, then use `$out` |
| `local x=$(false)` (also `declare`, `export`, `readonly`) | the builtin's own status (0) replaces cmd's | `local x; x=$(cmd)` (SC2155) |
| `cmd <(failing)` | process substitution status is never checked | capture in a file, or check after with `wait $!` (bash 4.4+) |
| `a \| b` without pipefail | only `b`'s status counts | `set -o pipefail` |

Two cases where `-e` exits when you did not expect it:

- `((i++))` with `i=0`: post-increment evaluates to the old value, 0, so `((...))` returns
  1 and the script dies. Same for `((i--))` from 0 and `((--i))` from 1; `((++i))` from 0
  is fine. *Tested:* the `((...))` forms exit on 4.3+ and survive on 3.2, because `[[`
  and `((` only became subject to `set -e` in bash 4.1 (CHANGES, bash-4.1-alpha item j).
  `let i++` from 0 exits on every version, 3.2 included. Write `i=$((i + 1))`, which
  always returns 0.
- `cond && cmd` as the **last line of a function**: when `cond` is false the function
  returns 1 and the caller dies. End with `if cond; then cmd; fi` or `return 0`.

## pipefail and exit 141

With pipefail, `producer | head -n1` usually fails with 141: `head` exits after one
line, the producer's next write gets SIGPIPE (signal 13), and 128 + 13 = 141. *Tested:*
`seq 1 100000 | head -n1` gives 141 and `PIPESTATUS` is `141 0` on every version.
Whether it happens depends on how much the producer writes before `head` exits, so it can
pass in testing and fail in production. Same for `grep -q` and `grep -m`.

Fixes, in order of preference:

1. Let the reader consume everything: `awk 'NR==1'` or `sed -n 1p` instead of `head -n1`;
   `grep -c pat >/dev/null` or a full `grep` instead of `grep -q` when input is small.
2. Remove the pipe: `first=$(head -n1 < file)`, `grep -q pat < file`, `grep -q pat <<<"$var"`.
3. Accept 141 explicitly:
   ```bash
   producer | head -n1 >out || {
     s=("${PIPESTATUS[@]}")          # copy immediately; the next command overwrites it
     ((s[0] == 141 && s[1] == 0)) || exit 1
   }
   ```

## nounset (`set -u`)

- On bash 3.2, the braced forms `"${@}"` and `"${*}"` abort with no arguments under
  `set -u` (*tested*; 4.3+ is fine). Write `"$@"` and `"$*"`, unbraced, even though Google
  style otherwise prefers braces. Copies like `args=("$@")` are then empty arrays, which
  the next point covers.
- `$@`, `$*`, `${arr[@]}` and `${arr[*]}` are exempt, **except** that on bash before 4.4
  an empty array still errors: `arr=(); for x in "${arr[@]}"` dies with
  "arr[@]: unbound variable". *Tested:* fails on 3.2 and 4.3, fine on 4.4+. Stock macOS
  bash is 3.2, so a script that must run there uses `${arr[@]+"${arr[@]}"}`, which
  expands to nothing when the array is empty and to the quoted elements otherwise.
- Optional variables: `${VAR-}` (empty if unset) or `${VAR:-default}`. Optional
  positional parameters: `${1-}`.
- `${#arr[@]}` on an empty array is fine everywhere; `${arr[0]}` on it errors.
- Sourcing files you don't control (a virtualenv's `activate`, distro scripts) can trip
  `-u`: wrap with `set +u; source x; set -u`.

## The ERR trap and stack traces

The ERR trap runs when a command fails, under the same exemptions as `set -e`. Without
`-E` it is not inherited by functions. *Tested:* with `set -e` and no `-E`, a failure
inside a function exits the script **with no trap output at all**, because the shell
exits inside the function, where the trap is not set.

A handler that prints the failing command and the call stack:

```bash
set -Eeuo pipefail
on_error() {
  local status=$? i              # must be the first statement: $? is the failure's status
  printf 'error: %s exited %d at %s:%s\n' "$BASH_COMMAND" "$status" \
    "${BASH_SOURCE[1]:-$0}" "${BASH_LINENO[0]}" >&2
  for ((i = 1; i < ${#FUNCNAME[@]} - 1; i++)); do
    printf '  %s() called at %s:%s\n' "${FUNCNAME[i]}" "${BASH_SOURCE[i + 1]}" "${BASH_LINENO[i]}" >&2
  done
}
trap on_error ERR
```

On bash 3.2, a failure inside a function reports the line where the function starts
(both `$LINENO` and `BASH_LINENO[0]`; *tested*); 4.3+ reports the failing line. The
command text in `BASH_COMMAND` is right on every version, so print it too.

Variables: `BASH_COMMAND` is the command that was running when the trap fired;
`FUNCNAME[i]` is the call stack (index 0 is the current function); `FUNCNAME[i]` was
called from file `BASH_SOURCE[i+1]` at line `BASH_LINENO[i]` (bash manual, Bash
Variables). With `inherit_errexit`, a failure inside `$(...)` reports twice: once in the
subshell, once for the assignment. That is noise, not a bug.

## EXIT trap, signals, and children

Use exactly one cleanup function on EXIT. A second `trap ... EXIT` replaces the first.

```bash
tmpdir=''
cleanup() {
  if [[ -n $tmpdir ]]; then rm -rf -- "${tmpdir:?}" || true; fi
}
trap cleanup EXIT
tmpdir=$(mktemp -d)   # assign only after mktemp succeeds, so cleanup never deletes someone else's path
```

What happens on each kind of exit (*tested* on 5.2 unless noted):

| Event | EXIT trap runs? | Exit status seen by parent | Children |
|---|---|---|---|
| normal end, `exit N`, `set -e` failure | yes | N | n/a |
| Ctrl-C (SIGINT to the whole foreground process group) | yes | killed by signal 2 (130 in `$?`); see the 3.2 note below | foreground child gets SIGINT too and dies |
| `kill PID` (SIGTERM to the script only), e.g. from a supervisor or `timeout` | yes (all versions) | 143 | **a foreground child keeps running, orphaned** |
| SIGHUP | yes | 129 | same as TERM |
| SIGKILL | no | 137 | orphaned |

Bash runs the EXIT trap when it dies from a signal. Greg's Wiki (SignalTrap) says zsh
and dash do not; in a quick test zsh 5.9 skipped it on SIGTERM and SIGINT but ran it on
SIGHUP. Either way, don't rely on EXIT traps in other shells.

Bash 3.2 note (*tested*): when SIGINT arrives while the script sits in `wait`, bash 3.2
runs the EXIT trap and then exits *normally* with status 129, so the parent does not see
an interrupt. 4.3+ dies by SIGINT. Scripts that must run on 3.2 and use `wait` should add
the re-raise traps below, which fix it on 3.2 too.

For long-running children that must not outlive the script, run them in the background,
`wait` for them, and kill them in cleanup:

```bash
child=''
cleanup() {
  if [[ -n $child ]]; then kill "$child" 2>/dev/null || true; fi
  # other cleanup...
}
trap cleanup EXIT
long_running_command & child=$!
wait "$child"
```

*Tested:* SIGTERM to the script now leaves no child behind. Background jobs in a script
ignore SIGINT (job control is off; bash manual, Signals), so Ctrl-C reaches them only
through this cleanup. While bash waits for a *foreground* command, a *trapped* signal
is deferred until that command exits; `wait` is interrupted immediately, which is the
other reason for the pattern.

If you trap INT or TERM yourself (to print a message, say), re-raise the signal at the
end so the parent sees a signal death instead of a normal exit, which matters for
callers like `make` and loops in other scripts (Cracauer, "Proper handling of SIGINT/
SIGQUIT"; Greg's Wiki SignalTrap):

```bash
trap 'trap - INT; kill -s INT "$$"' INT
trap 'trap - TERM; kill -s TERM "$$"' TERM
```

*Tested:* with this plus an EXIT trap, cleanup runs once and the parent sees status 130
reported as an interrupt. `exit 130` gives the same `$?` but looks like a normal exit.

Cleanup must tolerate failure. *Tested:* under `set -e`, a failing command inside the
EXIT trap aborts the rest of the trap and replaces the exit status (an `exit 4` became 1).
Use `|| true` on each cleanup step. Also guard `${var:?}` uses in cleanup with
`[[ -n $var ]]` first, or an early exit before `var` is set prints "parameter null or not
set" from inside the trap.

Use single quotes for trap bodies (`trap 'rm -f -- "$f"' EXIT`) so variables expand when
the trap runs, not when it is set (SC2064).

## Temp files

- `mktemp` creates the file atomically with mode 600; `mktemp -d` creates a private
  directory (700). The randomness is not the security; the atomic create is
  (Greg's Wiki BashFAQ/062).
- Portable calls: `mktemp`, `mktemp -d`, or with an explicit template
  `mktemp -d "${TMPDIR:-/tmp}/name.XXXXXX"`. Avoid `-t`: deprecated in GNU, different
  meaning in BSD.
- Never write to fixed names like `/tmp/myscript.log`: another user can create that path
  first, or plant a symlink there.
- Prefer one `mktemp -d` per run and put every temp file inside it; cleanup is one `rm -rf`.

## Exit codes

| Code | Meaning | Source |
|---|---|---|
| 0 | success | bash manual, Exit Status |
| 1 | general failure | convention |
| 2 | usage error (bad options) | bash builtins use 2 for this |
| 126 | found but not executable | bash manual |
| 127 | command not found | bash manual |
| 128+N | killed by signal N (130 INT, 141 PIPE, 143 TERM) | bash manual |

Use 0/1/2 unless a caller needs more. `sysexits.h` codes (64 EX_USAGE ... 78 EX_CONFIG)
exist, but FreeBSD's sysexits(3) now marks the interface deprecated; use them only when
something downstream depends on them. Never use 126 and above for your own meanings.
Document every non-default code in `--help`.
