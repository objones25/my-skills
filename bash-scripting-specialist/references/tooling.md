# Tooling: ShellCheck, shfmt, bats, debugging

Versions current on 2026-10-22: ShellCheck 0.11.0 (2025-08-03), shfmt 3.14.1
(2026-09-06), bats-core 1.14.0 (2026-07-21; npm's `bats` package is still 1.13.0),
kcov v43. Sources: each project's CHANGELOG and the Homebrew formula API. Check for
newer releases before pinning versions in CI.

`scripts/check.sh FILE...` runs `bash -n`, ShellCheck, shfmt and the audit in one go and
skips (and says so) whatever is not installed.

## ShellCheck

Install: `brew install shellcheck`, `apt install shellcheck`, or `pip install shellcheck-py`
(a wheel with the binary).

Configuration lives in `.shellcheckrc` (or `shellcheckrc`); ShellCheck uses the first one
found walking up from each script's directory, then `~/.shellcheckrc`. Lines are
`key=value`, the same keys as inline directives. `assets/shellcheckrc` is this skill's
default: it enables `check-set-e-suppressed` (SC2310/SC2311: a function called from
`if`/`&&`/`||`/`$(...)`, where `set -e` is off) and `check-extra-masked-returns` (SC2312:
a command's status hidden inside `$(...)` used as an argument). The ShellCheck wiki
advises against `enable=all` as a default because optional checks are opinions and some
conflict; use `--list-optional` to see them.

Directives are comments of the form `# shellcheck key=value`:

- Before the first command they apply to the whole file; otherwise to the next command.
- `# shellcheck disable=SC2034` (comma lists and ranges allowed); always add the reason
  after a second `#`: `# shellcheck disable=SC2016 # expands in the child shell`.
- `# shellcheck source=lib/common.sh` tells it where a dynamic `source` goes;
  `source-path=SCRIPTDIR` in the rc resolves paths relative to the script.
- Run with `-x` (or `external-sources=true`) so it follows sourced files.
- Gotcha (*tested*, 0.11.0): any comment that begins with the word `shellcheck` is parsed
  as a directive, so `# shellcheck checks quoting` fails with SC1073/SC1072. Capitalize it
  or reword.

Exit codes: 0 clean, 1 findings (including script syntax errors), 2 some files could not
be processed, 3 bad command-line usage (e.g. an unknown flag), 4 a bad option value.

Codes worth knowing by number:

| Code | Meaning |
|---|---|
| SC2086 / SC2046 / SC2068 | unquoted variable / `$(...)` / array expansion |
| SC2155 | `local x=$(cmd)` hides cmd's status |
| SC2164 | `cd` without `\|\| exit` |
| SC2181 | `if [ $? ... ]` instead of `if cmd` |
| SC2015 | `a && b \|\| c` is not if/else |
| SC2115 | `rm -rf "$x/"*` without `${x:?}` |
| SC2064 | double-quoted trap body expands too early |
| SC2207 | `arr=($(cmd))` |
| SC2162 | `read` without `-r` |
| SC2059 | variable in a printf format |
| SC2034 / SC2154 | assigned but unused / used but unassigned |
| SC1090 / SC1091 | cannot follow a `source` |
| SC2317 / SC2329 | unreachable code / function never called; false positives for trap handlers and indirect calls, disable with a reason |
| SC2310 / SC2311 / SC2312 | (optional) `set -e` suppressed / status masked |

What ShellCheck does not do: it does not know which bash *version* a feature needs
(it treats all bash as one dialect), it does not know GNU from BSD flags, and it does not
check errexit runtime traps like `((i++))` or pipefail with `head`. That is what
`scripts/audit_bash.py` adds.

## shfmt

Install: `brew install shfmt`, `go install mvdan.cc/sh/v3/cmd/shfmt@latest`, or
`pip install shfmt-py`.

- `shfmt -i 2 -ci -bn -d file` shows a diff in Google style (2-space indent, indented
  `case` branches, `&&`/`|` at the start of continuation lines; the shfmt man page names
  this flag set as the Google-style preset). `-w` writes the result. `-d` exits non-zero
  when formatting differs, which is the CI check.
- Formatting flags on the command line disable `.editorconfig` settings entirely; use one
  or the other. The EditorConfig equivalent of `-i 2 -ci -bn` is `indent_style = space`,
  `indent_size = 2`, `case_indent = true`, `binary_next_line = true` (shfmt man page;
  `switch_case_indent` is the older spelling).
- `-ln bash` forces the dialect. Without it, a `#!/bin/sh` shebang makes shfmt parse the
  file as POSIX and reject bash syntax such as arrays (*tested*, 3.14.1).
- shfmt parses more strictly than `bash -n` and catches some errors bash only finds at
  run time.

## bats-core (tests)

Install: `brew install bats-core`, `npm install -g bats` (1.13.0 as of this writing), or
clone the repo and run `./install.sh /usr/local`. Helper libraries `bats-support`,
`bats-assert` and `bats-file` are separate repos, usually added as git submodules under
`test/test_helper/`.

`assets/template.bats` is a working suite for `assets/template.sh` (10 tests, passing on
bats 1.13.0 and 1.14.0, and on bash 3.2 and 5.2; October 2026, machine not recorded; not rerun on 2026-11-03). Patterns it shows:

```bash
bats_require_minimum_version 1.5.0     # otherwise old bats reads new run flags as the command
run -2 --separate-stderr "$SCRIPT" --bogus   # assert exit 2, keep stderr in $stderr
[[ $stderr == *"unknown option"* ]]
```

- `run` executes in a subshell and sets `$status`, `$output` (stdout+stderr unless
  `--separate-stderr`), `$stderr`, and `${lines[@]}`. `run -N` asserts the status;
  `run !` asserts non-zero.
- A bare `! cmd` in the middle of a test never fails it, because bash's errexit ignores
  `!`. Use `run ! cmd`.
- `run cmd | grep x` pipes `run`'s (empty) output; use `run bats_pipe cmd \| grep x`.
- Each test gets `$BATS_TEST_TMPDIR`; `cd` into it in `setup` so scripts that write files
  don't touch the repo.
- Background processes inherit fd 3, which bats uses; close it (`cmd 3>&- &`) or bats
  waits for them.
- To unit-test functions, source the script *in a child bash*:
  `run bash -c 'source "$1"; my_func arg' _ "$SCRIPT"`. Sourcing directly into the test
  shell imports the script's `set -e` and traps, and its function names can shadow bats
  helpers. *Tested:* a template function named `run` replaced bats' `run` and silently
  aborted the suite, which is why the template calls it `run_cmd`. This works only if
  the script guards its entry point with `if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi`.
- bats 1.14.0 breaking changes: `run` no longer suppresses a `set -e` that is already in
  effect inside the called code (for example a function that runs `set -e`, or one from a
  sourced script); `--errexit` turns errexit on for everything run. An empty suite is now
  an error (`--allow-empty-suite` restores the old behavior). *Tested:* a function that
  runs `set -e; false; echo after` returns 0 under `run` on 1.13 and 1 on 1.14.
- Test old bash: `PATH=/bin:$PATH bats test/` on a Mac runs both bats and the script under
  3.2 (*tested* October 2026, machine not recorded, with a locally built 3.2.57: all 10 template tests pass).

Alternatives: ShellSpec (BDD style, mocking, built-in kcov; last release 0.28.1 in 2021)
and shUnit2 (xUnit style, sourced at the end of the test file).

## Debugging

- `bash -n script` parses without running (syntax only, and only for the bash you run it
  with).
- `bash -x script args`, or `set -x` ... `set +x` around a region, prints each command
  after expansion to stderr. Make the prefix useful:
  ```bash
  PS4='+ ${BASH_SOURCE[0]}:${LINENO}:${FUNCNAME[0]:-main}: '
  ```
  The `:-main` matters: on bash 3.2 and 5.x `FUNCNAME` is unset at the top level, so
  under `set -u` the common recipe without it prints "unbound variable" on every
  top-level line (on 4.3 and 4.4 it happens to read "main").
- `BASH_XTRACEFD` (4.1+) sends the trace to another fd so it doesn't mix with stderr:
  `exec {fd}>trace.log; BASH_XTRACEFD=$fd; set -x`. On 3.2 the variable has no effect and
  the trace stays on stderr. Don't set it to 2 and later unset it: unsetting closes that fd.
- Opt-in tracing: `if [[ ${TRACE-0} == 1 ]]; then set -x; fi`, run as `TRACE=1 ./script`.
- Step through: `trap 'read -rp "[$LINENO] $BASH_COMMAND? " _' DEBUG`.
- Coverage: `kcov out/ ./script args` writes HTML and Cobertura XML.
- Version-dependent behavior: `scripts/test_matrix.sh -- script args` runs it under
  bash 3.2, 4.4 and 5.3 (Docker) or under local binaries (`--bash PATH`).
