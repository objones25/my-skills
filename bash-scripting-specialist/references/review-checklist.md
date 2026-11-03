# Review checklist for an existing script

Run `scripts/check.sh --target X.Y [--platform macos] FILE` first and fix or justify every
finding. Then walk this list in order; earlier items cause data loss or silent wrong
results, later ones cost readability. Report each finding with the line, what goes wrong
(a concrete input or situation), and the exact fix.

## 1. Destructive operations
- [ ] Every `rm -r`, `mv`, `>` truncation, and `find -delete` targets a path that cannot be
      empty or `/`: `${var:?}`, `--`, and a check that the path is where it should be.
- [ ] `cd` failures stop the script (`cd -- "$d" || exit`), especially before relative
      deletes.
- [ ] Destructive actions have `--dry-run`, or a confirmation that refuses without a
      terminal.

## 2. Failure handling
- [ ] An errexit strategy exists (`set -Eeuo pipefail`), *and* commands whose failure
      matters are checked explicitly.
- [ ] No functions that rely on `set -e` are called from `if`, `&&`, `||`, `!`, or `$(...)`
      (SC2310/SC2311); predicates check their own errors.
- [ ] No `local x=$(cmd)` (SC2155); no `cmd "$(other)"` where `other` can fail (SC2312).
- [ ] No `((x++))` / `((x--))` statements under `set -e`; no function ending in
      `cond && cmd`.
- [ ] Pipelines with `head`, `grep -q`, `grep -m` under pipefail handle exit 141.
- [ ] Background jobs are waited on by PID and their statuses collected.
- [ ] One EXIT trap does all cleanup; cleanup tolerates failures; long-running children
      are killed in cleanup.
- [ ] Temp files come from `mktemp`, live in one `mktemp -d`, and are removed on exit.
- [ ] Exit codes: 0 success, 1 failure, 2 usage, documented in `--help`.

## 3. Input and security
- [ ] Every expansion is quoted unless splitting is intended and explained (SC2086).
- [ ] No `eval` of anything containing outside data; commands built in arrays.
- [ ] Numbers from outside are validated before arithmetic or `-eq` comparisons.
- [ ] Filenames handled NUL-safely (`-print0`, `read -d ''`); no `for f in $(ls ...)`.
- [ ] No secrets in command-line arguments or xtrace output; secret files created under
      `umask 077`.
- [ ] No `curl | sh`; downloads verified by checksum.
- [ ] Commands passed to `ssh`/`sh -c` quote arguments for the remote side.

## 4. Portability
- [ ] The target bash version is stated, the audit passes against it, and a version guard
      exists if the target is above 3.2 and macOS users might run it.
- [ ] Shebang is `#!/usr/bin/env bash` (or a deliberate fixed path).
- [ ] No GNU-only flags if it can run on macOS (sed -i, date -d, stat -c, grep -P,
      find -printf, timeout, flock, getopt --long).
- [ ] `${var/pat/rep}` replacements containing `&` behave the same on 5.2+.
- [ ] Ran on the oldest target bash (`/bin/bash` on a Mac, or `scripts/test_matrix.sh`).

## 5. Interface
- [ ] `--help` describes usage, options, inputs, outputs, and exit codes.
- [ ] Data goes to stdout, messages to stderr; errors name the thing that failed.
- [ ] Usage errors exit 2 and point to `--help`; unknown options are rejected.
- [ ] `--` ends options; filenames starting with `-` work.

## 6. Structure and style
- [ ] Functions, then `main`, then a guarded `main "$@"`; variables in functions are
      `local`; constants `readonly`.
- [ ] `shfmt -i 2 -ci -bn -d` clean (or the repo's `.editorconfig`).
- [ ] Comments explain why, not what; every ShellCheck/audit suppression has a reason.
- [ ] Over ~100 lines with real logic, data structures beyond flat lists, JSON, or
      floating point: say whether it should be Python instead.
- [ ] Tests exist for the interface (bats), at least `--help`, a usage error, the main
      path, and one failure path.
