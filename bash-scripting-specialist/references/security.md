# Security

Read this when a script handles input it did not create (arguments, files, network
data, environment from another user), runs with elevated privileges, handles secrets,
or builds commands for another shell (ssh, `sh -c`, `find -exec`).

## Data must never become code

The shell re-parses text in a few places. Untrusted data must not reach any of them.

**eval.** `eval` parses its argument as shell code, so `name='x; rm -rf ~'; eval "user=$name"`
runs the `rm` (BashFAQ/048). Alternatives:

| Goal | Instead of eval |
|---|---|
| run a command assembled at run time | an array: `cmd=(grep -r -- "$pat" "$dir"); "${cmd[@]}"` |
| assign to a variable whose name is computed | `printf -v "$name" '%s' "$value"` |
| refer to a variable by name | `declare -n ref=$name` (4.3+) |
| look up keyed data | `declare -A map` (4.0+) or a `case` statement |

Validate computed names before using them, even with `printf -v` or namerefs:
`[[ $name =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "bad name"`. A nameref to the string
`x[$(date)]` still runs `date` (BashFAQ/048).

**Arithmetic.** Every arithmetic context evaluates variable *values* as expressions, and
an array subscript inside that expression runs command substitutions. Contexts:
`$(( ))`, `(( ))`, `let`, array indexes, `${v:offset:len}`, `declare -i` assignments, and
both sides of `[[ ... -eq/-ne/-lt/-le/-gt/-ge ... ]]` (Vidar Holen, "Security
implications of arithmetic expansion in bash", 2018; NCC Group, 2020). Quoting does not
help. *Tested on 3.2 through 5.3 (October 2026, machine not recorded):*

```bash
x='a[$(echo INJECTED >&2)]'
[[ $x -eq 0 ]]      # prints INJECTED
(( x == 0 ))        # prints INJECTED
```

Validate with a pattern match, which never evaluates the input, then strip leading zeros:

```bash
is_uint() { [[ -n $1 && $1 != *[!0123456789]* ]]; }   # Greg's Wiki BashFAQ/054
is_uint "$n" || die "not a number: $n"
n=$((10#$n))
```

The wiki lists the digits instead of `[0-9]` because ranges can match other characters
in some locales.

**Commands for another shell.** `ssh host cmd args` joins the arguments into one string
that the remote shell parses again, so spaces and metacharacters are reinterpreted
(BashFAQ/096). Quote each argument for the remote side:

```bash
ssh host "printf '%s\n' ${*@Q}"           # bash 4.4+, remote login shell must be bash
printf -v q ' %q' "$@"; ssh host "cmd$q"  # bash 3.1+, same remote-bash caveat
ssh host 'bash -s' <<'EOF'                # or send a fixed script on stdin
...
EOF
```

With `find -exec` or `sh -c`, pass data as arguments, never inside the code string:
`sh -c 'for f; do grep -q x "$f"; done' _ {} +`.

## Secrets

- Command-line arguments are visible to every user on the machine through `ps` and
  `/proc/PID/cmdline` (default `hidepid=0`, proc(5)). *Tested:* `ps -eo args` showed
  another process's `--password=...`. Pass secrets on stdin
  (`printf '%s' "$token" | tool --password-stdin`; `printf` is a builtin, so the secret
  never appears in an argv), in a 0600 file (`--password-file`), or in an environment
  variable (readable by the same user and root via `/proc/PID/environ`).
- Never `set -x` around secret handling; xtrace prints expanded values. Turn it off
  locally: `{ set +x; } 2>/dev/null` before, and back on after if needed.
- Create secret files with a restrictive umask so they are never briefly world-readable:
  `(umask 077; printf '%s' "$secret" > "$file")`.
- Don't put secrets in files under a shared `/tmp` with fixed names; use `mktemp`
  (mode 600) inside a `mktemp -d` directory.

## Temp files and races

- Fixed names in `/tmp` can be created first by another user, or pre-placed as a
  symlink pointing at a file you can write. Use `mktemp`/`mktemp -d`; the atomic create is
  what makes them safe (BashFAQ/062).
- Check-then-act on files (`[[ ! -e $f ]] && echo x > "$f"`) is a race. For "create only
  if absent", use `set -o noclobber` (`set -C`) so `>` fails on existing files, or
  `mkdir` (atomic) for locks.
- Locking: `flock` is util-linux only (Linux); on macOS use `mkdir "$lockdir"` with an
  EXIT trap that removes it. A lock left by a SIGKILLed run must be removed by hand
  (BashFAQ/045).

## Privileged scripts

- SUID/SGID on shell scripts is forbidden; use `sudo` with a specific rule (Google Shell
  Style Guide).
- Set `PATH` explicitly at the top (`PATH=/usr/sbin:/usr/bin:/sbin:/bin; export PATH`) and
  don't trust `HOME`, `USER`, or `TMPDIR` from the environment.
- Exported functions override commands: *tested*,
  `env 'BASH_FUNC_ls%%=() { echo hijacked; }' bash -c ls` prints "hijacked". Privileged
  mode (`bash -p`, or `set -p`) skips `BASH_ENV`/`ENV`, does not import exported functions,
  and ignores `SHELLOPTS`, `BASHOPTS`, `CDPATH` and `GLOBIGNORE` from the environment (bash
  manual, The Set Builtin); *tested*: with `-p` the real `ls` ran. Root-run scripts called
  with an untrusted environment can use `#!/bin/bash -p` (note: a fixed path, so on macOS
  that is 3.2) or call commands by absolute path.
- Bash ignores an `IFS` value inherited from the environment (*tested*), but a file you
  `source` can change it; scripts that source untrusted files have bigger problems.
- Shellshock (CVE-2014-6271, NVD) let environment variable values run code in bash up to
  4.3 before patches. Lesson: environment crossing a privilege boundary is untrusted
  input. The fixes shipped as patches for every maintained release; for 3.2 they are
  bash32-052 through 057 (ftp.gnu.org), so stock macOS 3.2.57 includes them.

## Downloads

`curl ... | sh` runs whatever arrives, including a truncated download, and the server can
tell it is being piped to a shell and serve different content. Download to a file,
verify a pinned checksum (`sha256sum -c`, or `shasum -a 256 -c` on macOS) or signature,
then run it:

```bash
curl -fsSLo install.sh "https://example.com/install-1.2.3.sh"
printf '%s  install.sh\n' "$EXPECTED_SHA256" | sha256sum -c -
bash install.sh
```

## Input validation checklist

- Positional parameters and option values: check presence (`(($# >= 2))`), then shape
  (number, path under an expected directory, one of a fixed set via `case`).
- Paths: use `--` before them in every command; reject or resolve `..` if the path must
  stay inside a directory; never `rm -rf` a path built from unchecked input.
- Filenames from `find`/globs: NUL-delimit (`-print0`, `read -d ''`, `mapfile -d ''`).
- Anything going into arithmetic: `is_uint` first.
- Anything going to another shell: `%q`/`@Q`, or pass as arguments.
