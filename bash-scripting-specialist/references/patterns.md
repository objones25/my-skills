# Patterns

Read this when the script needs one of these jobs done. Each snippet ran under
`set -Eeuo pipefail` on bash 3.2.57 and 5.3.20 unless a minimum version is named.

## Script layout

`assets/template.sh` is the full skeleton. The order, from Google's Shell Style Guide:
shebang and header comment; `set` options; constants (`readonly`, UPPER_CASE for
exported or constant values); functions; `main`; and last, the guarded call:

```bash
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
```

The guard runs `main` when the file is executed and skips it when the file is sourced,
which lets tests source the functions. Keep stdout for the script's real output (data a
caller might pipe) and send every message, warning and prompt to stderr.

## Arguments

Short options only: the `getopts` builtin. A leading `:` in the option string makes it
silent so you print your own errors.

```bash
verbose=0 output=''
while getopts ':vo:h' opt; do
  case $opt in
    v) verbose=$((verbose + 1)) ;;
    o) output=$OPTARG ;;
    h) usage; exit 0 ;;
    :) usage_error "-$OPTARG needs an argument" ;;
    \?) usage_error "unknown option -$OPTARG" ;;
  esac
done
shift $((OPTIND - 1))
```

Reset `OPTIND=1` (or `local OPTIND`) before calling getopts again in the same shell.

Long options (`--output=FILE`): a `while`/`case` loop, as in `parse_args` in
`assets/template.sh`. Don't use the external `getopt`: the long-option version is
util-linux only, and macOS ships the old BSD one, which breaks on arguments with spaces
(Greg's Wiki BashFAQ/035, ComplexOptionParsing).

Conventions: `-h`/`--help` prints usage to stdout and exits 0; a usage error prints a
short message plus "Try --help" to stderr and exits 2; `--` ends options; options and
positional arguments may be mixed.

## Requiring commands

```bash
require() {
  local cmd missing=0
  for cmd in "$@"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      printf 'missing required command: %s\n' "$cmd" >&2
      missing=1
    fi
  done
  return "$missing"
}
require jq curl || exit 1
```

Check all dependencies up front, so a run doesn't fail halfway after changing things.

## Logging

```bash
LOG_LEVEL=${LOG_LEVEL:-1}   # 0 errors only, 1 info, 2 debug
log() {
  local level=$1 now
  shift
  now=$(date '+%Y-%m-%dT%H:%M:%S')
  printf '%s %s %s\n' "$now" "$level" "$*" >&2
}
info() { if ((LOG_LEVEL >= 1)); then log INFO "$@"; fi; }
debug() { if ((LOG_LEVEL >= 2)); then log DEBUG "$@"; fi; }
error() { log ERROR "$@"; }
```

On bash 4.2+ the `date` call can be the builtin `printf -v now '%(%Y-%m-%dT%H:%M:%S)T' -1`,
which avoids starting a process per log line.

Write the level checks as `if ...; then ...; fi`, not `((...)) && log`: the `&&` form
returns 1 when the level is off, which kills the caller under `set -e` if it is the last
command in a function.

Color only when stderr is a terminal and the user has not opted out (no-color.org):

```bash
if [[ -t 2 && -z ${NO_COLOR-} ]]; then red=$'\e[31m' reset=$'\e[0m'; else red='' reset=''; fi
```

## Reading files and command output

```bash
while IFS= read -r line || [[ -n $line ]]; do      # every line, even an unterminated last one
  printf '%s\n' "$line"
done < "$file"

while IFS= read -r -d '' path; do                   # filenames, safe for any character
  printf '%s\n' "$path"
done < <(find "$dir" -type f -name '*.log' -print0)

mapfile -t lines < "$file"                           # whole file into an array, bash 4.0+
lines=(); while IFS= read -r l || [[ -n $l ]]; do lines+=("$l"); done < "$file"   # same on 3.2

IFS=, read -r name age city <<<"$csv_line"           # split one line into fields
```

`< <(cmd)` (process substitution) keeps the loop in the current shell, so variables set
inside survive; `cmd | while` does not. Process substitution hides `cmd`'s exit status;
if it matters, write to a temp file first and check the command directly.

## Script directory

```bash
script_dir=$(CDPATH='' cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
```

Works on bash 3.2 and macOS (no `readlink -f`). `-P` resolves symlinked directories, but
if the script file itself is a symlink this gives the symlink's directory. Greg's Wiki
(BashFAQ/028) adds that needing this at all is often a design smell: prefer config in
fixed places (`/etc/app`, `~/.config/app`) or paths passed as arguments.

## Temp files and cleanup

See errors-and-traps.md. In short: one `mktemp -d` per run, one `cleanup` on EXIT,
failure-tolerant cleanup steps, long-running children run with `& wait` and killed in
cleanup.

## Running jobs in parallel

`wait` with no arguments returns 0 even when jobs failed (*tested* on every version), so
collect PIDs and wait for each:

```bash
((${#hosts[@]} > 0)) || die "no hosts given"   # also keeps set -u happy on bash < 4.4
pids=() failed=0
for host in "${hosts[@]}"; do
  deploy "$host" & pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
((failed == 0)) || die "some deploys failed"
```

`pids` is non-empty after the loop because `hosts` was. A bounded pool (bash 4.3+, `wait -n` returns as soon as any job ends; 127 when none are
left):

```bash
max=4 running=0 failed=0
for item in "${items[@]}"; do
  if ((running >= max)); then
    wait -n || failed=1
    running=$((running - 1))
  fi
  work "$item" &
  running=$((running + 1))
done
while ((running > 0)); do
  wait -n || failed=1
  running=$((running - 1))
done
```

For many small independent commands, `xargs -0 -P 4 -n 1 cmd` is simpler; it exits 123
if any invocation failed. Output from parallel jobs interleaves; write each job's output
to its own file in the temp directory if order matters.

## Locking (one instance at a time)

Portable (macOS and Linux): `mkdir` is atomic.

```bash
lockdir=${TMPDIR:-/tmp}/myscript.lock
if ! mkdir -- "$lockdir" 2>/dev/null; then
  die "another run holds $lockdir (remove it if no run is active)"
fi
trap 'rmdir -- "$lockdir"' EXIT   # rmdir: removes only an empty dir; merge into cleanup()
```

A SIGKILLed run leaves the lock behind. On Linux only, `flock` releases automatically
when the process dies: `exec 9>/var/lock/myscript; flock -n 9 || die "already running"`.

## Retries with backoff

```bash
retry() {
  local attempts=$1 delay=$2 n=1
  shift 2
  until "$@"; do
    if ((n >= attempts)); then
      printf 'failed after %d attempts: %s\n' "$n" "$*" >&2
      return 1
    fi
    sleep "$delay"
    n=$((n + 1)) delay=$((delay * 2))
  done
}
retry 5 1 curl -fsS -o out.json "$url"
```

The command runs in an `until` condition, where `set -e` is off, so its failures are
handled by the loop instead of killing the script.

## Dry run and confirmation

```bash
run_cmd() {   # print instead of run when --dry-run was given
  if ((dry_run)); then
    printf 'would run:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
  else
    "$@"     # the real command's stdout stays on stdout
  fi
}

confirm() {   # require an explicit yes from a terminal; refuse when not interactive
  local reply
  [[ -t 0 ]] || return 1
  read -r -p "$1 [y/N] " reply
  [[ $reply == [yY] || $reply == [yY][eE][sS] ]]
}
confirm "Delete ${#files[@]} files?" || die "aborted"
```

For anything destructive, offer `--dry-run` and make the default the safe choice. Scripts
run from cron or CI have no terminal, so `confirm` returns 1 there; give them a `--yes`
flag instead of piping `yes`.

## Strings without external commands

| Need | Bash (3.2+ unless noted) |
|---|---|
| strip prefix / suffix | `${path#*/}` `${path##*/}` (basename) `${path%/*}` (dirname) `${file%.*}` |
| replace | `${s/old/new}` first, `${s//old/new}` all (see portability.md on `&` in 5.2+) |
| length, substring | `${#s}`, `${s:2:3}` |
| lower/upper | `${s,,}` `${s^^}` (4.0+); 3.2: `tr '[:upper:]' '[:lower:]'` |
| default value | `${v:-default}`, assign if unset/empty `${v:=default}`, required `${v:?message}` |
| regex match | `[[ $s =~ ^([a-z]+)-([0-9]+)$ ]] && name=${BASH_REMATCH[1]}` |
| trim whitespace | `s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}` |
| join array with commas | `(IFS=,; printf '%s\n' "${arr[*]}")` |
