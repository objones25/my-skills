#!/usr/bin/env bash
#
# NAME: one-line description of what this script does.
#
# Usage: see usage() below, or run with --help.
# Requires: bash 3.2+ (stock macOS), plus: <list external commands>.
# bash-target: 3.2

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly SCRIPT_NAME=${0##*/}

# ---- logging: everything diagnostic goes to stderr; stdout is for data ----

verbose=0

log() {
  local now
  now=$(date '+%Y-%m-%dT%H:%M:%S')
  printf '%s %s: %s\n' "$now" "$SCRIPT_NAME" "$*" >&2
}
debug() {
  if ((verbose > 0)); then log "debug: $*"; fi
}
die() {
  log "error: $*"
  exit 1
}

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options] [--] INPUT...

One-paragraph description of what the script does and what it prints.

Options:
  -o, --output FILE   write results to FILE (default: stdout)
  -n, --dry-run       print what would be done without doing it
  -v, --verbose       print debug messages to stderr
  -h, --help          show this help and exit

Short options cannot be bundled: write -n -v, not -nv.
Exit status: 0 success, 1 runtime error, 2 usage error.
EOF
}

usage_error() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
  printf "Try '%s --help'.\n" "$SCRIPT_NAME" >&2
  exit 2
}

# ---- cleanup and error reporting ----

tmpdir=''

cleanup() {
  # Runs on normal exit, on errors, and when bash dies from SIGINT/SIGTERM/SIGHUP.
  # Keep it failure-tolerant: under set -e a failing command here would abort the
  # rest of cleanup and replace the script's exit status.
  if [[ -n $tmpdir ]]; then rm -rf -- "${tmpdir:?}" || true; fi
}

on_error() {
  local status=$? # must stay the first statement: $? is the failed command's status
  # BASH_LINENO[0] is the failing line on bash 4+. On 3.2, a failure inside a function
  # reports the line where that function starts.
  log "error: line ${BASH_LINENO[0]}: '$BASH_COMMAND' exited with status $status"
}

trap cleanup EXIT
trap on_error ERR

# ---- argument parsing (handles short, long, --opt=value, and --) ----

output=''
dry_run=0
inputs=()

parse_args() {
  while (($# > 0)); do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      -v | --verbose) verbose=$((verbose + 1)) ;;
      -n | --dry-run) dry_run=1 ;;
      -o | --output)
        (($# >= 2)) || usage_error "$1 needs an argument"
        output=$2
        shift
        ;;
      --output=*) output=${1#*=} ;;
      --)
        shift
        inputs+=("$@")
        break
        ;;
      -?*) usage_error "unknown option: $1" ;;
      *) inputs+=("$1") ;;
    esac
    shift
  done
  ((${#inputs[@]} > 0)) || usage_error "at least one INPUT is required"
}

# ---- the actual work ----

run_cmd() {
  # Runs a command, or only prints it in dry-run mode.
  if ((dry_run)); then
    printf 'would run:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
  else
    "$@"
  fi
}

process_one() {
  local input=$1
  [[ -r $input ]] || die "cannot read: $input"
  debug "processing $input"
  run_cmd wc -l -- "$input"
}

main() {
  parse_args "$@"

  tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/$SCRIPT_NAME.XXXXXX")

  if [[ -n $output ]]; then
    exec >"$output"
  fi

  # ${arr[@]+"${arr[@]}"} keeps set -u quiet on empty arrays in bash < 4.4.
  local input
  for input in ${inputs[@]+"${inputs[@]}"}; do
    process_one "$input"
  done
}

# Run main only when executed, not when sourced (lets tests source the functions).
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
