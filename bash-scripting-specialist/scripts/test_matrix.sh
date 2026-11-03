#!/usr/bin/env bash
#
# test_matrix.sh: run one command under several bash versions and compare exit codes.
#
# Static checks cannot prove a script runs on old bash; running it can. By default this
# uses the official Docker images (bash:3.2 is the version stock macOS ships). Pass
# --bash PATH (repeatable) to use local binaries instead, e.g. /bin/bash on a Mac.
#
# Usage: test_matrix.sh [--versions "3.2 4.4 5.3"] [--bash PATH]... -- SCRIPT [ARGS...]
#
# The current directory is mounted at /work in each container, so SCRIPT and its inputs
# should be given relative to it. The images are Alpine with BusyBox tools, so GNU-only
# flags fail there just as BSD tools would on macOS; treat that as a portability signal.
# Exit status: 0 every run matched the first (exit code and output), 1 a run differed,
# 2 usage error.

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true
# Runs on stock macOS bash 3.2; cmd and bashes are length-checked before expansion.
# audit-bash disable-file=B202

readonly SCRIPT_NAME=${0##*/}

usage() {
  sed -n '3,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

usage_error() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
  printf "Try '%s --help'.\n" "$SCRIPT_NAME" >&2
  exit 2
}

versions='3.2 4.4 5.3'
bashes=()
cmd=()

parse_args() {
  while (($# > 0)); do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      --versions)
        (($# >= 2)) || usage_error "--versions needs a list like \"3.2 5.3\""
        versions=$2
        shift
        ;;
      --bash)
        (($# >= 2)) || usage_error "--bash needs a path"
        bashes+=("$2")
        shift
        ;;
      --)
        shift
        cmd=("$@")
        break
        ;;
      *) usage_error "unexpected argument: $1 (put the command after --)" ;;
    esac
    shift
  done
  ((${#cmd[@]} > 0)) || usage_error "no command given after --"
  [[ -f ${cmd[0]} ]] || usage_error "script not found: ${cmd[0]} (paths are relative to the current directory)"
}

# run_one LABEL CMD... : runs the command, prints one result line, sets last_rc/last_out
last_rc=0
last_out=''
run_one() {
  local label=$1 out rc tail_lines
  shift
  out=$("$@" 2>&1) && rc=0 || rc=$?
  last_rc=$rc
  last_out=$out
  tail_lines=$(printf '%s\n' "$out" | tail -n 3 | tr '\n' '|' | cut -c1-110) || true
  printf '%-22s exit=%-4s %s\n' "$label" "$rc" "$tail_lines"
}

# compare: remember the first run, flag any later run whose exit code or output differs
first_rc=''
first_out=''
rc_differ=0
out_differ=0
compare() {
  if [[ -z $first_rc ]]; then
    first_rc=$last_rc
    first_out=$last_out
    return 0
  fi
  if [[ $last_rc != "$first_rc" ]]; then rc_differ=1; fi
  if [[ $last_out != "$first_out" ]]; then out_differ=1; fi
}

main() {
  parse_args "$@"
  local v b label

  if ((${#bashes[@]} > 0)); then
    for b in "${bashes[@]}"; do
      [[ -x $b ]] || usage_error "not executable: $b"
      # shellcheck disable=SC2016 # $BASH_VERSION must expand in the child bash
      label=$("$b" -c 'printf "bash %s" "$BASH_VERSION"') || label=$b
      run_one "${label:0:22}" "$b" "${cmd[@]}"
      compare
    done
  else
    command -v docker >/dev/null 2>&1 \
      || usage_error "docker not found; install it or pass --bash PATH for local bash binaries"
    if ! docker info >/dev/null 2>&1; then
      printf '%s: docker is installed but its daemon is not reachable; start it or use --bash PATH\n' "$SCRIPT_NAME" >&2
      return 1
    fi
    for v in $versions; do
      run_one "bash:$v (docker)" docker run --rm -v "$PWD:/work" -w /work "bash:$v" bash "${cmd[@]}"
      compare
    done
  fi

  if ((rc_differ)); then
    printf '\nExit codes differ across versions: behavior depends on the bash version.\n' >&2
    return 1
  fi
  if ((out_differ)); then
    printf '\nSame exit code (%s) but output differs across versions; read the lines above.\n' "$first_rc" >&2
    return 1
  fi
  printf '\nSame exit code (%s) and output on every version.\n' "$first_rc"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
