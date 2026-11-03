#!/usr/bin/env bash
#
# check.sh: run every available static check on bash scripts and summarize.
#
#   bash -n        syntax, as parsed by the local bash
#   ShellCheck     quoting, splitting, masked exit codes, common bugs
#   shfmt -d       formatting (diff only; --fix rewrites the files)
#   audit_bash.py  target-version, GNU/BSD, errexit, and security checks
#
# Usage: check.sh [--target X.Y] [--platform any|linux|macos] [--fix] [--] FILE...
# Exit status: 0 all run checks passed, 1 a check failed, 2 usage error.
# Missing tools are reported and skipped, never counted as passes.

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true
# Runs on stock macOS bash 3.2. files/summary are never empty where expanded and the one
# possibly-empty array (audit opts) uses the ${a[@]+"${a[@]}"} guard.
# audit-bash disable-file=B202

readonly SCRIPT_NAME=${0##*/}
script_dir=$(CDPATH='' cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly script_dir

usage() {
  sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

usage_error() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
  printf "Try '%s --help'.\n" "$SCRIPT_NAME" >&2
  exit 2
}

target=''
platform=''
fix=0
files=()

parse_args() {
  while (($# > 0)); do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      --target)
        (($# >= 2)) || usage_error "--target needs a version like 4.4"
        target=$2
        shift
        ;;
      --target=*) target=${1#*=} ;;
      --platform)
        (($# >= 2)) || usage_error "--platform needs any, linux, or macos"
        platform=$2
        shift
        ;;
      --platform=*) platform=${1#*=} ;;
      --fix) fix=1 ;;
      --)
        shift
        files+=("$@")
        break
        ;;
      -?*) usage_error "unknown option: $1" ;;
      *) files+=("$1") ;;
    esac
    shift
  done
  ((${#files[@]} > 0)) || usage_error "no files given"
  if [[ -n $target && ! $target =~ ^[0-9]+\.[0-9]+$ ]]; then
    usage_error "--target must look like 3.2 or 4.4, got: $target"
  fi
  case $platform in
    '' | any | linux | macos) ;;
    *) usage_error "--platform must be any, linux, or macos, got: $platform" ;;
  esac
}

# Predicates below are called from `if`, where set -e is off (SC2310), so they
# check their own errors explicitly.

# Prints the first .shellcheckrc/shellcheckrc found from the file's directory upward.
find_shellcheckrc() {
  local dir
  dir=$(CDPATH='' cd -P -- "$(dirname -- "$1")" && pwd -P) || return 1
  while :; do
    if [[ -f $dir/.shellcheckrc ]]; then
      printf '%s\n' "$dir/.shellcheckrc"
      return 0
    fi
    if [[ -f $dir/shellcheckrc ]]; then
      printf '%s\n' "$dir/shellcheckrc"
      return 0
    fi
    [[ $dir != / ]] || return 1
    dir=$(dirname -- "$dir")
  done
}

has_editorconfig() {
  local dir
  dir=$(CDPATH='' cd -P -- "$(dirname -- "$1")" && pwd -P) || return 1
  while :; do
    [[ ! -f $dir/.editorconfig ]] || return 0
    [[ $dir != / ]] || return 1
    dir=$(dirname -- "$dir")
  done
}

failed=0
summary=()

record() {
  # record TOOL STATUS(pass|FAIL|skipped) [NOTE]
  summary+=("$(printf '%-12s %-8s %s' "$1" "$2" "${3-}")")
  if [[ $2 == FAIL ]]; then failed=1; fi
}

check_syntax() {
  local f ok=1
  for f in "${files[@]}"; do
    if ! bash -n -- "$f"; then ok=0; fi
  done
  if ((ok)); then
    record "bash -n" pass "(local bash $BASH_VERSION; older bash may still fail at run time)"
  else
    record "bash -n" FAIL
  fi
}

check_shellcheck() {
  if ! command -v shellcheck >/dev/null 2>&1; then
    record shellcheck skipped "not installed: brew install shellcheck / apt install shellcheck / pip install shellcheck-py"
    return 0
  fi
  local f ok=1 rc_note='' opts
  for f in "${files[@]}"; do
    opts=(-x)
    # shellcheck disable=SC2310 # predicate; checks its own errors
    if ! find_shellcheckrc "$f" >/dev/null; then
      opts+=(--rcfile "$script_dir/../assets/shellcheckrc")
      rc_note='(skill default rc)'
    fi
    if ! shellcheck "${opts[@]}" -- "$f"; then ok=0; fi
  done
  if ((ok)); then record shellcheck pass "$rc_note"; else record shellcheck FAIL "$rc_note"; fi
}

check_shfmt() {
  if ! command -v shfmt >/dev/null 2>&1; then
    record shfmt skipped "not installed: brew install shfmt / go install mvdan.cc/sh/v3/cmd/shfmt@latest / pip install shfmt-py"
    return 0
  fi
  local f ok=1 opts first
  for f in "${files[@]}"; do
    # Any parser or printer flag disables .editorconfig, so with one present pass none
    # (set language_dialect there if needed). Otherwise pick the dialect from the shebang.
    opts=(-d)
    # shellcheck disable=SC2310 # predicate; checks its own errors
    if ! has_editorconfig "$f"; then
      IFS= read -r first <"$f" || true
      case $first in
        '#!'*bash*) opts+=(-ln bash) ;;
        '#!'*/sh | '#!'*' sh') opts+=(-ln posix) ;;
        *) ;;
      esac
      opts+=(-i 2 -ci -bn)
    fi
    if ((fix)); then opts[0]=-w; fi
    if ! shfmt "${opts[@]}" -- "$f"; then ok=0; fi
  done
  if ((ok && fix)); then
    record shfmt pass "files rewritten"
  elif ((ok)); then
    record shfmt pass "no diff"
  elif ((fix)); then
    record shfmt FAIL "could not parse"
  else
    record shfmt FAIL "see the diff or parse error above; --fix applies a diff"
  fi
}

check_audit() {
  if ! command -v python3 >/dev/null 2>&1; then
    record audit_bash skipped "python3 not found"
    return 0
  fi
  local opts=()
  if [[ -n $target ]]; then opts+=(--target "$target"); fi
  if [[ -n $platform ]]; then opts+=(--platform "$platform"); fi
  if python3 "$script_dir/audit_bash.py" ${opts[@]+"${opts[@]}"} -- "${files[@]}"; then
    record audit_bash pass "(bash scripts only; warnings above, if any, are review prompts)"
  else
    record audit_bash FAIL
  fi
}

main() {
  parse_args "$@"
  local f
  for f in "${files[@]}"; do
    [[ -r $f ]] || usage_error "cannot read: $f"
  done

  check_syntax
  check_shellcheck
  check_shfmt
  check_audit

  printf '\n%s\n' '---- summary ----'
  printf '%s\n' "${summary[@]}"
  return "$failed"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
