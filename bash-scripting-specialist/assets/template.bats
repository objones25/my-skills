#!/usr/bin/env bats
# Tests for template.sh. Run: bats assets/template.bats
# Copy next to your script as test/<name>.bats and point SCRIPT at it.

bats_require_minimum_version 1.5.0

setup() {
  SCRIPT="${BATS_TEST_DIRNAME}/template.sh"
  cd "$BATS_TEST_TMPDIR" || return 1
  printf 'one\ntwo\n' >'input file.txt'
}

@test "--help prints usage to stdout and exits 0" {
  run -0 "$SCRIPT" --help
  [[ $output == *"Usage:"* ]]
}

@test "no arguments is a usage error (exit 2, message on stderr)" {
  run -2 --separate-stderr "$SCRIPT"
  [[ -z $output ]]
  [[ $stderr == *"at least one INPUT"* ]]
}

@test "unknown option is a usage error" {
  run -2 --separate-stderr "$SCRIPT" --bogus
  [[ $stderr == *"unknown option: --bogus"* ]]
}

@test "filename with a space is handled as one argument" {
  run -0 "$SCRIPT" 'input file.txt'
  [[ $output == *"2 input file.txt"* ]]
}

@test "--output=FILE and -o FILE both write the file" {
  run -0 "$SCRIPT" --output=a.txt 'input file.txt'
  run -0 "$SCRIPT" -o b.txt 'input file.txt'
  cmp a.txt b.txt
}

@test "dry run prints the command and changes nothing" {
  run -0 --separate-stderr "$SCRIPT" -n 'input file.txt'
  [[ -z $output ]]
  [[ $stderr == *"would run: wc -l --"* ]]
}

@test "missing input fails with exit 1" {
  run -1 --separate-stderr "$SCRIPT" nope.txt
  [[ $stderr == *"cannot read: nope.txt"* ]]
}

@test "arguments after -- are inputs even if they start with -" {
  run -1 --separate-stderr "$SCRIPT" -- -weird
  [[ $stderr == *"cannot read: -weird"* ]]
}

@test "temp directory is removed on exit" {
  TMPDIR=$BATS_TEST_TMPDIR run -0 "$SCRIPT" 'input file.txt'
  run ls -d "$BATS_TEST_TMPDIR"/template.sh.*
  [[ $status -ne 0 ]]
}

@test "functions can be tested by sourcing (main does not run)" {
  # Source in a child bash: the script's set -e and traps would otherwise leak into
  # the bats shell, and its function names could shadow bats helpers such as run.
  # shellcheck disable=SC2016 # $1 expands in the child bash
  run -0 --separate-stderr bash -c 'source "$1"; dry_run=1; run_cmd rm -rf /should-not-run' _ "$SCRIPT"
  [[ $stderr == *"would run: rm -rf /should-not-run"* ]]
}
