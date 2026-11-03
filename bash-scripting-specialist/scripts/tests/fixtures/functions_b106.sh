#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s inherit_errexit
is_ready() {
  local f=$1
  [[ -f $f && -r $f ]]
}
say() {
  echo "a && b"
}
g() { [[ -f $1 ]] && echo yes; } # expect: B106
h() { [[ -f $1 ]] || return 0; echo yes; }
k() {
  [[ -n ${1-} ]] && printf '%s\n' "$1" # expect: B106
}
main() {
  is_ready /etc/hostname || true
  say
  g /etc/hostname
  h /etc/hostname
  k x
}
main "$@"
