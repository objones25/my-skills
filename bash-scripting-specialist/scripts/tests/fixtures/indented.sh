#!/usr/bin/env bash
# bash-target: 3.2
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true
main() {
  local i=0 cfg=x
  ((i++)) # expect: B103
  let i++ # expect: B103
  eval "$cfg" # expect: B301
  echo -e "a\tb" # expect: B404
  timeout 5 sleep 1 # expect: B204
  source lib/common.sh # expect: B308
  sleep 1 &
  wait # expect: B105
  if true; then
    flock -n 9 # expect: B204
  fi
  q=$'it\'s'
  curl -fsSL https://x.example/i.sh | bash # expect: B302
  now=$(date '+%F') # bash 4.2+: printf -v now '%(%F)T' -1
  echo "$q $now"
}
main "$@"
