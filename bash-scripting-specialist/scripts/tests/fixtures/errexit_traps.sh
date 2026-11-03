#!/usr/bin/env bash
# bash-target: 4.4
set -euo pipefail # expect: B108
trap 'echo failed' ERR # expect: B102
count=0
((count++)) # expect: B103
let count++ # expect: B103
((++count))
((--count)) # expect: B103
if true; then ((count--)); fi # expect: B103
first=$(seq 1 100000 | head -n1) # expect: B104
if printf '%s\n' "$first" | grep -q 1; then echo found; fi # expect: B104
sleep 1 &
sleep 2 &
wait # expect: B105
is_ready() { # expect: B403
  [[ -f /etc/hostname ]] && echo ready # expect: B106
}
f() {
  local out=$(date) # expect: B109
  echo "$out"
}
tmp=$(mktemp) # expect: B107
is_ready
f
echo "$tmp"
