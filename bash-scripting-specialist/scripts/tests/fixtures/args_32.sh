#!/usr/bin/env bash
# bash-target: 3.2
set -Eeuo pipefail
args=("$@")
for a in "${args[@]}"; do echo "$a"; done # expect: B202
echo "count ${#args[@]} [${@}]" # expect: B202
for a in "$@"; do echo "$a"; done
trap 'echo bye' EXIT
