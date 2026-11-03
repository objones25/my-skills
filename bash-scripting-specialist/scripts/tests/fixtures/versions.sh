#!/bin/bash
# expect-file: B203
# bash-target: 3.2
set -eu
shopt -s inherit_errexit 2>/dev/null || true
trap 'rm -f -- "${t:?}"' EXIT
declare -A seen=() # expect: B201
mapfile -t lines < /etc/hosts # expect: B201
name=World
echo "${name,,}" # expect: B201
printf '%(%F)T\n' -1 # expect: B201
exec {fd}>/dev/null # expect: B201
[[ -v name ]] && echo set # expect: B201
declare -n ref=name # expect: B201
sleep 0 & wait -n # expect: B201
echo "${name@Q}" # expect: B201
echo "$EPOCHSECONDS" # expect: B201
echo "${ date; }" # expect: B201
for l in "${lines[@]}"; do echo "$l"; done # expect: B202
for l in ${lines[@]+"${lines[@]}"}; do echo "$l"; done
echo "$fd ${seen[*]} $ref"
r='X&Y'
echo "${name/o/$r}" # expect: B205
echo "${name//o/ & }" # expect: B205
echo "${name/o/\&}"
echo "${name/o/0}"
