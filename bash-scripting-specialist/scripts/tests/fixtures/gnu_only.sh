#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s inherit_errexit
f=config.txt
sed -i 's/a/b/' "$f" # expect: B204
sed -i.bak 's/a/b/' "$f"
date -d yesterday # expect: B204
stat -c %s "$f" # expect: B204
grep -P '\d+' "$f" # expect: B204
find . -name '*.txt' -printf '%p\n' # expect: B204
timeout 5 sleep 1 # expect: B204
readlink -f "$f"
readlink -m "$f" # expect: B204
sed -i '' 's/a/b/' "$f" # expect: B204
find . -print0 | xargs -0 -r ls
sort -V "$f"
getopt -o ab: --long alpha -- "$@" # expect: B204
