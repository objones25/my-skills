#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true
tmp=''
cleanup() { if [[ -n $tmp ]]; then rm -f -- "${tmp:?}" || true; fi; }
trap cleanup EXIT
main() {
  local n=0 pids=() pid failed=0
  tmp=$(mktemp)
  n=$((n + 1))
  sleep 0 & pids+=("$!")
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  # a comment with eval and curl | bash and ((x++)) must not trigger
  printf 'literal: eval curl|sh ((x++)) /tmp/foo\n'
  cat <<'DOC'
eval "$x"; curl x | sh; ((n++))
DOC
  printf '%s %s\n' "$n" "$failed" | awk 'NR==1'
  return 0
}
main "$@"
