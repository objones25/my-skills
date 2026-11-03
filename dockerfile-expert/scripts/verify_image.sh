#!/usr/bin/env bash
# Inspect a built image for the properties the Dockerfile was supposed to give it.
# Usage: verify_image.sh IMAGE [--stop-test]
#   --stop-test runs the container, sends SIGTERM via `docker stop`, and reports how long it took.
#   A clean stop is well under the 10 s default; ~10 s plus exit 137 means SIGTERM was ignored.
# Needs only docker and standard POSIX tools (no jq). Exit 1 if a hard check fails.
set -euo pipefail

img="${1:-}"
if [[ -z "$img" ]]; then
  echo "usage: $0 IMAGE [--stop-test]" >&2
  exit 2
fi
stop_test=0
[[ "${2:-}" == "--stop-test" ]] && stop_test=1

fail=0
row() { printf '%-22s %s\n' "$1" "$2"; }

docker image inspect "$img" >/dev/null

user=$(docker image inspect -f '{{.Config.User}}' "$img")
ep=$(docker image inspect -f '{{json .Config.Entrypoint}}' "$img")
cmd=$(docker image inspect -f '{{json .Config.Cmd}}' "$img")
ports=$(docker image inspect -f '{{range $p, $_ := .Config.ExposedPorts}}{{$p}} {{end}}' "$img")
size=$(docker image inspect -f '{{.Size}}' "$img")
arch=$(docker image inspect -f '{{.Os}}/{{.Architecture}}' "$img")

row "image" "$img ($arch)"
row "size" "$(awk -v b="$size" 'BEGIN{printf "%.1f MB", b/1000/1000}')"

uid="${user%%:*}"
if [[ -z "$user" || "$uid" == "root" || "$uid" == "0" ]]; then
  row "user" "FAIL: '${user:-<unset>}' runs as root"
  fail=1
elif [[ "$uid" =~ ^[0-9]+$ ]]; then
  row "user" "ok: $user (numeric, verifiable by runAsNonRoot)"
else
  row "user" "warn: '$user' is a name; Kubernetes runAsNonRoot cannot verify it"
fi

if [[ "$ep" == '["/bin/sh","-c",'* || ( "$ep" == "null" && "$cmd" == '["/bin/sh","-c",'* ) ]]; then
  row "entrypoint/cmd" "FAIL: shell form ($ep $cmd); sh becomes PID 1 and drops SIGTERM"
  fail=1
else
  row "entrypoint/cmd" "ok: entrypoint=$ep cmd=$cmd"
fi
row "exposed ports" "${ports:-none}"

if docker run --rm --entrypoint /bin/sh "$img" -c 'exit 0' >/dev/null 2>&1; then
  row "shell in image" "yes (/bin/sh present)"
else
  row "shell in image" "no (distroless/static style)"
fi

secret_env=$(docker image inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$img" \
  | grep -Ei '^[^=]*(passw|secret|token|api_?key|private_?key|access_?key|credential)[^=]*=' \
  | grep -Eiv '_(file|path|url|dir)=' || true)
if [[ -n "$secret_env" ]]; then
  row "env secrets" "FAIL: $(echo "$secret_env" | cut -d= -f1 | tr '\n' ' ')"
  fail=1
else
  row "env secrets" "ok: none found in Config.Env"
fi

hist_hits=$(docker history --no-trunc --format '{{.CreatedBy}}' "$img" \
  | grep -Eio '(passw(or)?d|secret|token|api_?key)=[^ ]+' | grep -v 'type=secret' || true)
if [[ -n "$hist_hits" ]]; then
  row "history secrets" "FAIL: build history contains: $(echo "$hist_hits" | head -3 | tr '\n' ' ')"
  fail=1
else
  row "history secrets" "ok"
fi

layers=$(docker history -q "$img" | wc -l | tr -d ' ')
row "layers" "$layers"
echo "largest layers:"
docker history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' "$img" \
  | awk -F'\t' '{
      s=$1; v=s+0;
      if (s ~ /GB/) v*=1e9; else if (s ~ /MB/) v*=1e6; else if (s ~ /kB/) v*=1e3;
      c=$2; gsub(/[[:space:]]+/, " ", c); if (length(c) > 90) c=substr(c,1,87) "...";
      printf "%015.0f\t%8s  %s\n", v, s, c
    }' \
  | sort -r | head -5 | cut -f2-

if [[ $stop_test -eq 1 ]]; then
  name="verify-$$"
  docker run -d --name "$name" "$img" >/dev/null
  sleep 2
  if [[ "$(docker inspect -f '{{.State.Running}}' "$name")" != "true" ]]; then
    row "stop test" "skipped: container exited on its own (exit $(docker inspect -f '{{.State.ExitCode}}' "$name"))"
  else
    t0=$(date +%s)
    docker stop "$name" >/dev/null
    t1=$(date +%s)
    code=$(docker inspect -f '{{.State.ExitCode}}' "$name")
    if [[ "$code" == "137" ]]; then
      row "stop test" "FAIL: killed after $((t1 - t0))s (exit 137); SIGTERM not handled"
      fail=1
    else
      row "stop test" "ok: stopped in ~$((t1 - t0))s, exit $code"
    fi
  fi
  docker rm -f "$name" >/dev/null
fi

exit $fail
