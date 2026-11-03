# syntax=docker/dockerfile:1
# check=error=true
ARG BASE_TAG=3.14-slim-trixie
FROM python:${BASE_TAG}@sha256:0000000000000000000000000000000000000000000000000000000000000000 AS base
RUN groupadd --system --gid 10001 app \
    # a comment inside a continuation is skipped by the parser
 && useradd --system --uid 10001 --gid app app
USER 10001:10001

FROM base AS builder
USER 0
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev
RUN <<EOT
#!/usr/bin/env bash
set -euo pipefail
echo "heredoc body with a | pipe" | tr a-z A-Z
EOT

FROM base
COPY --from=builder /usr/lib /usr/lib
HEALTHCHECK CMD ["python", "-c", "print(1)"]
CMD ["python", "-m", "http.server"]
