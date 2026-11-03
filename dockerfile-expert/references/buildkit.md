# BuildKit features worth using

All examples below were built with buildx 0.38 / Engine 29.9 using `# syntax=docker/dockerfile:1`.

## Cache mounts

A cache mount persists a directory across builds on the same builder without putting it in
the image. Use it instead of `--no-cache-dir` + reinstalling everything.

| Ecosystem | Mount target | Notes |
|---|---|---|
| uv | `/root/.cache/uv` | Set `UV_LINK_MODE=copy` (cache and target are different filesystems) |
| pip | `/root/.cache/pip` | Drop `--no-cache-dir` when using the mount |
| npm | `/root/.npm` | |
| pnpm | whatever `pnpm store path` prints in the builder | or set `PNPM_HOME` and mount a fixed store path |
| yarn (berry) | `/root/.yarn/berry/cache` | |
| Go modules | `/go/pkg/mod` | official golang image's GOPATH |
| Go build cache | `/root/.cache/go-build` | |
| Cargo | `/usr/local/cargo/registry` and `/app/target` | copy the binary out of target in the same RUN |
| apt | `/var/cache/apt` and `/var/lib/apt`, both `sharing=locked` | first remove Debian's auto-clean hook, see below |

apt with cache mounts (verified):
```dockerfile
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
 && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends jq
```
Paths are relative to the user running the step; if a build stage runs as non-root, mount at
that user's cache directory instead of `/root/...`.

Cache mounts live on the builder. In CI with ephemeral runners they start empty unless you use
a persistent builder; registry or GitHub Actions cache backends (`--cache-to/--cache-from`)
cache layers, not cache mounts.

## Bind mounts

`--mount=type=bind,source=X,target=X` exposes a context file to one RUN without creating a
layer. Use it for lockfiles in the dependency step and for whole-source compilation when only
the output should persist (`--mount=type=bind,target=.` in the Go template).

## Secrets

```dockerfile
RUN --mount=type=secret,id=api_token,env=API_TOKEN \
    some-command-that-reads "$API_TOKEN"
```
```
docker build --secret id=api_token,src=token.txt .
docker build --secret id=api_token,env=API_TOKEN .
```
Verified: the value is visible during that RUN only, absent from later steps' environment, and
absent from `docker history`. Without `env=`, it is a file at `/run/secrets/<id>`. Typical
uses: private package indexes (`UV_INDEX_..._PASSWORD`, `NPM_TOKEN`, `.netrc`), private Go
modules (`GOPRIVATE` plus a netrc secret).

For private git over SSH: `RUN --mount=type=ssh git clone git@github.com:org/repo.git`
with `docker build --ssh default .`.

## Heredocs

```dockerfile
RUN <<EOT
#!/usr/bin/env bash
set -euo pipefail
step one
step two
EOT
```
Without a shebang the body runs under `/bin/sh` (dash on Debian, which may not support
`pipefail`).

## COPY and ADD flags

- `COPY --chmod=0755 src dst` sets mode without an extra layer.
- `COPY --link` makes the layer independent of earlier layers, so rebases and cache reuse are
  cheaper. Safe when the destination does not depend on earlier filesystem state (a fresh
  directory, a binary into distroless).
- `COPY --exclude=*.md . /dst/` filters at copy time (verified with the current frontend).
- `ADD --checksum=sha256:<hex> https://... /dst` downloads and verifies. Remote files arrive
  with mode **600** (verified), so add `--chmod`.
- Prefer `COPY` for local files. `ADD` auto-extracts local tarballs, which surprises readers.

## Multi-platform builds

```
docker buildx build --platform linux/amd64,linux/arm64 -t registry/app:1.2.3 --push .
```
- Stages declared `FROM --platform=$BUILDPLATFORM` run natively and should cross-compile using
  `ARG TARGETOS` and `ARG TARGETARCH` (Go, Rust, Node bundlers).
- Any `RUN` in a stage for the target platform needs QEMU (`docker run --privileged --rm
  tonistiigi/binfmt --install all`) or a native builder node. Verified (October 2026, Engine 29.9, buildx 0.38): the Go template built
  for arm64 on an amd64 host with no QEMU because its final stage has no RUN.
- Loading a multi-platform image locally requires the containerd image store (it was on by
  default in the Engine 29.9 used for testing) or `--push` to a registry.

## Build checks

`docker build --check .` runs BuildKit's checks without building and exits non-zero on
violations. `# check=error=true` makes a normal build fail on them; `# check=skip=RuleA,RuleB`
skips rules. The current rule list is at docs.docker.com/reference/build-checks.

## Provenance and SBOM

`docker buildx build --provenance=mode=max --sbom=true --push ...` attaches SLSA provenance and
an SBOM as attestations. Useful when the registry or policy tooling consumes them.
