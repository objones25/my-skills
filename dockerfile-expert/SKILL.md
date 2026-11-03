---
name: dockerfile-expert
description: Write, review, and fix Dockerfiles and .dockerignore files to production best practices (multi-stage builds, minimal pinned base images, non-root numeric users, BuildKit cache and secret mounts, exec-form entrypoints, small reproducible images). Ships build-tested templates for Python (uv), Node/TypeScript, and Go, a linter that catches what docker build --check and hadolint miss, and an image verifier. Use this whenever the user asks to create, containerize, dockerize, optimize, shrink, harden, speed up, debug, or review a Dockerfile, Containerfile, container image, or docker build, or mentions BuildKit, buildx, image size, slow builds, or "make this run in a container", even if they never say "best practices". For Kubernetes manifests use kubernetes-expert.
---

# Dockerfile expert

Goal: every Dockerfile you write builds fast from cache, produces a small image that runs as a
non-root numeric UID, stops cleanly on SIGTERM, contains no secrets or build tools, and can be
rebuilt reproducibly. Verify those properties by running tools, not by inspection alone.

## Workflow

1. **Read the project first.** Find, before writing anything:
   - runtime and its version source (`requires-python` in pyproject.toml, `engines`/`.nvmrc`,
     the `go` line in go.mod; the base image must be at least that version)
   - the lockfile (uv.lock, poetry.lock, requirements with hashes, package-lock.json,
     pnpm-lock.yaml, yarn.lock, go.sum). If there is none, say so: installs will not be
     reproducible, and offer to generate one.
   - entry point, listening port, whether it writes to disk, native or GPU dependencies
   - where it will run (Kubernetes, Compose, Cloud Run/ECS, local dev) and which CPU
     architectures (an Apple Silicon laptop plus amd64 servers means multi-platform)
   - any existing Dockerfile, compose file, or CI workflow to stay consistent with
2. **Pick base images** with `references/base-images.md`. Default: Debian slim of the current
   stable release for interpreted languages, distroless static for static Go binaries.
3. **Start from the matching template** in `assets/` (`Dockerfile.python-uv`,
   `Dockerfile.node`, `Dockerfile.go`) and adapt the placeholders. Each was built, run, and
   stop-tested. Language specifics and variants (pip, pnpm, CGO, GPU) are in
   `references/python.md`, `references/node.md`, `references/go.md`.
4. **Write `.dockerignore`** from `assets/dockerignore.<lang>`. Without it, `COPY . .`
   sends `.git`, `.env`, and virtualenvs into the build and can bake them into the image.
5. **Validate**, running whatever is available and showing the real output:
   ```
   python3 scripts/lint_dockerfile.py Dockerfile
   docker build --check .
   hadolint Dockerfile
   docker build -t app:dev .
   scripts/verify_image.sh app:dev --stop-test
   ```
   If Docker is not available, run the linter and say the build was not tested.
6. **Deliver** the Dockerfile, the .dockerignore, and a short decisions note: base image and
   why, what is pinned and how to bump it, the UID, the port, the exact build command
   (including `--secret` and `--platform` flags if used), and anything not verified.

When reviewing an existing Dockerfile, run the linter and `docker build --check`, then walk
`references/review-checklist.md`. Report findings by impact (security, then correctness, then
build speed, then size) with an exact diff for each.

## Rules and why they matter

**Structure**
- First lines: `# syntax=docker/dockerfile:1` (current stable frontend, needed for heredocs,
  `--mount`, `--exclude`) and `# check=error=true` (build checks fail the build instead of warning).
- Multi-stage. Compile, install dev dependencies, and run bundlers in builder stages; the final
  stage receives only runtime artifacts via `COPY --from=`. A final stage based on a full SDK
  image (golang, maven, full python) ships compilers and hundreds of MB of attack surface.
- Order layers from least to most frequently changing: system packages, then dependency
  install from lockfiles only, then source. Bind-mount the lockfiles for the install step
  (`--mount=type=bind,source=uv.lock,target=uv.lock`) so a source edit never reinstalls
  dependencies.
- Use BuildKit cache mounts for package-manager caches instead of deleting caches inside the
  layer. Paths per ecosystem are in `references/buildkit.md`.

**Reproducibility**
- Pin base images to at least a minor-version tag and a named distro release
  (`python:3.14-slim-trixie`, not `python:3` or `latest`). For fully reproducible builds add
  the index digest (`name:tag@sha256:...`) and let Dependabot or Renovate open bump PRs; a
  digest alone without automated bumps means silently missing security fixes.
- Pin tools copied from images (`COPY --from=ghcr.io/astral-sh/uv:0.13.3`), never `:latest`.
- Install from the lockfile in frozen mode: `uv sync --locked`, `npm ci`,
  `pnpm install --frozen-lockfile`, `go mod download` with go.sum present.

**Security**
- The final stage runs as a non-root user with a **numeric** UID:GID (`USER 10001:10001`).
  Kubernetes `runAsNonRoot` can only verify numeric UIDs; a name like `USER app` makes the pod
  fail to start unless `runAsUser` is also set.
- Leave application files owned by root (no `--chown` on code). The app user can read and
  execute them but cannot modify them. Create and chown only the directories it must write to.
- Never put secrets in `ARG`, `ENV`, or a `COPY`ed file: all three persist in the image or its
  history. Use `RUN --mount=type=secret,id=name,env=VAR` (or `target=/run/secrets/name`) at
  build time and inject runtime secrets from the orchestrator.
- No `curl | sh`. Download with `ADD --checksum=sha256:...` or verify a checksum yourself.
- No `sudo`, no `chmod 777`, no package manager or shell in the final image unless needed.

**Runtime behavior**
- Exec-form `CMD`/`ENTRYPOINT` (`["uvicorn", "app:app"]`). Shell form makes `/bin/sh -c`
  PID 1, which does not forward SIGTERM: in testing (October 2026, Engine 29.9, architecture not recorded), a shell-form image was SIGKILLed after
  the 10 s timeout (exit 137) while the exec-form templates stopped in under 0.5 s.
- Start the runtime directly, not through `npm start` or `yarn start`.
- The process must handle SIGTERM. The kernel does not apply default signal actions to PID 1,
  so a process without its own SIGTERM handler ignores it. Tested (same environment): a plain `python -c` script
  and a plain `node -e` script as PID 1 were both SIGKILLed after 10 s (exit 137); with
  `docker run --init` the same Node process exited on SIGTERM immediately (exit 143). Servers
  like uvicorn and gunicorn, and Go code using `signal.NotifyContext`, handle it themselves;
  plain scripts and plain Node servers do not. If the app cannot be changed, add an init
  (`tini` in the image, or `docker run --init`).
- Log to stdout/stderr (`PYTHONUNBUFFERED=1`). One concern per container. Configuration comes
  from environment at run time, not from per-environment images.
- `HEALTHCHECK` is used by Docker, Compose, and Swarm only; Kubernetes ignores it and uses
  probes. Add one only when the image runs outside Kubernetes and has a way to run the check
  (distroless images have no curl or shell).

**Size**
- Slim or distroless final stages, `--no-install-recommends`, `rm -rf /var/lib/apt/lists/*`
  (or apt cache mounts), runtime libraries only (`libpq5`, not `libpq-dev`).
- Check what actually takes space with `scripts/verify_image.sh` (largest layers) before
  optimizing anything else.

## Gotchas confirmed by testing (October 2026, Engine 29.9, buildx 0.38)

The multi-platform test ran on an amd64 host (`references/buildkit.md`); the architecture of the other tests was not recorded. Image sizes, stop times and exit codes in this skill depend on architecture and host. On 2026-11-03 (macOS arm64) only the linter tests were rerun (3 passed); no image was built.

- `ADD https://...` creates the file with mode 600, unreadable by a non-root user. Add
  `--chmod=0644` (or `0755` for executables).
- `COPY . .` with no `.dockerignore` copied a token file from the context into the image.
- A uv virtualenv must run on the same base image it was built on: its `python` is a symlink
  to the builder's interpreter path.
- Cross-compiling (Go with `FROM --platform=$BUILDPLATFORM` plus `TARGETOS`/`TARGETARCH`)
  builds other architectures with no emulation only if the final stage has no `RUN`. A `RUN`
  in a target-platform stage needs QEMU or a native builder for that architecture.
- The official golang image sets `GOTOOLCHAIN=local`, so a go.mod whose `go` line is newer than
  the image's Go fails instead of downloading a toolchain. Match the image to go.mod.

## Files

- `assets/Dockerfile.python-uv`, `assets/Dockerfile.node`, `assets/Dockerfile.go`: tested templates
- `assets/dockerignore.{python,node,go}`: starting .dockerignore files
- `references/base-images.md`: choosing and pinning base images (slim, Alpine, distroless,
  Docker Hardened Images, scratch)
- `references/buildkit.md`: cache, secret, ssh, and bind mounts; heredocs; COPY flags;
  multi-platform; build checks; attestations
- `references/python.md`, `references/node.md`, `references/go.md`: language specifics,
  including GPU/ML images in python.md
- `references/review-checklist.md`: ordered checklist for reviewing an existing Dockerfile
- `scripts/lint_dockerfile.py`: dependency-free linter, `--rules` lists rules, `--json` for CI,
  `--strict` fails on warnings. Tests: `python3 scripts/tests/test_lint_dockerfile.py`
- `scripts/verify_image.sh IMAGE [--stop-test]`: checks the built image (user, entrypoint
  form, shell presence, env/history secrets, largest layers, SIGTERM handling)

Versions in templates were current when written (Python 3.14, Node 24 LTS, Go 1.27, Debian 13
"trixie", uv 0.13.3). Before using them, check the current releases and adjust.
