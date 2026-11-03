# Reviewing an existing Dockerfile

Run first and quote the real output:
```
python3 scripts/lint_dockerfile.py --context . Dockerfile
docker build --check .
hadolint Dockerfile
```
Then build it and run `scripts/verify_image.sh IMAGE --stop-test` if Docker is available.
Report findings in this order, each with the exact change.

## 1. Security
- [ ] Final stage `USER` is a non-root numeric UID:GID
- [ ] No secrets in `ARG`, `ENV`, copied files, or build history (`docker history --no-trunc`)
- [ ] `.dockerignore` excludes `.env*`, keys, `.git`, credentials files
- [ ] Build-time credentials use `--mount=type=secret` or `--mount=type=ssh`
- [ ] No `curl | sh`; downloads verified with `ADD --checksum` or an explicit checksum
- [ ] No `sudo`, `chmod 777`, or setuid tricks
- [ ] Final image has no compiler, package-manager caches, or `-dev` packages
- [ ] Base image is from a trusted publisher and currently maintained

## 2. Correctness at runtime
- [ ] Exec-form `CMD`/`ENTRYPOINT`, process started directly (not via npm/yarn/sh)
- [ ] App handles SIGTERM, or the container uses an init
- [ ] Listens on `0.0.0.0`, port matches `EXPOSE` and the orchestrator config
- [ ] Logs go to stdout/stderr
- [ ] Writes only to declared writable paths (works with a read-only root filesystem)
- [ ] Runtime libraries present for every native extension (no missing `.so` at start)
- [ ] Same distro release in builder and runtime when copying venvs, `node_modules`, or
      dynamically linked binaries

## 3. Reproducibility
- [ ] Base images pinned to version + distro release; digest pinning with automated bumps if
      required
- [ ] Tools copied from images pinned to exact versions
- [ ] Dependencies installed from a lockfile in frozen mode
- [ ] `# syntax=docker/dockerfile:1` present

## 4. Build speed
- [ ] Dependency install happens before `COPY` of source and sees only lockfiles
- [ ] Package-manager caches on cache mounts
- [ ] Context is small (`.dockerignore` covers `node_modules`, `.venv`, data, build outputs)
- [ ] Independent stages can run in parallel (BuildKit builds unrelated stages concurrently)

## 5. Size
- [ ] Multi-stage, final base is slim/distroless
- [ ] `--no-install-recommends`, apt lists removed or on cache mounts
- [ ] Largest layers checked with `verify_image.sh` or `docker history`; nothing large that is
      deleted in a later layer (deleting in a later layer does not shrink the image)
