# Choosing and pinning base images

## Decision guide

| Final-stage base | Use when | Watch out for |
|---|---|---|
| Debian slim of the current stable release (`python:3.14-slim-trixie`, `node:24-trixie-slim`) | Default for Python, Node, Ruby, anything needing glibc and the occasional apt package | Has a shell and apt; add only runtime libraries |
| Distroless static (`gcr.io/distroless/static-debian13:nonroot`) | Static binaries: Go with `CGO_ENABLED=0`, Rust with musl | No shell, no libc. Includes CA certs, tzdata, `/etc/passwd`. `:nonroot` = UID 65532 |
| Distroless base or cc (`base-debian13`, `cc-debian13`) | Dynamically linked binaries (CGO, Rust/C++ with glibc; cc adds libgcc/libstdc++) | Same no-shell constraints |
| Distroless language images (`python3-debian13`, `nodejs24-debian13`) | Very small Python/Node runtimes when Debian's interpreter version is acceptable | `python3-debian13` ships Debian's Python (3.13.5 when checked), not the latest; a venv built on `python:3.14-slim` will not run on it |
| Docker Hardened Images (`dhi.io/<image>:<tag>`) | Teams that want vendor-maintained minimal images with SBOMs, provenance, and low CVE counts | Requires `docker login dhi.io` (free Docker account). Runtime variants have no shell or package manager and run non-root by default; build stages use `-dev` tags. See below |
| Alpine (`*-alpine3.23`) | Tiny images where every dependency supports musl | musl, not glibc: some Python wheels and native Node modules fall back to source builds or misbehave; never build on Alpine and run on Debian or the reverse |
| `scratch` | A single static binary with nothing else | You must supply CA certs, tzdata, and a numeric `USER`; distroless static already does |
| Ubuntu LTS (`ubuntu:24.04`) | You need Ubuntu packages or vendor images based on it (CUDA) | Larger than Debian slim |

Defaults: Debian slim for interpreted languages, distroless static for static Go. Use the
same distro release in builder and runtime stages so shared libraries and interpreter paths
match.

## Docker Hardened Images

Docker made its hardened image catalog free and open source under Apache 2.0 (Docker press
release and product page). Usage per the DHI docs:

- Registry `dhi.io`, e.g. `FROM dhi.io/python:3.13`; log in with `docker login dhi.io`.
  Kubernetes needs an image pull secret for `dhi.io`.
- Runtime variants: no shell or package manager, non-root by default. Build stages use `-dev`
  tags; compiled binaries go on `dhi.io/static`.
- Paid tiers add FIPS/STIG variants and extended support for end-of-life versions.

Docker's own figure is "up to 95% fewer vulnerabilities" than community images; that is a
vendor claim. Before switching, scan both with the same scanner (`docker scout cves`, Trivy,
or Grype) and compare. Chainguard sells a similar catalog; check which tags its free tier
covers before depending on it.

## Tags and digests

A tag such as `python:3.14-slim-trixie` is mutable: the publisher rebuilds it with security
patches. That is good (you get fixes on rebuild) and bad (two builds a month apart differ).

- Minimum: pin runtime minor version and distro release in the tag. Never `latest`, never a
  bare major like `node:24` if the distro could change under you.
- Reproducible: append the multi-arch index digest. Get it with
  ```
  docker buildx imagetools inspect python:3.14-slim-trixie --format '{{json .Manifest.Digest}}'
  ```
  then write `FROM python:3.14-slim-trixie@sha256:<digest>`. Keep the tag for readability; the
  digest wins.
- Pair digests with automated bumps: Dependabot (`package-ecosystem: "docker"`) or Renovate
  opens PRs when the tag moves. Docker's best-practices page recommends this over floating
  tags because you keep both the fixes and an audit trail.
- Rebuild on a schedule even without code changes (`docker build --pull`) so OS patches land.

## Checking an image before choosing it

```
docker buildx imagetools inspect IMAGE
docker scout cves IMAGE
docker run --rm --entrypoint sh IMAGE -c 'id; cat /etc/os-release'
```
The last command fails on distroless, which itself answers "does it have a shell".
