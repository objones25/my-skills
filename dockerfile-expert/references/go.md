# Go images

## Default: static binary on distroless (assets/Dockerfile.go)

- Builder runs on `$BUILDPLATFORM` and cross-compiles with `GOOS=$TARGETOS GOARCH=$TARGETARCH`,
  so multi-platform builds need no emulation (tested: arm64 image built on amd64 with no QEMU).
- `go mod download` with only go.mod/go.sum bind-mounted, cached at `/go/pkg/mod`.
- `go build` with the source bind-mounted and `/root/.cache/go-build` cached; the stage keeps
  only `/out/server`.
- `CGO_ENABLED=0` gives a static binary. `-trimpath` strips local paths (reproducibility);
  `-ldflags="-s -w"` drops symbol tables and DWARF (smaller; you lose symbolized panics in
  external debuggers, not Go stack traces).
- Runtime `gcr.io/distroless/static-debian13:nonroot`: CA certificates, tzdata, and a passwd
  entry for UID 65532, no shell.

Tested result (October 2026, Engine 29.9, buildx 0.38, architecture of this test not recorded; sizes and stop times vary by architecture and host): 16.6 MB image, healthz as UID 65532, graceful stop in 0.2 s (the server uses
`signal.NotifyContext` + `srv.Shutdown`).

## Gotchas

- **No go.sum** (module without dependencies): remove the go.sum bind mount, or the build
  fails because the source file does not exist.
- **Toolchain version**: the official image sets `GOTOOLCHAIN=local`. If go.mod says
  `go 1.27.2` and the image is older, the build fails rather than downloading a toolchain.
  Keep `GO_VERSION` at or above the go.mod line.
- **CGO needed** (sqlite3 via mattn, some crypto, cgo DNS resolver): build with
  `CGO_ENABLED=1` and ship on `gcr.io/distroless/base-debian13:nonroot` (glibc) built from a
  matching Debian release, or `cc-debian13` if it also needs libgcc/libstdc++.
- **scratch instead of distroless**: copy `/etc/ssl/certs/ca-certificates.crt` from the builder,
  either copy `/usr/share/zoneinfo` or build with `-tags timetzdata`, and set `USER 65532:65532`.
- **Version stamping**: pass `ARG VERSION` and `-ldflags="-s -w -X main.version=${VERSION}"`.
  `.git` is excluded by .dockerignore, so `debug.ReadBuildInfo` VCS fields are empty; that is
  expected.
- **Private modules**: `GOPRIVATE=github.com/org/*` plus a netrc build secret:
  `RUN --mount=type=secret,id=netrc,target=/root/.netrc go mod download`.
- **Health checks**: distroless has no curl. In Kubernetes use an httpGet probe. Under Compose,
  add a `healthcheck` subcommand to the binary itself (`ENTRYPOINT ["/server"]`,
  `HEALTHCHECK CMD ["/server", "healthcheck"]`).
