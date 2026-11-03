# syntax=docker/dockerfile:1
# check=error=true

# Go service built as a static binary, shipped on distroless.
# Placeholders to adapt: GO_VERSION, the package path (./cmd/server), EXPOSE.
# If the module has no dependencies (no go.sum), delete the go.sum bind mount.
# Tested with Go 1.27 and gcr.io/distroless/static-debian13.

ARG GO_VERSION=1.27

# Build on the host's native platform and cross-compile for the target; this
# is much faster than emulating the target CPU for multi-platform builds.
FROM --platform=$BUILDPLATFORM golang:${GO_VERSION}-trixie AS build
WORKDIR /src

# Layer 1: download modules using only go.mod/go.sum.
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=bind,source=go.mod,target=go.mod \
    --mount=type=bind,source=go.sum,target=go.sum \
    go mod download

# Layer 2: compile. Source is bind-mounted, so nothing but the binary is kept.
ARG TARGETOS
ARG TARGETARCH
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=bind,target=. \
    CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \
    go build -trimpath -ldflags="-s -w" -o /out/server ./cmd/server

# ---------- runtime: no shell, no package manager, CA certs + tzdata included ----------
FROM gcr.io/distroless/static-debian13:nonroot
COPY --from=build /out/server /server
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/server"]
