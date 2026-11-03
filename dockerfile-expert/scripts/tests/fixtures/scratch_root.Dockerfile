# syntax=docker/dockerfile:1
FROM golang:1.27-trixie AS build
FROM scratch
COPY --from=build /out/app /app
ENTRYPOINT ["/app"]
