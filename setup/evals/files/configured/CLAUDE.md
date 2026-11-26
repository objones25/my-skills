# notes-cli

Personal notes CLI in Go. Single binary, no network access.

## Commands
- Test: `go test ./...`; one package: `go test ./internal/store`
- Lint: `golangci-lint run`

## Done means
- `go test ./...` and `golangci-lint run` both pass.

## Gotchas
- `internal/legacy/` is still used by the import command; do not delete or copy its patterns.
