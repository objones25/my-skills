# syntax=docker/dockerfile:1
FROM golang:1.27-trixie
WORKDIR /src
COPY . .
RUN go mod download && go build -o /app .
USER nobody
ENTRYPOINT ["/app"]
