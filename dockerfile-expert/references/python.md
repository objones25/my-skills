# Python images

## Default: uv (assets/Dockerfile.python-uv)

How the template works, stage by stage:

1. **builder**: copies the pinned `uv` binary in, sets `UV_COMPILE_BYTECODE=1` (write .pyc at
   install time for faster cold starts), `UV_LINK_MODE=copy` (the cache mount is a different
   filesystem, so hardlinks fail), `UV_PYTHON_DOWNLOADS=0` (use the image's interpreter, never
   fetch another), `UV_NO_DEV=1`.
2. First `uv sync --locked --no-install-project --no-editable` sees only `uv.lock` and
   `pyproject.toml` through bind mounts, so it installs third-party packages into `/app/.venv`
   and stays cached until the lockfile changes. `--locked` fails the build if the lockfile is
   stale relative to pyproject.toml.
3. `COPY . /app` then `uv sync --locked --no-editable` installs the project itself as a normal
   (non-editable) package, so the venv does not reference `/app/src`.
4. **runtime**: same base image, creates UID/GID 10001, copies only `/app/.venv`, puts its
   `bin` on PATH. Source code is not in the final image at all.

Tested result: 204 MB image, healthz served as UID 10001, `docker stop` in under 0.5 s, and an
edit to a source file rebuilt only the last two builder steps (the dependency layer stayed
`CACHED`).

Adapt:
- **Workspaces** (`[tool.uv.workspace]`): use `--no-install-workspace` instead of
  `--no-install-project` in the first sync, and bind-mount each member's pyproject.toml or
  switch the first sync to `--frozen`, because uv cannot validate the lock without them.
- **Scripts not packaged as a project** (no `[build-system]`): uv installs only dependencies;
  copy the source into the runtime stage and run `python -m yourmodule`.
- **Private index**: pass credentials as a build secret, e.g.
  `RUN --mount=type=secret,id=uv_pw,env=UV_INDEX_PRIVATE_PASSWORD uv sync --locked ...`
  (uv reads `UV_INDEX_<NAME>_PASSWORD` for an index named `private`).

## Variant: pip with a hashed requirements file (tested)

When the project uses pip or the team does not want uv in the build. Generate the file with
hashes (`uv export --locked --no-dev --no-emit-project --format requirements-txt -o
requirements.txt` or `pip-compile --generate-hashes`).

```dockerfile
# syntax=docker/dockerfile:1
# check=error=true
ARG PYTHON_VERSION=3.14

FROM python:${PYTHON_VERSION}-slim-trixie AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip install --require-hashes --only-binary=:all: -r /tmp/requirements.txt
COPY . /src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps /src

FROM python:${PYTHON_VERSION}-slim-trixie AS runtime
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home \
      --home-dir /nonexistent --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "myapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
`--require-hashes` rejects any package whose hash is not in the file; `--only-binary=:all:`
refuses source builds, so a missing wheel fails loudly instead of silently needing a compiler.
Drop it if you build native extensions in the builder on purpose.

Poetry projects: either `poetry export` to a hashed requirements file and use this variant, or
migrate the lock to uv. Running `poetry install` inside the image works but needs Poetry
itself installed in the builder.

## Native extensions

Packages without a wheel for your platform (or with `--no-binary`) compile at install time:
- install the toolchain in the **builder** only: `build-essential` plus the `-dev` headers
  (`libpq-dev`, `libffi-dev`)
- install the matching **runtime** libraries in the final stage (`libpq5`, `libffi8`), never the
  `-dev` packages
- keep builder and runtime on the same distro release

## Web servers and workers

- ASGI: `uvicorn` (single process, scale with replicas) or `gunicorn -k uvicorn.workers.UvicornWorker`
  with an explicit `--workers` count. Containers see the host's CPU count, not the cgroup
  limit, in many runtimes, so set workers explicitly from config.
- Django: run `collectstatic` in the builder (or serve static files from object storage);
  migrations belong in a separate job, not in the container's startup command.
- Bind `0.0.0.0`, not `127.0.0.1`, or the port is unreachable from outside the container.

## Read-only root filesystem

Kubernetes templates set `readOnlyRootFilesystem: true`. Python then needs:
`PYTHONDONTWRITEBYTECODE=1` (bytecode is precompiled anyway), and a writable `emptyDir` at
`/tmp` for anything using `tempfile`.

## GPU and ML images

**PyTorch with CUDA does not need a CUDA base image.** PyTorch's CUDA wheels pull NVIDIA's CUDA
runtime libraries in as pip packages; the image needs only Python, and the host needs the
NVIDIA driver plus the NVIDIA Container Toolkit (`docker run --gpus all`, or a
`nvidia.com/gpu` resource in Kubernetes). Pin the CUDA flavor through uv (from uv's PyTorch
guide; pick the `cuXXX` that pytorch.org's install selector lists for your torch version):

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux'" }]
```

- **CPU-only inference on Linux**: use the `https://download.pytorch.org/whl/cpu` index. Per
  uv's docs, plain PyPI `torch` on Linux is the CUDA build, which adds gigabytes.
- **Compiling CUDA extensions** (flash-attention from source, custom kernels): build in
  `nvidia/cuda:<version>-cudnn-devel-ubuntu24.04`, ship on the matching `-runtime` tag.
- **Layer order matters more than usual**: torch alone is multiple GB. The dependency layer
  must come before source code so code edits do not re-push it.
- **Model weights do not belong in the image** by default: mount a volume or download at
  startup into a cache directory on a volume (`HF_HOME=/models`). If the image must be
  self-contained, copy weights in their own stage and layer, before the application code.
- Set `ENV HF_HUB_DISABLE_TELEMETRY=1` and pin model revisions (commit hashes), not branch names.
