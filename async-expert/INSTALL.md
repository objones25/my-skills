# Installing this skill in a project

## Requirements

- Python **3.11+** (`asyncio.TaskGroup`, `asyncio.timeout`, `ExceptionGroup`). 3.14 recommended:
  several rules exist because 3.14 changed the defaults.
- An HTTP client for the `aioutil` assets. **New projects: `httpx2`**, the Pydantic-maintained
  fork of httpx 0.28.1, identical API, actively released, and what Starlette's `TestClient` now
  expects. Existing httpx codebases work as-is; see `references/httpx.md`.
- `ruff` for the lint gate. `scripts/audit_async.py` needs nothing beyond the standard library.

New projects:

```bash
uv add httpx2
uv add --dev ruff mypy pytest
```

Existing httpx codebases: pin it so a pre-release resolver cannot pull in `1.0.devN`.

```bash
uv add "httpx==0.28.1"
uv add --dev ruff mypy pytest
```

Optional profiling and blocking-call detection:

```bash
uv add --dev scalene py-spy blockbuster
```

`assets/aioutil/` imports `httpx`. Under httpx2, call `httpx2.alias_httpx()` at the very top of
your entry point (before anything imports httpx, or it raises `RuntimeError`), or rewrite the
import when you copy the package in. This works with both GNU and BSD/macOS tools:

```bash
perl -pi -e 's/^import httpx$/import httpx2 as httpx/' aioutil/*.py
```

## Wiring it in

**1. Copy the audit script.**

```bash
mkdir -p scripts && cp SKILL_DIR/scripts/audit_async.py scripts/
python scripts/audit_async.py src/
```

`SKILL_DIR` is wherever this skill is installed.

**2. Copy the helpers you want.** `assets/aioutil/` is three small modules with no dependency on
each other beyond `httpx`. Drop the package into your source tree and import from it, or copy the
two or three functions you need. It is meant to be read and adapted, not vendored blind.

**3. Add the lint config.** Either point ruff at `assets/ruff-async.toml` directly, or merge its
`select` list into your existing `pyproject.toml`:

```toml
[tool.ruff.lint]
extend-select = ["ASYNC", "RUF006"]
```

**4. Add the CI job.** `assets/ci.yml` runs audit, then lint, then tests. Copy it to
`.github/workflows/async-checks.yml` and adjust the paths. For blocking-call detection in tests,
add the blockbuster autouse fixture from `references/debugging.md` to `conftest.py`.

**5. Add a pre-commit hook** so violations never reach the shared branch:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: audit-async
        name: audit async patterns
        entry: python scripts/audit_async.py
        language: system
        types: [python]
        pass_filenames: true
```

## Adopting it in an existing codebase

Running the audit over a large async codebase for the first time usually produces a lot. Do not
try to fix it all at once.

1. Run `python scripts/audit_async.py src/ --format json` and count by code.
2. Fix the `BLOCKING_*` findings first. Those are the ones costing real throughput.
3. Then `SWALLOWED_CANCEL` and `FLOATING_TASK`, the ones causing hangs and vanishing work.
4. Then `UNBOUNDED_FANOUT` and `NO_CLIENT_TIMEOUT`, the ones that only hurt under load, which is
   also when they hurt most.
5. Add `--ignore SEQUENTIAL_AWAIT` while you work through the rest; it is the most heuristic check
   and a sequential await is legitimate when the iterations genuinely depend on each other.

Ratchet: once a code is at zero, remove it from `--ignore` so it stays there.

## Verifying the audit still works

`assets/examples/bad_example.py` is the fixture: one finding per numbered comment.

```bash
python scripts/audit_async.py assets/examples/bad_example.py
python scripts/audit_async.py assets/aioutil assets/examples/good_pipeline.py
python scripts/test_audit_async.py
uv run --with httpx python scripts/test_aioutil.py
```

The first ends with `19 finding(s): ASYNC_GEN_YIELD_IN_CM=1, BARE_GATHER=1, BLOCKING_HTTP=2, ...`
and exits 1; the second prints `audit_async: no findings` and exits 0; both test files end `OK`.

If you add a check, add its trigger to `bad_example.py` and a case to `test_audit_async.py` in the
same commit.
