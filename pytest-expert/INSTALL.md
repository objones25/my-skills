# Installing pytest-expert

## As a Claude Code skill

Copy the folder to your user skills (every project) or a project's skills (shared with
the team through git):

```
cp -r pytest-expert ~/.claude/skills/
```

```
mkdir -p .claude/skills && cp -r pytest-expert .claude/skills/
```

Restart Claude Code. The skill loads on requests about writing tests, flaky suites,
pytest config, fuzzing, mutation testing, test CI, or hooks.

In claude.ai or Claude Cowork, upload the skill folder as a zip file. This repo does
not include a packaged `.skill` file.

### Updating an existing install

`cp -r` into an existing folder merges: files a new version removed stay behind.
Replace the folder instead of copying over it:

```
rm -rf ~/.claude/skills/pytest-expert
cp -r pytest-expert ~/.claude/skills/
```

The 2026-10-04 version removed the old ci-and-coverage reference and the old ci.yml asset.
If you copied over an older install, delete those two files.

## Using the tooling without Claude

Everything in `scripts/` is stdlib only. From the project root, with the skill folder
next to it:

1. Config. Merge `pytest-expert/assets/pytest-config.toml` into `pyproject.toml` by
   hand, then edit `testpaths`, `source`, `pythonpath`, and the coverage floor.

2. Conftest and scripts:

```
mkdir -p tests scripts
cp pytest-expert/assets/conftest.py tests/conftest.py
cp pytest-expert/scripts/audit_tests.py pytest-expert/scripts/mutate_check.py pytest-expert/scripts/flake_hunt.py pytest-expert/scripts/diff_guard.py scripts/
```

   Edit `APP_ENV_PREFIXES` in `tests/conftest.py` to your own environment variable
   prefixes.

3. Lint config:

```
cp pytest-expert/assets/ruff-pytest.toml ruff.toml
```

4. CI and git hooks:

```
mkdir -p .github/workflows
cp pytest-expert/assets/github/workflows/*.yml .github/workflows/
cp pytest-expert/assets/github/dependabot.yml .github/
cp pytest-expert/assets/.pre-commit-config.yaml .
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push
```

5. Claude Code hooks (optional; merge into an existing `.claude/settings.json` instead
   of overwriting it):

```
mkdir -p .claude/hooks
cp pytest-expert/assets/claude/hooks/*.py pytest-expert/scripts/audit_tests.py pytest-expert/scripts/diff_guard.py .claude/hooks/
cp pytest-expert/assets/claude/settings.json .claude/settings.json
```

6. Test dependencies. The workflows install `.[test]`, so declare it in
   `pyproject.toml`; `examples/wallet/pyproject.toml` has a complete set, including
   the optional `symbolic`, `fuzz`, and `mutation` extras.

7. Ignore tool output: add `mutants/`, `.hypothesis/`, `coverage.xml`, and `.coverage*`
   to `.gitignore`. Commit `fuzz/crashes/`; those files are regression tests.

## Checking the install

```
python scripts/audit_tests.py tests/
pytest
pytest -m integration --no-cov
```

To check the skill itself:

```
cd pytest-expert
pytest selftest -q
cd examples/wallet
pytest
```

## What is in the box

```
pytest-expert/
├── SKILL.md                         workflow, testing ladder, 34 hard gates
├── INSTALL.md                       this file
├── CHANGELOG.md                     what changed in each audit
├── AUDIT.md                         the November 2026 audit: gaps, corrections, how sources were checked
├── evals/evals.json                 prompts for re-testing the skill after edits
├── references/
│   ├── rules.md                     every rule, rationale, source, audit code index
│   ├── fixtures.md                  scopes, conftest layering, factories
│   ├── parametrize-and-doubles.md   parametrize patterns, test-double decision tree
│   ├── property-fuzz-symbolic.md    Hypothesis, CrossHair, Atheris, measured results
│   ├── security-testing.md          attack class to test pattern
│   ├── mutation-testing.md          mutate_check, mutmut, equivalent mutants
│   ├── integration-testing.md       real dependencies, containers, API contracts
│   ├── flaky-tests.md               causes, flake_hunt, no retries
│   ├── ci-cd.md                     hardened workflows, coverage, supply chain
│   ├── hooks.md                     pre-commit and Claude Code hooks
│   ├── ml-testing.md                models, tensors, training loops
│   ├── async-testing.md             pytest-asyncio, anyio, hangs, cancellation
│   ├── llm-written-tests.md         agent-written tests, diff guard codes, escalation
│   ├── sources.md                   research sources and how each was read
│   └── papers/                      16 paper digests with located claims
├── assets/                          configs, conftests, templates, workflows, hooks
├── scripts/                         audit_tests.py, mutate_check.py, flake_hunt.py, diff_guard.py
├── selftest/                        tests for the scripts, the hooks, and the skill's structure
├── examples/wallet/                 runnable worked example, plus demo/ with planted bugs
└── examples/diagnoses/              five worked diagnoses with real script output
```

## Verified against

pytest 9.2.0, pytest-cov 7.1.0, pytest-xdist 3.8.0, pytest-randomly 5.0.0,
pytest-asyncio 1.4.1, pytest-timeout 2.4.0, anyio 4.16.0, Hypothesis 6.170.14,
hypothesis-crosshair 0.0.31, CrossHair 0.0.112, Atheris 3.1.0, mutmut 3.8.1, syrupy
6.1.1, inline-snapshot 0.36.1, PyTorch 2.15.0 (CPU), ruff 0.17.3, zizmor 1.31.1,
actionlint 1.7.12, pre-commit 4.7.0, Python 3.13, November 2026. GitHub Actions and
pre-commit revs looked up with `git ls-remote` on 2026-11-02.
