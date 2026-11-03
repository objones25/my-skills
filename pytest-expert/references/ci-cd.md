# CI/CD, coverage, and supply chain

Making the gates stick. A rule enforced by a person is enforced sometimes.

## Contents

- [The pipeline](#the-pipeline)
- [Workflow hardening](#workflow-hardening)
- [Verifying the workflows](#verifying-the-workflows)
- [Keeping pins current](#keeping-pins-current)
- [Coverage](#coverage)
- [Speed](#speed)
- [Supply chain](#supply-chain)
- [Debugging a failure](#debugging-a-failure)
- [Sources](#sources)

## The pipeline

```
every push / PR   gates (audit, ruff) ──► tests (3.11 to 3.14, random order, -n auto) ──► integration
nightly           hypothesis (5,000 ex) │ crosshair │ atheris (5 min/target) │ mutmut (gated) │ flake hunt
push + weekly     pip-audit │ ruff S │ zizmor │ gitleaks (full history) │ dependency review (PRs)
weekly            dependabot: actions and pip, 7-day cooldown
```

| File (copy to `.github/`) | Purpose |
|---|---|
| `assets/github/workflows/tests.yml` | merge gate |
| `assets/github/workflows/nightly.yml` | deep testing too slow for every push |
| `assets/github/workflows/security.yml` | dependencies, SAST, workflow audit, secrets |
| `assets/github/dependabot.yml` | keeps SHA pins and dependencies current |

The audit runs before pytest on purpose: a suite of assertion-free tests passes
pytest and fails the audit, and that is the right outcome.

## Workflow hardening

| Setting | Why |
|---|---|
| `permissions: {}` at the top, `contents: read` per job | the default `GITHUB_TOKEN` can be broader than a test job needs |
| Actions pinned to a full commit SHA with a `# vX.Y.Z` comment | GitHub's guidance: pinning to a full-length SHA is the only way to use an action as an immutable release. A tag can be moved, This is how the tj-actions compromise reached CI workflows. |
| `persist-credentials: false` on checkout | otherwise the token stays in `.git/config` for every later step |
| No `${{ github.event.* }}` inside `run:` | attacker-controlled text (PR titles, branch names) becomes shell code; pass it through `env:` |
| No `pull_request_target` for test jobs | it runs with secrets in the context of untrusted PR code |
| `concurrency` with `cancel-in-progress` | superseded runs stop instead of queueing |
| `timeout-minutes` on every job | a hung test cannot burn six hours |
| `fail-fast: false` in the matrix | you learn whether one Python version failed or all of them |
| `PYTHONHASHSEED: "0"` in workflow `env` | set before the interpreter starts, so it actually applies |

The original skill's `ci.yml` had no `permissions`, tag-pinned actions, and default
credential persistence. All three are fixed in the new workflows.

## Verifying the workflows

Both checkers were run on the shipped files:

```
actionlint: clean
No findings to report. Good job!
```

The second line is zizmor 1.31.1 with `--persona=auditor`, its strictest setting.
The only finding it raised before that was missing `concurrency` on two workflows,
now added. The same two tools run in pre-commit and in `security.yml`.

Point zizmor at the files, or at a repository root. Given a folder that is not inside
`.github/workflows` (such as this skill's `assets/github/workflows/`), zizmor 1.31.1
collects nothing and exits with `fatal: no audit was performed ... no inputs collected`.
That one fails loudly; a wrapper script that ignores the exit code would report a clean
run that audited nothing.

## Keeping pins current

Dependabot updates the SHA and the version comment together; `cooldown:
default-days: 7` waits a week before proposing a new release, which avoids pulling
in a compromised version during its first days. To look up a SHA by hand without the
API (which rate-limits anonymous use):

```
git ls-remote https://github.com/actions/checkout.git refs/tags/v7.0.1
```

For an annotated tag, use the `refs/tags/vX^{}` line, which is the commit.

## Coverage

Config is in `assets/pytest-config.toml`.

| Practice | Why |
|---|---|
| Branch coverage (`--cov-branch`, `branch = true`) | `if x:` counts only when both outcomes ran |
| Floor at today's number, then ratchet up | a floor you lower to merge teaches that the number is negotiable |
| `parallel = true` | required for pytest-xdist workers to merge data |
| `exclude_also`, not `exclude_lines` | `exclude_lines` replaces coverage's defaults, including `pragma: no cover` |
| Read the report, do not chase 100% | coverage shows what is untested, not whether tested code is tested well; see `mutation-testing.md` |

## Speed

| Lever | Command |
|---|---|
| Deselect expensive layers by default | `-m "not slow and not integration and not e2e and not gpu and not symbolic"` in `addopts` |
| Parallelism | `pytest -n auto`; `--dist loadfile` keeps a file's tests on one worker |
| Find the slow tests | `pytest --durations=10` |
| Hang detection | built-in `faulthandler_timeout = 120`; with pytest-timeout, `timeout = 120` |

Iteration flags: `-x` stop on first failure, `--lf` last failures only, `--ff`
failures first, `--sw` stepwise (resume from the last failure), `-k expr` select by
name.

Ruff changed its defaults the same way: since 0.17 a `ruff check` with no `select`
enables an expanded default set of about 400 rules (it was 59: E4, E7, E9, F), so an
unconfigured run reports things like EXE001 and PLW1510 that older versions never did
(ruff 0.15.2 changelog, 0.17.0 breaking changes; measured 12 such findings on this
skill before fixing them). `assets/ruff-pytest.toml` selects its rules explicitly and is
unaffected; pin ruff alongside it.

`strict = true` adopts future strictness options automatically, so pin pytest
(`pytest==9.2.0` in your lock file) or list the four strict keys individually.

## Supply chain

| Risk | Control |
|---|---|
| Known-vulnerable dependency | `pip-audit` in `security.yml`, weekly as well as per push |
| A new release is compromised | lock file with hashes (`pip install --require-hashes`, or `uv.lock`) plus Dependabot cooldown |
| Workflow action compromised | SHA pins, `zizmor` |
| Secret committed | gitleaks in pre-commit and over full history in CI |

The workflows install with `pip install -e ".[test]"` so they work in any project
on day one. Once the project has a lock file, install from it instead.

Testing against the installed package (tox or nox, or a built wheel) catches
packaging mistakes a source-tree run cannot, such as a module missing from the
wheel. `--import-mode=importlib` in the config keeps pytest from importing the
working tree by accident.

## Debugging a failure

| Command | Use |
|---|---|
| `pytest path::test -vv` | one test, maximum detail |
| `pytest -x --tb=long --showlocals` | first failure with locals |
| `pytest --pdb` | debugger at the failure |
| `pytest -s` | disable capture so `print` and `breakpoint()` work |
| `pytest -p randomly --randomly-seed=N` | replay the order from a CI run (the seed is in the header) |
| `pytest --hypothesis-seed=N` | replay a Hypothesis run |

## Sources

- GitHub, secure use reference (third-party actions, SHA pinning): https://docs.github.com/en/actions/reference/security/secure-use
- GitHub, script injections: https://docs.github.com/en/actions/concepts/security/script-injections
- zizmor: https://docs.zizmor.sh/ ; actionlint: https://github.com/rhysd/actionlint
- coverage.py configuration: https://coverage.readthedocs.io/
- pytest, Good Integration Practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
