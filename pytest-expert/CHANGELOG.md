# Changelog

## 2026-11-02: re-verification, test-gaming defenses, research digests

Re-verified on Python 3.13 with current releases, then expanded. Gaps, corrections,
how each source was checked, and what was left undone are in `AUDIT.md`.

### Fixed

| Problem | Evidence | Fix |
|---|---|---|
| pytest-asyncio 1.4.1 plus `filterwarnings = ["error"]`: `INTERNALERROR` at startup, in the config asset and in the wallet example | reproduced; the wallet example still crashed after the first fix whenever the plugin was installed | a targeted `filterwarnings` ignore that works with and without the plugin (asset and wallet verified both ways), plus commented asyncio keys for projects that depend on the plugin |
| sqlite3 connections leaked in the demo and the security template; Python 3.13 turns that into a `ResourceWarning` that fails a later test or crashes at exit | 3.13 warns, 3.12 does not; templates `1 failed, 67 passed` under strict config | closing fixtures |
| Audit `NO_ASSERT` false positive on `with pytest.RaisesGroup(...)` | async template | `RaisesGroup`/`RaisesExc` count as checks |
| Flaky rerun guidance gave only the non-order-dependent figure | Gruber et al. §IV-C | both figures, plus reruns needed to classify one failure |
| Dead Just et al. 2014 link; Inozemtseva and Holmes cited without Chen et al. 2020 | 404; digest | new link; Chen 2020 added |
| Versions | pytest 9.1.1, Hypothesis 6.168.3, CrossHair 0.0.111, mutmut 3.8.0, ruff 0.16.10, zizmor 1.30.1, setup-python v7.0.0, upload-artifact v7.0.1 | pytest 9.2.0, Hypothesis 6.170.14, CrossHair 0.0.112, mutmut 3.8.1, ruff 0.17.3, zizmor 1.31.1, setup-python v7.1.0, upload-artifact v7.0.2 (SHAs from `git ls-remote`) |
| Em dashes in references | 92 | removed |

### Added

- `scripts/diff_guard.py`: test-weakening edits since a git base (deleted tests or files,
  dropped parametrize rows, fewer assertions, trivial asserts, new skips, pragmas,
  collection hooks, config deselection, lowered coverage floor) and source-side
  warnings (constant `__eq__`, special-casing a test's literal), with snapshot rewrites
  reported too.
- Hooks: SessionStart base commit, diff guard and escalation exit in the Stop gate,
  `SubagentStop` entry, stricter block message.
- pytest 9.1/9.2: `max_warnings` ratchet (exit code 6), `-m` marker validation,
  `YIELD_FIXTURE` audit code, `RaisesGroup` rule, `register_fixture` section.
- References: `async-testing.md`, `llm-written-tests.md`, `sources.md`, 16 paper
  digests; snapshot testing; generator checks for Hypothesis.
- `assets/test_async_template.py`; `examples/diagnoses/` (five worked diagnoses).
- Selftests for every script and for the skill's structure; evals 6 to 17 in the
  expectations format.

### Changed

- Workflow step 4 runs the diff guard; step 5 accepts "unproductive, with a reason";
  a short policy for when tests and code disagree.
- Solver search on the wallet demo now about 0.4 s (was 1.7 s on CrossHair 0.0.111).
- Scripts are executable (`chmod +x`) and pass ruff 0.17's expanded default rule set; `ci-cd.md` explains that default change.

### Verification at release

| Check | Result |
|---|---|
| `pytest selftest` | 255 passed (audit 56, diff guard 46, hooks 17, mutate_check 5, flake_hunt 3, structure 128) |
| `examples/wallet`: `pytest` | 87 passed, 96.95% branch coverage |
| `examples/wallet`: `-m integration --no-cov`, `-m symbolic --no-cov` | 2 passed, 1 passed |
| `examples/wallet/demo` | 6 failed, 1 passed (as designed) |
| `mutate_check.py` on `transfer_fee_cents` | 32 of 34 killed (2 equivalent, as before) |
| Templates under the strict config with pytest-asyncio keys | 68 passed, 1 skipped, 1 xfailed, in file order and random order |
| Audit on `assets/`, `examples/wallet/tests`, `selftest/` | clean |
| `ruff check --config assets/ruff-pytest.toml` and `ruff check --isolated` (ruff 0.17's expanded defaults) on scripts and selftests | clean under both |
| Independent fact-check (separate agent re-read the papers, reran the scripts and diagnoses, checked pins with `git ls-remote`) | 2 errors and 12 imprecisions, all fixed: `CASES_REDUCED` missing from the guard's table, a digest contradicting Chen 2020 §6.4, `RaisesExc` deletions not detected by the guard (now tested), `-m` validation undocumented, SessionStart missing `fork`, Gruber 13% row mislabeled, a Petrovic correlation removed as unverifiable, stale pins; about 60 claims confirmed |
| zizmor 1.31.1 (auditor persona), actionlint 1.7.12, pre-commit validate-config | no findings, clean, valid |
| `conftest_ml.py` fixtures on PyTorch 2.15 CPU | 4 passed (shape, overfit, determinism, checkpoint) |

## 2026-10-04: audit and expansion

The previous version (August 2026) was audited by running its own tooling against
deliberately broken tests and its own reference module. Every fix below was
reproduced before it was made and re-checked after.

### Fixed

| Problem | Evidence | Fix |
|---|---|---|
| `audit_tests.py` missed 7 defect classes | reported `clean: no findings` on 7 defective tests; pytest passed 6 of them | negative and left-side float literals, `approx` without `assert`, tuple asserts, `patch.object` without autospec, swallowed `AssertionError`, existence-only asserts; one regression test each in `selftest/` |
| `XFAIL_NOT_STRICT` fired even when the shipped config sets `strict = true` | every xfail flagged under the skill's own config | the audit reads the nearest pytest config |
| `SHADOWED_TEST_NAME` flagged `TestA.test_x` vs `TestB.test_x` | one name table per file | per-scope tables |
| `PYTHONHASHSEED` in a conftest counted as seeding | it only affects child processes and seeds string hashing | removed from seed detection; docs corrected |
| Flaky-test guide blamed dict iteration order | dicts are insertion-ordered | only `set`/`frozenset` of `str`/`bytes` depend on the hash seed |
| Reference `Wallet` accepted `withdraw(nan)` with 19 passing tests and a clean audit | balance became `nan` | property example rewritten with `@example(math.nan)`; the old one used `allow_nan=False` and hid the bug |
| Network guard blocked loopback and Unix sockets | multiprocessing and local test servers fail | loopback and AF_UNIX allowed; gaps documented |
| `frozen_clock` limits undocumented; ruff PT022 | `from time import time` and `datetime.now()` unaffected | documented; `return` instead of `yield` |
| Config passed `--strict-config` and `--strict-markers` on top of `strict = true` | redundant | removed |
| Ruff per-file-ignore globs never matched when run with `--config` | 306 findings, mostly S101 in tests | globs start with `**/`; `mutants/` excluded |
| Ruff SIM105 rewrites a swallowed assertion into `contextlib.suppress` | makes the hidden failure permanent | SIM105 off for tests; audit reports `SWALLOWED_FAILURE` |
| Stale versions | ruff 0.15.11, ruff hook id `ruff`, pre-commit-hooks v5.0.0, checkout v4, setup-python v5 | ruff 0.16.10, `ruff-check`, v6.0.0, checkout v7.0.1, setup-python v7.0.0 |
| CI workflow security | no `permissions`, tag-pinned actions, credentials persisted | least privilege, SHA pins, `persist-credentials: false`, concurrency; actionlint clean, zizmor auditor persona no findings |

### Added

- Audit: `SWALLOWED_FAILURE`, `BARE_COMPARISON`, `NAN_COMPARE`, `WEAK_ASSERT`,
  `SUPPRESSION_NO_REASON`; inline suppression with required reason; `github` and
  `sarif` output; severities; integration-directory exemption for `NETWORK`;
  `asyncio.sleep(0)` allowed; fixtures checked for `SLEEP`.
- `scripts/mutate_check.py`: stdlib targeted mutation testing (workflow step 5).
- `scripts/flake_hunt.py`: multi-seed reruns, order-dependent vs nondeterministic.
- Claude Code hooks: post-edit audit and a Stop gate with fingerprint skip and loop guard.
- Workflows: `tests.yml`, `nightly.yml`, `security.yml`, `dependabot.yml`.
- Templates: property, security, fuzz target, integration conftest.
- Hypothesis profiles (`dev`, `ci`, `nightly`, `crosshair`) and directory markers in `conftest.py`.
- References: property/fuzz/symbolic, security, mutation, integration, flaky tests,
  CI/CD, hooks.
- `examples/wallet` worked example with a `demo/` of planted bugs.
- `selftest/` (60 tests) and `evals/evals.json`.

### Changed

- `references/ci-and-coverage.md` split into `ci-cd.md` and `flaky-tests.md`.
- `assets/ci.yml` replaced by `assets/github/workflows/tests.yml`.
- Copying this version over an older install leaves `references/ci-and-coverage.md`
  and `assets/ci.yml` behind; delete them, or replace the folder (see INSTALL.md).
- Package no longer includes empty `.benchmarks/` directories created by
  pytest-benchmark during verification.
- `SKILL.md` workflow: every step has a completion check; mutation checking is now
  step 5; new escalation step and testing ladder. Rules 28 to 31 added.
- Shell snippets in docs contain no `#` comment lines, so they paste cleanly into zsh.

### Verification at release

| Check | Result |
|---|---|
| `pytest selftest` | 60 passed |
| `examples/wallet`: `pytest` | 87 passed, 97% branch coverage |
| `examples/wallet`: `-m integration`, `-m symbolic` | 2 passed, 1 passed |
| Templates | 39 passed, 1 skipped (Docker unavailable, skip path verified) |
| Audit on every test file in the skill | clean |
| `ruff check` across the skill (demo excluded on purpose) | clean |
| actionlint, zizmor (auditor), pre-commit validate-config | clean, no findings, valid |
| mutmut on the example | 183 of 194 mutants killed |
