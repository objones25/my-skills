# Changelog

## 0.1.4, 2026-11-24

`scripts/mutate_check.py` is about 4x faster, with the same verdicts (resynced from pytest-expert, which has the same fix, on top of the 0.1.3 re-sync below). Profiled with cProfile on `scripts/quote_match.py` and `tests/test_quote_match.py` (57 mutants): 98% of the time was spent waiting on test runs, and under 1 s in the script itself.

| Cause | Evidence | Fix |
|---|---|---|
| Each test run got a new, empty `PYTHONPYCACHEPREFIX` while `PYTHONDONTWRITEBYTECODE=1` was set, so every run recompiled pytest and the stdlib from source | One pytest run took 0.81-0.90 s that way, 0.28-0.32 s without the prefix | Dropped the prefix; kept `PYTHONDONTWRITEBYTECODE=1`, which already stops a mutant from being cached |
| Timeout floor of 10 s | Two infinite-loop mutants (`+ -> -`) took 10.0 s each against a 0.3 s baseline | `--timeout-floor` now defaults to 2 s (`MUTANT_TIMEOUT_FLOOR_S = 2.0`) |

The benchmark went from 62.0 s to 15.4 s of test-run time; score 66.7% and every mutant's verdict were identical before and after. New tests: `test_children_reuse_the_bytecode_cache_without_writing_to_it` and `test_infinite_loop_mutant_is_killed_without_waiting_ten_seconds`, both seen failing before the fix.

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-24 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0 | 217 passed, 0 skipped (215 before, plus the 2 new tests); measured before merging main's 0.1.3 |
| 2026-11-24, after merging 0.1.3 | macOS arm64 (Apple Silicon) | Python 3.13.12, pytest via `uv run --no-project --with pytest` | 217 passed (215 from 0.1.3 plus the 2 new tests); the speed fix was re-applied to 0.1.3's rewritten `run()` and `--timeout-floor` default |

## 0.1.3, 2026-11-20

`scripts/mutate_check.py` is re-synced from pytest-expert's (the file header says "Copied from pytest-expert/scripts/mutate_check.py (2026-11-20)"). No other script, document or test changed. What this changes for someone running it from audit-skill, each checked against the script's `--help` and a small run:

| Before | Now |
|---|---|
| Any non-zero exit of the test command counted as a kill, so a mutant that broke the import (pytest exit 2) looked killed | `survived` (exit 0), `killed` (exit 1 by default; change with `--kill-codes`), `killed (timeout)`, and `errored (exit N)` for any other exit. Errored mutants are left out of the score `killed / (killed + survived)` and have their own summary line; if nothing could be evaluated the exit status is 2. Scores from earlier audit runs are not comparable |
| One baseline run | The baseline runs twice and must pass both times (a flaky baseline exits 2); the slower run sets the mutant timeout |
| Exit 1 on a missing test executable, an unparseable or non-UTF-8 source (a traceback) | Exit 2 with one line; 130 after SIGINT or SIGTERM, with the source restored |
| Mutant timeout floor fixed at 10 s | `--timeout-floor` (default 10); timeouts are counted separately |
| Only the direct child was killed on a timeout | The test command runs in its own process group and the group is killed (POSIX) |
| `-h` without `--` printed one usage line and exited 2 | Full help, exit 0 |
| Kinds: comparison, arithmetic, augmented, boolean, `not`, bool, int, `max`/`min`/`any`/`all`, return | Also `matmul`, `shift`, `bitop`, `neg` and `float`; `--skip-kinds` and `--only-kinds` choose kinds. More kinds means more mutants and a longer run |
| No record of which test killed a mutant | Text shows `killed   by <test ids>` (first 3); JSON mutants carry `killed_by`, `exit_code`; `--show-output` adds the last lines of output for errored mutants. The ids come from pytest's short summary and unittest `FAIL:` lines, so a command run with `-p no:terminal` reports none |

Not done: audit-skill's own tests for this script (`tests/test_mutate_check.py`, 9 tests) were not extended; the new behaviour is tested by 178 tests in pytest-expert's `selftest/test_mutate_check.py` against the same file. No independent review of this change. `references/quality-checks.md` (step 5) and `references/sources.md` (the Tools row still says the copy was checked 2026-11-07, "Full; copied with its tests") were not edited.

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-20 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0 | 215 passed, 0 skipped (`python -m pytest tests -q -p no:cacheprovider -p no:randomly`; same count as 0.1.2) |

## 0.1.2, 2026-11-19

Type-checking fixes only; no change in behaviour. `pyright audit-skill` went from 16 errors to 0.

- `scripts/mutate_check.py`: `is_docstring` and `SiteFinder.in_scope` return a real `bool` (they returned a list or `[]` through `and`); `mutate` types the node being rewritten as `Any`, because the site finder has already matched its kind to the node class.

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-19 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0 | 215 passed, 0 skipped (same count before and after the change) |

## 0.1.1, 2026-11-15

Found by the first real full audit (documentation-expert, 2026-11-15; report in the repository's gitignored `audits/` folder).

| Problem | Evidence | Fix |
|---|---|---|
| A claim the reviewer reported missing in the last round was added as `disputed` with no verdict and no evidence, so the report counted it as checked and read "(full)" | documentation-expert run: "Checked 823 of 823 items", with item R3-0823 (revive's `exported` rule, `evals/evals.json:33`) never checked by anyone | `merge` adds it as `not-checked` and lists it in `missed_at_cap_ids`; AS012 accepts an unchecked in-scope item only if it is on that list; the report counts it as unchecked and titles itself "Partial audit". Tests `test_missed_item_in_scope_at_cap_is_not_checked_and_the_report_is_partial` (watched failing first) and `test_unchecked_in_scope_item_not_recorded_by_merge_still_fails_validation` (watched failing against a mutant that disables the AS012 check) |
| Workers logged searches that never ran (budget refusals) as searches with 0 hits, twice in one run | Rows Q-B11-020..022 and Q-B04-016..018 in that run, removed by the lead | New `scripts/search.py`: queries Crossref, Semantic Scholar or arXiv and writes the log row itself, only when the search ran; a 429 is retried at most 3 times honouring Retry-After; exit 2 and no row otherwise. 11 tests in `tests/test_search.py`, run against a local server; a mutant that logs failed searches is caught by 4 of them. `searching.md` (new step 8) and the worker brief say a refused search is not "0 hits" and that no tool may route around a search budget or a bot check |

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-15 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0, pdftotext 26.02.0 | 215 passed, 0 skipped |
| 2026-11-15 | macOS arm64 (Apple Silicon) | Python 3.11.18, pytest 9.2.0, pdftotext 26.02.0 (run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`) | 215 passed, 0 skipped |

Not done: no independent review of this change, no eval rerun, and audit-skill was not audited by itself (skill-builder Update step 2) for this targeted fix.

## 0.1.0, 2026-11-07

First version, built from the design in `docs/superpowers/specs/2026-11-07-audit-skill-design.md` of the `my-skills` repository, with skill-builder's Create mode. The method is grounded in four research passes (systematic-review search methods, fact-checking and source credibility, evidence grading, LLM citation failures); see `references/sources.md`. The quotes this skill uses from those sources were string-matched against raw fetches with its own `fetch_source.py` and `quote_match.py`.

- Scripts: `start_audit.py`, `find_candidates.py`, `fetch_source.py` (raw HTML and PDF text, login and error pages refused, file-locked index for parallel workers), `quote_match.py`, `validate_audit.py` (codes AS002 to AS027), `review_round.py` (blind packet, mechanical merge, 3-round cap, freeze when review ends), `render_report.py` (coverage first, refuses before review ends or after tampering), `plant_errors.py` (planted-error eval), and `mutate_check.py` copied from pytest-expert.
- Two independent reviews and three eval runs found defects before release; each is listed with its fix in `AUDIT.md`.
- `skill-builder` 0.2.0 calls this skill from its Update mode.

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-07 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0, pdftotext 26.02.0 | 202 passed, 0 skipped |
| 2026-11-07 | macOS arm64 (Apple Silicon) | Python 3.11.14, pytest 9.2.0, pdftotext 26.02.0 (run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: Homebrew's 3.11 has a broken site plugin) | 202 passed, 0 skipped |

### Evals

Run on 2026-11-07 on macOS arm64, each by a fresh Claude subagent following `SKILL.md`, with a separate fresh Sonnet subagent as the blind reviewer. One run per case, so these show direction, not rates.

| Case | Version | Result |
|---|---|---|
| 4, control skill, full scope | Final (after both reviews) | "Audit of control-skill (full)", checked 5 of 5, 5 confirmed, review clean after 3 rounds; no false flags. Two script gaps it exposed (false "missing" rows from later-round reviewers, unknown row ids accepted) were fixed afterwards |
| 4, control skill, full scope | Draft (before the first review's fixes) | Checked 6 of 6; flagged the fixture's own version note as stale, correctly, which was a fixture bug; the note was removed |
| 1, planted copy of pytest-expert, `references/mutation-testing.md` | Mid-fix: the scripts changed during the run, which forced the auditor to re-record the tree hash and redo its search log | Detected all 3 planted errors in that file (P2, P3, P7; `plant_errors.py score`: 3/7, the other four plants are in other files). Validation passed before render, so every quote matched its saved source. It also flagged 6 items in pytest-expert that were not planted; the auditor checked them by hand and judged them real errors (listed in the run report). 3 items disputed at the cap; one of those disputes was a strength-only disagreement, since fixed |
| 2, 3, 5, 6 | | Not run |

Trigger evals (`evals/trigger_evals.json`) have not been run.
