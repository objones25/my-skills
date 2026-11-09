# Changelog

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
