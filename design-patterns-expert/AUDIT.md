# Audit, 2026-11-16 (version 1.0.0)

## Fact-check

audit-skill run of 2026-11-16, risk scope. Report: `audits/design-patterns-expert/2026-11-16.md` in the My_Skills repo (gitignored, local only); run folder beside it. Coverage line, verbatim: "Checked 244 of 1216 items (20%): 194 confirmed, 16 wrong, 1 stale, 2 version-specific, 15 disputed, 16 could not be verified. Independent review: stopped at the 3-round cap; 15 item(s) disputed."

Every wrong, stale and version-specific item was fixed on 2026-11-16 (`CHANGELOG.md`, by item id). Of the 15 disputed items, 10 had the same verdict from auditor and reviewer and were held open only over how evidence was quoted; the 7 of those whose verdict was "wrong" were fixed anyway. The five verdict disputes:

- I-FIX-F1-0014 (Repository "hard to back out of"): the sentence was removed from `architectural.md`.
- I-FIX-F1-0022 ("version fields"): reworded to Azure's "version files".
- I-FIX-F2-0001 (cost of repair): reworded to Fowler's definition.
- I-FIX-F2-0017 (IETF draft "best available"): the phrase was removed.
- I-FIX-F3-0002 (generator pipeline as Iterator): unchanged; the claim is the skill's reading of Rhodes and is not contradicted.

## Not verified

- The 972 items the audit did not check: the 344 recommendations, most citations in `references/` (see `references/sources.md` for which pages were re-read), most Python behaviour prose outside `verify_python_claims.py`, and the example files beyond running their tests.
- The couldnt-verify items (16): histories of earlier drafts that predate the repo, unsourced generalisations, and absence claims. Where the text still makes such a claim it is phrased as the skill's view.
- The machine-specific timings are from one machine (Apple M4 Pro).
- The new `evals/` cases and `[auto]` checks have not been run against real replies; evals were not run for 1.0.0.
- No independent review of the 1.0.0 changes themselves.

## Mechanical checks, 2026-11-16

- `check_skill.py` before the changes: 1 error (`.DS_Store`), 10 warnings (no negative scope, CHANGELOG, evals, trigger evals, test-command, sources, structure test or Extend section; em dashes in two reference files). After: see `CHANGELOG.md`.
- Tests before: 74 passed, 3 skipped without mypy and import-linter on PATH; 76 passed, 1 skipped with them (Python 3.15.2, pytest 9.2.0, mypy 2.4.1, import-linter 2.16, macOS arm64).
