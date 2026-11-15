# Audit of documentation-expert

## 2026-11-15: full audit with audit-skill 0.1.0

Report: `audits/documentation-expert/2026-11-15.md` in the `my-skills` repository (gitignored, local only), run folder beside it.

Coverage as the report states it: "Checked 823 of 823 items (100%): 596 confirmed, 35 wrong, 1 stale, 102 disputed, 89 could not be verified. Independent review: stopped at the 3-round cap; 102 item(s) disputed." That line overstates coverage by one: item R3-0823 (revive's `exported` rule, `evals/evals.json:33`) was reported missing in the last round and never checked, but audit-skill 0.1.0 recorded it as disputed and counted it. The true figure is 822 of 823, a partial audit. The bug is fixed in audit-skill 0.1.1.

How it was checked: 20 checking agents ran every script and command in a throwaway copy (Python 3.13 and 3.15, Rust 1.100.0, Go 1.27, Sphinx 9.2.0, staticcheck 2026.2.2, TypeScript 7.0.2, github-slugger 2.0.0), fetched each cited source raw and string-matched every quote. Fresh Sonnet reviewers rechecked every item blind over 3 rounds.

All 35 wrong items and the stale one are fixed; see CHANGELOG 1.1.

Limits of this audit:
- The web-search budget ran out partway; many owed searches went to Crossref, which returns titles only, so many "could not be verified" verdicts rest on thin searching.
- The Uddin and Robillard paper was read only as the authors' manuscript (a draft comment is still in it); the findings that rest on it are strength C.
- Two reviewers saw some auditor verdicts by accident; fresh reviewers redid those entries and agreed.
- R3-0823 (revive's `exported` rule) is unchecked.
- The 102 disputed items were not resolved; most are reviewers giving "correct, strength D" on practitioner sources where the auditors said "could not verify".
