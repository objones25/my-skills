# Changelog

## 0.1.0, 2026-11-03

First version, built from the audit in `docs/SKILL_AUDIT.md` of the `my-skills` repository. Reviewed and verified independently: see `AUDIT.md`.

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-03 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0, PyYAML 6.0.3 | 200 passed, 1 skipped (a test of the fallback parser) |
| 2026-11-03 | macOS arm64 (Apple Silicon) | Python 3.13.2, pytest 9.2.0, no PyYAML | 200 passed, 1 skipped (a test needing PyYAML) |

### Evals

`evals/evals.json` and `evals/trigger_evals.json` have not been run through a grader or skill-creator's runner, so trigger rates are unknown.

### With and without the skill

Three tasks, each given to fresh agents on 2026-11-03 (model: Sonnet), once without this skill and once with an earlier draft of it. Scored afterwards with the final `check_skill.py`, leaving out SB012 because test runs create caches. This compares conformance with the standard, which is what the skill teaches. It does not show that the resulting skills work better, and it is one run per task.

| Task | Without the skill | With the skill |
|---|---|---|
| Create a commit-message skill | 1 error, 4 warnings | 0 errors, 0 warnings |
| Create a Postgres migration skill with a script | 1 error, 7 warnings (the error is a folder name, from where that run saved it) | 0 errors, 0 warnings |
| Update `negative-space-programming` | 1 error, 7 warnings | 0 errors, 1 warning (SB043, a rule added after that run) |

Other differences, read from the runs' reports and files:

- **Trigger cases:** none without the skill, and 8 of each kind in all three runs with it.
- **Ground truth:** the migrations run with the skill ran its lock and rewrite claims on a throwaway Postgres 17.9. The run without it reported that it did not test on a live Postgres.
- **Update task:** both runs found the stale suppression and the version error. Only the run with the skill found the invalid frontmatter, installed pyright instead of skipping it, and corrected the ruff note.
- **Not followed even with the skill:** none of the runs executed the evals or packaged the skill, and one reviewed its own work instead of using a fresh agent. The final gate was reworded afterwards because of this.
