# Updating a skill

*Illustrative.* The `widget-audit` skill here is invented.

**Request:** "Our widget-audit skill feels out of date. Can you bring it up to standard?"

## 1. Measure

```bash
python3 scripts/machine_stamp.py --package pytest
python3 scripts/check_skill.py widget-audit
```

```text
warning SB006 SKILL.md: description has no negative scope or handoff
warning SB008 CHANGELOG.md: no CHANGELOG.md at the skill root
warning SB010 evals/trigger_evals.json: evals/trigger_evals.json is missing or lacks both should-trigger and should-not-trigger cases
warning SB050 SKILL.md:40: test result stated without a date and machine: All 12 passed.
widget-audit: 0 error(s), 4 warning(s)
```

Its tests give 11 passed and 1 skipped, because `mypy` is not installed. With `mypy` installed, the skipped test fails. That is the first real bug, and it was hidden by the skip.

## 2. Run its promises

`SKILL.md` says "14 checks", and `--list-codes` prints 15. It says "All 12 passed", which is now false on this machine.

## 3. Rank

1. Broken: the type-check test fails.
2. Wrong: the "14 checks" count, and "All 12 passed".
3. Missing: negative scope, trigger cases, CHANGELOG.

## 4. Fix test-first

The count becomes a test comparing the docs to the rule table. It fails with `14 != 15`, and the doc is corrected. The type error gets the same treatment, with the failing test already in place.

## 5. Report

> **widget-audit now passes.** On 2026-11-03 on macOS arm64 with Python 3.13.2 and mypy installed: 13 passed, 0 skipped. The checker reports 0 errors and 0 warnings.
>
> Fixed: a type error hidden by a skipped test, and a wrong check count, now guarded by a test. Added: negative scope, 16 trigger cases and a CHANGELOG.
>
> Not done: the evals have not been run against a model, so trigger rates are unknown. The earlier "12 passed" result is kept in the CHANGELOG with "machine not recorded".
