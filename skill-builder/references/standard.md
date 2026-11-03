# The skill standard

## Contents

- Tiers
- Conventions
- Checker codes

Requirements depend on what a skill does, so a small skill stays small.

## Tiers

| Tier | Applies when | Requires |
|---|---|---|
| Core | Always | `SKILL.md` with valid YAML and `name` matching the folder; a description covering what, when and negative scope; a way to verify the work; `CHANGELOG.md`; `evals/evals.json`; `evals/trigger_evals.json` |
| Tooling | It ships scripts | Tests, with failing and clean fixtures; exit codes 0, 1, 2; `metadata.test-command` |
| Knowledge | It states facts that can age or be wrong | `references/sources.md` recording check depth; dates and versions beside claims; `AUDIT.md` with a "Not verified" section |
| Large | More than one reference (not counting `sources.md`), or over 150 lines | A routing table; a structure test; an "Extend this skill" recipe |

These are optional, and earn their place when:

- **`examples/`:** a worked example changes how the agent answers. Mark each one illustrative so invented details are not repeated as facts, and index them in a README.
- **`assets/`:** users copy a template or file. Test anything they copy.
- **`INSTALL.md`:** installing takes more than copying the folder.

A one-file skill that wraps an external tool can be correct at Core tier with no references, scripts or assets.

## Conventions

| Topic | Use |
|---|---|
| Layout | `SKILL.md`, `CHANGELOG.md` and `AUDIT.md` at the root; `references/`, `scripts/`, `tests/`, `assets/`, `examples/`, `evals/` |
| Skill tests | `tests/` at the root. A drop-in package may keep tests beside itself if `metadata.test-command` runs them |
| Test command | `metadata.test-command`, naming its dependencies, such as `uv run --no-project --with pytest python -m pytest tests` |
| Version | `metadata.version` |
| Size | Under 500 lines for `SKILL.md`, as the Claude Code docs recommend. Most good skills here run 100 to 280 |
| Name | Lowercase letters, digits and single hyphens, 64 characters at most |
| Style | No em dashes in markdown; some existing structure tests enforce this |

## Checker codes

`check_skill.py --list-codes` prints these with their severity. A warning may be fine for a given skill: use `--ignore CODE` and state the reason.

| Code | Fix |
|---|---|
| SB002 | Quote the value. An unquoted `: ` inside a value is the usual cause |
| SB003 | Rename the folder or the `name` so they match |
| SB004 | Shorten the description and remove `<` and `>`. Note that `->` contains one |
| SB005 | Say when to use the skill, with a "Use when" clause |
| SB006 | Add "Not for X" or "For X use other-skill" |
| SB007 | Fix the path. Ignore this code if it means a path in the user's project |
| SB008 | Add `CHANGELOG.md` |
| SB009, SB011 | Add or fix `evals/evals.json`; see `evals.md` |
| SB010 | Add trigger cases of both kinds; see `evals.md` |
| SB012 | Delete the cache or junk. `package_skill.py` also leaves it out of the zip |
| SB013 | Replace the em dash |
| SB014 | Finish the `TODO`. `package_skill.py` refuses to package while any remain |
| SB015 | Save the file as UTF-8, or remove it |
| SB020 | Write tests for the scripts |
| SB022 | Add `metadata.test-command` |
| SB030, SB031 | Add `references/sources.md` and `AUDIT.md`; see `provenance.md` |
| SB040 | Name the reference in `SKILL.md`, or remove it. Nested files may route through their folder's `README.md` |
| SB041 | Add a test named `test_skill_*.py`; `assets/templates/test_skill_structure.py` is a starting point |
| SB042 | Move material into `references/` |
| SB043 | Add an "Extend this skill" section saying which test or table each new fact must update |
| SB050 | State the date and machine, or say they were not recorded; see `provenance.md`. A heading above the result can give the date |
| SB051 | Fix the filename |
