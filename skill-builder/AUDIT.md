# Audit of skill-builder

Date: 2026-11-03. Reviewers: two fresh agents that wrote none of this skill, one reviewing and one verifying the fixes, each working from the files and running its own experiments. Tests were added for the behaviour fixes below. History does not show the order within a commit, so "written first" rests on the author's report.

## Errors found

| Claim or behaviour | Evidence | Fix |
|---|---|---|
| `package_skill.py` reported passing tests that never ran | Without PyYAML, a block-scalar test command parsed to the literal `>-`, which the shell ran as a redirect | The parser reads block scalars, and exits 2 on YAML it cannot read reliably |
| "No zip is left" on failure | Undecodable output, a timeout, a non-text command, and a stale earlier zip all left a zip | Build to a temporary path, rename on success, and remove any old zip first |
| Packaging was safe to share | A link to a file outside the skill was zipped, and `.env`, `.git` and `.venv` were included | Refuse such links, and exclude those files |
| Exit 2 means could not check | A non-UTF-8 file caused a traceback and exit 1, the same code as findings | New SB015; an unreadable `SKILL.md` exits 2 |
| The final gate proves the skill is finished | An untouched scaffold, full of `TODO`, passed it | New SB014; packaging refuses while any remain |
| Packaging tests proved what they named | A test expected a failing command and only checked one field; 7 mutants survived | Rewrote the tests, including one that fails if tests run in the source folder |
| The skill met its own standard | No "Extend this skill" section, and a one-line CHANGELOG | Both added. New SB043, and SB041 now also covers skills with several references |
| `new_skill.py` exit contract | A file as `--dest` crashed, and `foo\n` was accepted as a name | Catch `OSError`, use `fullmatch`, and reject reserved names |
| SB050 false positives on labelled results | The date in a parent heading was not seen; paper digests were flagged | Use all parent headings, accept "not recorded", and skip `papers/` |
| A zip could contain itself | Running from inside the skill with the default `--out` | `--out` inside the skill exits 2 |
| Executable bits lost in the test copy | A test command running `./run.sh` failed with a permission error | Keep file modes when unpacking |
| The generated structure test failed on a block-scalar description | `>-` contains `>` | Read the description block properly |
| SB011 accepted missing ids | A case with no `id` passed | Require a unique integer or name |

### Found while verifying those fixes

| Claim or behaviour | Evidence | Fix |
|---|---|---|
| The packager crashed after the tests passed when the zip could not be moved | Temp and `--out` on different volumes gave a traceback | Build beside the target; a failed move exits 2 |
| A `\|` block still reported passing tests that never ran | Without PyYAML, its lines were joined into one command | The fallback parser refuses literal blocks and indentation indicators |
| Excluded secrets could be packaged through a link | `config.txt -> .env` inside the skill was zipped | Check the link's target against the exclusions |
| The final gate passed an unfilled description | Quoted text was ignored when looking for `TODO` | Frontmatter is checked in full |
| The fallback parser crashed on an empty key before a blank line | `min()` over nothing | Fixed |
| The checker read folders the packager excludes | `.venv` tests hid SB020, and non-UTF-8 files there gave SB015 | One shared exclusion rule |
| Broken and looping links | Dropped silently, or a crash on Python 3.12 | Refused |
| Unreadable folders disappeared with exit 0 | `chmod 000` folder | Exit 2 |
| Credential files were zipped | `.envrc`, `.netrc`, `id_rsa`, `*.pem` and others | Excluded; `.env.example` is kept |

### Finding checked and changed

The review suggested that eval ids must be integers, as skill-creator's schema states. Calibration showed most existing skills use string ids, so the checker accepts both, and the template uses integers.

## How things were checked

- The script and template fixes were written test-first, and the suite runs with and without PyYAML.
- The checker was run on all 17 other skills in the repo to look for false positives.
- Mutation testing on a copy, by the reviewers: first 25 mutants, of which 7 survived. After those fixes, 44 mutants, of which 4 survived; tests were added for 3 of those, and the fourth checks one JSON field.
- Docs and skill-creator source were read by the reviewer against `references/sources.md`.
- Three tasks were run in fresh agents with and without the skill. See `CHANGELOG.md`.

## Not verified

- Trigger rates beyond one pilot on one machine, in `CHANGELOG.md`. The description has not been tuned, and it has known misses.
- Whether skills built with this one work better in use. The with-and-without comparison measured conformance, which is what the skill teaches, and it used one run per task.
- `evals/evals.json` has not been graded across runs.
- Linux and Windows. Everything ran on macOS arm64.
- Whether the mutants from the first review are all killed now. The specific tests were rewritten, and the full mutation set was not rerun.
- Rarer packaging edge cases. A third full review was not run on the last fixes, so more may exist. This is a 0.1.0 tool: read `package_skill.py` before relying on it for anything sensitive, and do not treat its exclusion list as a complete secret scanner.
- The observation in `references/sources.md` about Claude Code listing a skill with invalid strict YAML was seen once.
