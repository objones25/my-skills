# Changelog

## 1.1, 2026-11-15: fixes from a full audit

Every fix below answers a finding of the 2026-11-15 audit-skill run (full scope, 823 items, blind 3-round review; details and the report's location in `AUDIT.md`). Item ids refer to that report.

### Scripts (each fixed test-first: the test was watched failing before the fix)

| Item | Problem | Fix | Test |
|---|---|---|---|
| I-W6-0077 | `scan_repo.py` crashed (AttributeError, exit 1) on a valid `package.json` with nested conditional exports | Recursive resolution of export conditions (import, default, require) | `test_nested_conditional_exports_do_not_crash` |
| I-W6-0069 | An unreadable directory exited 1 with a traceback, not the documented 2 | Checks read and search access first; exits 2 | `test_unreadable_directory_exits_2` (skipped as root) |
| I-W6-0078 | An env var read only under a nested test folder (`src/pkg/tests/`) was reported as undocumented | Test folders are ignored at any depth | `test_env_var_read_only_under_nested_test_dir_is_ignored` |
| I-W6-0067, I-F4-0013 | `scan_repo.py` docstring promised "documented commands that no longer exist", and readme-anatomy.md promised "config keys"; it lists neither | Docstring and reference now say what it lists; stale commands are `verify_claims.py`'s job | `test_docstring_does_not_promise_stale_commands_or_config_keys` |
| I-W6-0012 | Anchor slugs dropped underscores, unlike github-slugger 2.0.0, so `#_private` failed and `#private` passed | Underscores kept | `test_slug_keeps_leading_and_trailing_underscores` |
| I-W6-0015 | `HTTPS://...`, `ftp://` and `//cdn` links were reported as missing files | Schemes matched case-insensitively (RFC 3986); `//` links external, other schemes skipped | `test_link_schemes_are_case_insensitive` |
| I-W6-0018 | A `# comment` line made a shell block look like a transcript, so its commands went unchecked | `#` no longer counts as a prompt | `test_comment_line_does_not_make_a_shell_block_a_transcript` |
| I-W6-0021 | `make VAR=1 test`, `make -C dir t`, `make -f file t`, `make -j 4 t` read an option as the target | Variables and option arguments are skipped | `test_make_options_and_variables_are_not_targets` |
| I-W6-0002, I-W1-0034 | Docstring and SKILL.md said every verified claim gets a `file:line` | Both now say: `file:line` for env vars, flags and JS exports, a file for the rest | `test_verify_claims_docstring_does_not_promise_a_line_for_every_claim` |
| I-W6-0101 | No test failed when the setup.cfg extras guard was removed | New test; watched failing with the guard disabled | `test_setup_cfg_extras_requirement_is_not_an_entrypoint` |
| I-W6-0056 | RM003 counted a linked badge twice (4 linked badges reported as 8) | Linked badges counted once | `test_linked_badges_are_counted_once`, `test_plain_badges_still_count` |
| I-W6-0057 | RM004 let a 120-character description pass; standard-readme says "less than 120" | Fires at 120 or more; message says "under 120" | `test_description_of_exactly_120_characters_is_flagged` |
| I-F6-0001 | DOC001 counted `<h1>` shown inside code fences and code spans | Only `<h1>` outside code counts | `test_h1_in_code_fence_or_span_is_not_a_second_title`, `test_raw_html_h1_still_counts` |
| I-W5-0061 | The word table listed `powerful`, `robust` and `kill`, which the linter does not flag (and Google recommends "powerful") | Those rows are marked "not flagged" | `test_style_rules_word_table_matches_linter` |
| I-W5-0130 (code) | CODE001 was attributed to Google's code-samples page, which has no fence-language rule | Attributed to markdownlint MD040 | `test_code001_source_is_not_attributed_to_google_code_samples` |
| I-F6-0007 | "Run from anywhere" was false for the relative commands in `test_scripts.py` | Commands shown with absolute paths; the suite runs from any directory | `test_run_from_anywhere_commands_do_not_use_relative_paths`, `test_suite_runs_from_another_working_directory` |
| I-W6-0058 | A11Y002 was said to fire only for document position | Comment now says it matches by pattern and cannot tell document from rendering position (behavior unchanged) | none (comment only) |

### Documentation

| Item | Was | Now |
|---|---|---|
| I-W1-0005, I-W1-0006, I-W3-0087 | "a survey of 323 IBM developers" who "prioritized every content problem over every presentation problem" | Two surveys of 323 IBM software professionals; the 254 second-survey respondents (203 developers, 51 architects) prioritized five content-related problems over any presentation problem (SKILL.md, audit-rubric.md) |
| I-W3-0059 | `cargo test --doc` "compiles and runs every doc example" | `ignore` blocks are skipped and `no_run` blocks only compiled |
| I-W3-0124 | Worked example showed "state the step count instead" | The script's real output: "state the step count or the time instead" |
| I-W3-0125 | click's README "deliberately routes install and license" elsewhere | click's README has no install or license section |
| I-W4-0001 | standard-readme "opens its Background with" Ken Williams's quote | Its Background quotes it (second paragraph) |
| I-W5-0001, I-W5-0130 | Everything not marked comes from Google | The fence-language and credential rules are marked as the skill's own |
| I-W5-0002, I-W5-0077 | "enables you to" labelled as the skill's own | Labelled Google (word list, "enable") |
| I-W5-0062 | Each listed word "makes a claim... or replaces a fact with a feeling" | Each word is listed for the reason in its Why column |
| I-F5-0001 | Google says use "we" rarely | Google says use it carefully |
| I-F1-0008 | `main.go` listed as a Go package's entry file | Removed (a `package main` is a program, not an importable package) |
| I-G1-0002 | The manifest is the source of truth for "the entrypoints" | "any entrypoints it declares" (go.mod declares none) |
| I-F1-0015 | Gate 3 is "the only one" that tests the document | Gate 3 tests the document as a reader meets it (Gate 2 also exercises it) |
| I-G6-0001 | Every anti-pattern entry names its cost | Most do (19 of 25) |
| I-W1-0079 (stale) | Test record: 21 passed on Python 3.10 to 3.14 (2026-11-03, macOS arm64) | 40 passed on 3.10 to 3.15 (below) |

Also: `metadata.test-command` added, so skill-builder's packager runs the tests in the unpacked copy. Two long sentences in SKILL.md split after the corrections lengthened them.

Not fixed: A11Y002 still matches "above"/"below" by pattern (documented); a `make` claim's ledger key is still its first three tokens; the audit's 102 disputed and 89 could-not-verify items (mostly unsourced rules of thumb and the skill's own process rules) were not acted on.

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-15 | macOS arm64 (Apple Silicon) | Python 3.10.22, 3.11.18, 3.12.16, 3.13.2, 3.14.10, 3.15.2 (stdlib unittest) | 40 passed on each, 0 skipped |
| 2026-11-03 | macOS arm64 (Apple Silicon) | Python 3.10 to 3.14 | 21 passed on each (recorded before this changelog existed) |

Evals were not rerun for this change.
