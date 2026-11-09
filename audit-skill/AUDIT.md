# Audit of audit-skill

Date: 2026-11-07. Reviewer: a fresh general-purpose subagent that wrote none of this skill, asked to break its claims and its trust mechanisms. It worked on copies in scratch space and reproduced each finding below with a script unless marked otherwise. A control-skill eval run (see CHANGELOG) found further problems, listed separately.

## Errors found by the independent review

| Claim | Evidence | Fix |
|---|---|---|
| A report titled "(full)" means everything was checked | Full scope, one evidence-free `couldnt-verify` item, `SKILL.md` dismissed whole and `scripts/tool.py` dismissed by its line 1: validate, merge and render all exited 0 with "# Audit of sample (full)" | `couldnt-verify` needs logged searches (AS019); whole text files and scripts, tests, examples and assets cannot be dismissed (AS026); file lists now cover every file in the skill; "(full)" also needs every file to have an item or a whole-file dismissal of a non-text file |
| A reviewed finding cannot be hidden | An accepted `wrong` item flipped to `not-checked` after review rendered as "No item was found wrong" | `review_round.py` freezes hashes of all items and run files when review ends; `render_report.py` refuses on any change |
| The audit never writes inside the skill | A skill that is its own git repo: `start_audit.py` copied `audits/` into itself until `File name too long` | Root inside the skill moves to the skill's parent; test added |
| Python 3.11+ | `render_report.py` used a PEP 701 nested f-string; the reviewer's run of the suite on Python 3.11 (2026-11-07, macOS arm64) gave 4 failed | Rewritten; suite passes on 3.11.14; a test compiles every script with 3.11 when one is available |
| Source rules apply to source-backed verdicts | Adding a command `true` switched off AS017, AS019, AS020, AS023 | Source rules apply whenever a source is cited; tier 5 and 6 can never be cited |
| Searches are logged per item | One search row with `hits: 0` satisfied every item | Two searches with different queries that list the item in `item_ids`; duplicate search ids are AS025 |
| Strength rules in the references are enforced | One tier 4 source with strength B passed | AS027 enforces the tier ceilings and the single-source cap; the docs now say which rules are not enforced |
| Quotes are matched exactly | An ellipsis could drop "not"; `10⁵` matched "105"; "10 - 20" matched "10 -20"; gaps unbounded | Gaps bounded to 300 characters and may not hold a negation; superscripts keep a marker; the hyphen variant no longer rewrites every "- " |
| Saved sources are the real page | A 302 to a login page was saved as a tier 1 source; hidden text was kept; `<meta>` charset ignored | Login, consent and soft-error pages are refused; hidden elements dropped; meta charset honoured |
| The reviewer reran commands and refetched sources | Copying the packet's command strings passed; a missing reviewer strength passed | `reran` needs each command with its output; every row needs a strength; refetches must postdate the packet; docs now state what cannot be enforced (peeking, faked outputs) |
| A capped run can render | A round-3 "new" row outside the targeted paths wedged the run at AS012 | Such rows become `not-checked`; "new" rows must name a file in the skill and a line |
| Targeted paths are honoured | `./scripts`, `..`, `/etc` all started and matched nothing | Paths normalized relative to the skill; outside paths refused |
| The tree hash catches any edit | `chmod`, new `node_modules/`, nested `.venv/`, empty directories unseen | Hash covers modes, empty directories and virtualenvs |
| SourceCheckup: 50-90% for search-augmented systems | The abstract applies 50-90% to LLM responses; GPT-4 with RAG left "nearly half" unsupported | Corrected in `review-loop.md` and `sources.md` |
| Kitchenham §6.1 for "must make every effort..." | The passage is in §2.2; "already known primary studies" is §6.1.1 | Section numbers corrected |
| Malformed input gives AS011 | `"evidence": ["ran it"]` crashed with exit 1 | Schema checks evidence entries; crashes map to exit 2 |
| Item locations are real | `end_line < line`, a file not in the skill, line 99999, a span of 10^6 lines all passed | AS011 checks file, range and a 60-line cap on text-file items; dismissals must name a candidate's first line |
| "over-collects on purpose" | Plain factual sentences, 4-backtick fences, folded descriptions and `.rst` files were missed | Every prose line is a candidate; fences tracked by marker and length; folded descriptions spanned; `.rst`, `.txt`, `.mdx` read line by line |
| User text renders as text | A note with "\n# Audit of X (full)" made a second heading | Text, notes, quotes and reasons are flattened and escaped |
| The report says quotes matched the reviewer's fetch | Also printed for capped items whose problem was exactly that | Reworded to apply to accepted items |
| Planted-error scoring | One item spanning 10^6 lines "detected" every plant; a disputed item counted even if nobody flagged it | Items over 10 lines do not count; disputed counts only when a side gave an expected verdict; plants note matches the code |
| Tests guard behaviour | Two refusal tests passed when `render_report.py` could not parse; file-bucket logic survived mutation; `skill_name` test depended on the folder name | Refusal tests match exact messages; bucket and risk-scope tests added; the name comes from the frontmatter |
| Evals measure the agent | Eval 1 expected things the scripts guarantee | Those expectations removed; eval 6 added for trust shortcuts under pressure |

## Found by the control-skill eval run

| Problem | Fix |
|---|---|
| The control fixture carried a version self-claim, so it was rightly judged `stale` | Self-claim removed from the fixture |
| The reviewer brief did not forbid reading the auditor's files | Brief lists what the reviewer may and may not open |
| A hedged "if the auditor said X" problem blocked a clean merge; no path to reject a non-problem | Brief forbids conditional problems; the auditor may keep a verdict with counter-evidence, and the next blind reviewer decides |
| A "new" row with no file, line or text reached feedback | `merge` refuses it |
| The gitignore warning fired outside a git repository | Only warns inside a repository |
| Report: confirmed items duplicated, commands flattened into table cells, empty Not checked section | Table shows counts, details follow under a label, empty sections say "None." |
| Dated self-claims and forecasts had no rule | Rule added to `verdicts-and-strength.md` |
| No way to install a newer interpreter into the workspace | `UV_PYTHON_INSTALL_DIR` instruction added |

## Second independent review (fixes verified, new findings)

A second fresh subagent tried to reproduce every row above against the fixed code (all held as tested, with the bypasses below) and hunted for new holes.

| Finding | Evidence | Fix |
|---|---|---|
| An accepted finding could be hidden between rounds (deleted and its line dismissed, or set to not-checked) | Repro: an accepted `wrong` item deleted after round 1; round 2 clean; render gave "(full)" and "No item was found wrong" | `review_round.py` records every id sent to a reviewer; `packet` and `render_report.py` refuse if one is missing or not-checked |
| Markdown injection through command output, ids, files and URLs | A command output containing a fence closed it and added headings | Code fences sized past any backtick run; ids, paths and URLs escaped; dismissal files must be in the skill |
| A round-3 "new" row with a bad line wedged the run | `line: 99999` capped the run, then AS011 forever | `merge` checks the line is inside the file; generated ids cannot clash |
| An auditor could mark an item `disputed` and skip every evidence check | A tier 6 source with a fabricated quote validated and rendered | `merge` records the ids it disputes; AS024 rejects any other; reviewers cannot answer `disputed` or `not-checked` |
| Sources and `meta.json` were not frozen | A source URL, its text and the tools list were edited after review and rendered | The freeze covers `meta.json` and every file under `sources/` |
| Scripts outside `scripts/`, and YAML or JSON, could be dismissed whole for "(full)" | `bin/deploy.sh`, `helper.py`, `defaults.yaml` dismissed whole, report said full | File kind comes from suffix, shebang and content too; only binary files, licences and lockfiles may be dismissed whole |
| Citing one source twice lifted the single-source cap and the A rule | Same S001 twice at tier 2 gave strength A with no finding | Counts use distinct source ids |
| Reviewer "refetch" could point at the auditor's copy | `path: ../../../sources/S001.txt`, `fetched_at: 9999` merged clean | The copy must be an `S###.txt` file in the reviewer's folder with an ISO time after the packet |
| Ellipsis could drop "rarely", "only" and similar | `this method ... improves` matched `rarely improves` | Hedge words join the negation list |
| Merge crashed with exit 1 on malformed rows | `"verdict": ["correct"]` | Type checks; crashes exit 2 |
| Files planted under `__pycache__/` were not hashed | `__pycache__/evil.py` gave validate 0 | The hash skips only `.git`, `.DS_Store` and bytecode |
| Vendored folders were never inventoried but could still read "(full)" | Code read | Their presence prevents "(full)" and they are listed as not inventoried |
| Overclaims in SKILL.md and formats.md | "confirm it reran every command", "Frozen after review" | Reworded to what the scripts actually check |

Not changed: a `couldnt-verify` item backed only by a command still counts as checked; the coverage line now breaks the count down, including "N could not be verified", so the reader sees it.

## Found by the planted-error eval run

| Problem | Fix |
|---|---|
| A strength-only disagreement turned an agreed "wrong" into "disputed" | When verdicts agree, `merge` settles on the weaker strength and records it in `review_note` |
| Reviewers listed "problems" that only restated that the skill was wrong | Brief: a problem is a flaw in the audit, not in the skill |
| Commands with placeholders could not be rerun verbatim | Reviewer records the adapted form as `ran_as` |
| A targeted audit's 4% figure gave no in-scope view | Coverage adds an in-scope line |
| No rule for timing claims | Rule added to `verdicts-and-strength.md` |
| The scripts changed while the run was in progress, forcing it to re-record the tree hash and redo its search log | Author's error, not the skill's; the eval result is recorded with that caveat |

## How things were checked

- Reviewer: read every file; reproduced findings with scripts in scratch; mutation-tested `render_report.py`; fetched and string-matched ten cited sources with the skill's own `fetch_source.py` and `quote_match.py`.
- Author, before the review: string-matched 15 quotes in the references against raw fetches of 11 sources.
- After the fixes: 202 tests on Python 3.13.2 and 3.11.14 (CHANGELOG), each fix with a test added before the fix and watched failing. Mutation testing of `validate_audit.py` (141 of its mutants, before a timeout) left 30 survivors: 9 equivalent or unproductive (a frozen-dataclass flag, a counter used only as a flag, message wording and lengths, the tie-break between duplicate citations of one source), the rest missing tests, now added; three were re-applied by hand and are now killed. `review_round.py`, `render_report.py` and `quote_match.py` were mutation-tested only in their pre-review versions.

## Not verified

- The reviewer did not fetch Shepperd et al., Garousi Table 7, Wineburg & McGrew, IFCN, Grudniewicz, GRADE, Dybå & Dingsøyr, PRISMA, Booth, or the LLM-verification papers other than Walters & Wilder, AttributionBench and SourceCheckup. The author matched quotes from Wineburg & McGrew, Garousi, Grudniewicz, Shepperd, IFCN and PRISMA-S; the rest are cited at abstract level, as `sources.md` says.
- Cochrane section numbers (§4.3.1.1, §4.4.2, §4.4.11) were not checked against the page.
- Mutation results for `validate_audit.py`, `review_round.py` and `quote_match.py` were not finished when the reviewer reported.
- `fetch_source.py` on real paywalled pages, encodings beyond Latin-1, files over 25 MB, and Windows (it uses `fcntl`).
- The reviewer's peeking and faked command outputs cannot be prevented by the scripts; this is stated in `review-loop.md`.
- The fixes made after the second review (its findings, and the eval-driven changes) have not had a third independent review.
