# Run folder and file formats

Every file the scripts read or write, and every validator code with its fix. The vocabularies are defined once, in `scripts/audit_common.py`; `tests/test_skill_structure.py` fails if this page and that module disagree.

## Contents

- Layout
- items.jsonl
- dismissed.jsonl, suggestions.jsonl
- search_log.jsonl
- sources/index.jsonl
- meta.json tools
- Review files
- Validator codes

## Layout

```text
<repo root>/audits/<skill>/
  2026-11-07.md            the report (render_report.py writes it)
  2026-11-07/              the run folder
    meta.json              skill path, scope, tree hash of the skill, report path, tools
    workspace/             a copy of the skill: run, mutate and break things here only
    candidates.jsonl       lines and files the inventory must cover (find_candidates.py)
    items.jsonl            the inventory, with verdicts and evidence (the auditor writes it)
    dismissed.jsonl        candidate lines that hold nothing checkable, each with a reason
    suggestions.jsonl      unproven concerns, kept out of the findings
    search_log.jsonl       every query exactly as run
    sources/               raw text of each fetched source, plus index.jsonl
    evidence/              long command outputs, if you want them as files too
    review/state.json      round counter, accepted item hashes, status
    review/round-N/        packet.jsonl, hashes.json, verdicts.jsonl, sources/, feedback.jsonl
```

A second run on the same day is `2026-11-07-2`. The repo root is the audited skill's git top-level, else the current directory. `audits/` should be gitignored; `start_audit.py` warns when it is not and never edits `.gitignore`.

## items.jsonl

One JSON object per line, one atomic claim per item.

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Unique, e.g. `I0001` |
| `file`, `line` | string, int | Where the claim is, relative to the skill root |
| `end_line` | int, optional | Last line, for a claim or code block spanning lines |
| `kind` | enum | `tool-behavior`, `empirical`, `recommendation`, `citation`, `self-claim`, `script`, `example`, `test` |
| `text` | string | The claim, restated as one checkable sentence |
| `in_scope` | bool | Whether this run checks it |
| `verdict` | enum | `correct`, `wrong`, `stale`, `version-specific`, `couldnt-verify`, `disputed`, `not-checked` |
| `strength` | `A` to `E` | Required for every verdict except `not-checked` and `disputed`; see `verdicts-and-strength.md`. Capped by the best source's tier (AS027) |
| `evidence` | list | Command or source entries, below |
| `searches` | list of ids | Search-log ids behind a source-backed verdict |
| `note` | string | One or two sentences: why this verdict |
| `dispute` | object | Written by `review_round.py` at the cap: `auditor` and `reviewer` positions |
| `review_note` | string | Written by `review_round.py` when it settled a strength-only disagreement |

Evidence entries:

```json
{"type": "command", "command": "python -c \"import inspect, pytest; print(inspect.signature(pytest.fixture))\"", "output": "(fixture_function=None, *, scope='function', ...)", "exit_code": 0}
{"type": "source", "source_id": "S004", "tier": 2, "interest": "none", "quote": "73% of the real faults were coupled to common mutants"}
```

`interest` names any stake the source has in the answer (`vendor of the tool`, `authors evaluate their own tool`), or `none`.

## dismissed.jsonl, suggestions.jsonl

```json
{"file": "SKILL.md", "lines": [12, 40], "reason": "navigation text, no claim"}
{"file": "LICENSE", "all": true, "reason": "licence text, no claims about the skill's subject"}
{"file": "scripts/flake_hunt.py", "line": 84, "text": "The seed loop could share one subprocess; not shown to be faster"}
```

The first two lines are dismissals. The third is a suggestion. Only binary files, licences and lockfiles may be dismissed whole (AS026); scripts, tests, examples, assets and data files such as YAML or JSON are always inventoried, and a CHANGELOG states test results, so it is never dismissed whole. Every dismissal is listed in the report, so the reviewer and the owner can challenge it.

## search_log.jsonl

```json
{"id": "Q0007", "engine": "WebSearch", "query": "\"Are mutants a valid substitute for real faults\" Just 2014", "date": "2026-11-07", "hits": 10, "opened": ["https://www.cs.ubc.ca/~rtholmes/papers/fse_2014_just.pdf"], "purpose": "known-item", "item_ids": ["I0031"]}
```

`purpose` is one of `support`, `disconfirm`, `known-item`, `snowball`. Paste the query exactly as run, never retyped from memory. `engine` can be `WebSearch`, `Context7`, `Google Scholar via WebSearch`, `arXiv listing`, and so on.

## sources/index.jsonl

Written by `fetch_source.py`, never by hand: `id`, `url`, `final_url`, `status`, `content_type`, `sha256` (of the raw bytes), `fetched_at`, `path`. The text file is what quotes are matched against.

## meta.json tools

Append what was installed or consulted for tool-behavior checks:

```json
"tools": [{"name": "pytest", "version": "9.2.0", "how": "uv venv in workspace, latest stable on PyPI 2026-11-07"}]
```

## Review files

`covered.jsonl` (written with the packet): every inventoried item's `id`, `file`, `line`, `end_line` and `text`, without verdicts, so the reviewer can judge coverage.

`packet.jsonl` (written by `review_round.py packet`): `id`, `file`, `line`, `end_line`, `kind`, `text`, `commands`, `sources` (each `url` and `quote`). No verdict, strength, note, output or tier.

`verdicts.jsonl` (the reviewer writes it):

```json
{"id": "I0031", "verdict": "correct", "strength": "B", "reran": [], "problems": []}
{"id": "I0040", "verdict": "wrong", "strength": "A", "reran": [{"command": "python -c \"...\"", "output": "(..., scope='function', ...)"}], "problems": [{"code": "output-differs", "detail": "installed 9.2.0 prints scope='function'"}]}
{"id": "new", "file": "references/fixtures.md", "line": 88, "kind": "tool-behavior", "text": "the claim the auditor missed", "problems": [{"code": "missing-item", "detail": "states a default"}]}
```

`reran` must hold every command in the packet entry with the output the reviewer saw (and `ran_as` if it adapted the command), and every row needs a strength. The reviewer fetches every cited URL with `fetch_source.py --dest review/round-N/sources` after the packet is made. `merge` refuses (exit 2) until all of that is present; it cannot tell whether the reviewer really ran the commands or stayed out of the auditor's files.

Problem codes the reviewer uses: `does-not-support` (the quote is real but does not entail the claim), `tier-too-low`, `vested-interest`, `output-differs`, `not-atomic`, `missing-item`, `bad-dismissal`, `weak-search`. `merge` adds `quote-not-in-reviewer-copy`, `verdict-mismatch` and `strength-mismatch` itself.

## Frozen after review

When the review loop ends, `review_round.py` records hashes of every item and of `meta.json`, `candidates.jsonl`, `dismissed.jsonl`, `suggestions.jsonl`, `search_log.jsonl` and every file under `sources/` in `review/state.json`. `render_report.py` refuses if any of them changed afterwards. During the loop, an item once sent to the reviewer may be fixed or downgraded but not deleted or set to `not-checked` (`packet` and `render_report.py` refuse), so a reviewed finding cannot be hidden between rounds either.

## Validator codes

| Code | Meaning | Fix |
|---|---|---|
| AS002 | The audited skill changed during the audit | Restore it. Work only in `workspace/`. Start a new run if the change was deliberate |
| AS003 | `candidates.jsonl` missing | Run `find_candidates.py RUN` |
| AS010 | A candidate is covered by no item and no dismissal | Add the item, or dismiss the line with a reason |
| AS011 | Schema error in an item, dismissal or evidence entry, or a location not in the skill (including a dismissal's file) (unknown file, line past the end, `end_line` before `line`, a text-file item over 60 lines) | Fix the field named in the message |
| AS012 | Scope mismatch | In-scope items need a verdict; out-of-scope items are `not-checked`; targeted scope decides by path |
| AS013 | Definite verdict with no evidence | Add the command or source, or downgrade to `couldnt-verify` |
| AS014 | Evidence cites a source not in `sources/` | Fetch it with `fetch_source.py` |
| AS015 | Quote not in the saved source text | Copy the quote from `sources/S###.txt`, not from memory or WebFetch |
| AS016 | Quote too short to be specific | Quote at least 25 characters, 12 per `...` fragment |
| AS017 | A tier 5 or tier 6 source cited as evidence | Use it only as a lead to a tier 1 to 4 source, or `couldnt-verify` |
| AS018 | Verdict and strength disagree | `couldnt-verify` is E; definite verdicts are A to D |
| AS019 | Cited sources, or `couldnt-verify` without a command, with fewer than two logged searches (different queries) that list the item in `item_ids` | Log the searches you ran, with `item_ids`, and list their ids on the item |
| AS020 | Strength A without execution or two distinct tier 1-2 sources | Run it, add a second strong source, or lower to B |
| AS021 | Command evidence without command or output | Paste the exact command and its captured output |
| AS023 | No disconfirming search among the item's searches | Run and log a search for counter-evidence |
| AS024 | Disputed item without both positions, or one `review_round.py merge` did not mark | Only `merge` sets `disputed`; give the item a real verdict |
| AS025 | Search-log entry breaks the schema, or reuses an id | Fix the field named in the message |
| AS026 | A script, test, example, asset or data file dismissed, or a whole text file other than a licence or lockfile | Inventory it; dismiss only lines of prose with nothing checkable |
| AS027 | Strength above what the best source allows: tier 1 or 2 gives at most B (A needs a command or two tier 1-2 sources), tier 3 C, tier 4 D; a single non-tier-1 source caps an empirical claim or recommendation at C | Lower the strength, or add better evidence |
