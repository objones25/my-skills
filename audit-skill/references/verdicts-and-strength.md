# Verdicts and evidence strength

Each item gets a verdict (what is true) and a strength (how sure the evidence makes us). They are separate: a claim can be `correct` on weak evidence.

## Verdicts

| Verdict | Use when |
|---|---|
| `correct` | The evidence supports the claim as written, on the latest stable release |
| `wrong` | The evidence contradicts it, on any version; or a citation points to a source that does not say it; or a script fails on a valid input; or a test cannot fail |
| `stale` | False on the latest stable release, whether or not it held on an older one. This includes a skill that pins an older version which is no longer latest: the fact may be true there, but that version is out of date |
| `version-specific` | True on the latest release, but behaves differently on a recent older release, and the skill does not say so |
| `couldnt-verify` | Searched to the stopping rule and found too little either way. Strength E. Needs the logged searches behind it (AS019), unless a command was what failed to settle it |
| `disputed` | Set only by `review_round.py` when the 3-round cap is reached with the reviewer still disagreeing |
| `not-checked` | Out of scope for this run |

Two kinds of self-claim need care:

- **A dated record** ("verified on Python 3.13 on 2026-11-02") makes two claims. Whether the check happened is usually `couldnt-verify` unless the skill ships the log. Whether the pinned version is current is judged like any version fact: older than the latest stable release means `stale`.
- **A forecast** ("expected to hold on later releases") is checked up to the latest stable release and no further; say so in the note.
- **A timing or other machine-dependent number** ("takes about 70 s") is rerun when you can. If the skill states the machine and date, judge whether a comparable run reproduces it, and say how close; if it states a bare number as general, a clearly different result on a current machine is `wrong`. When you cannot rerun it, `couldnt-verify`.

Restate each claim atomically before judging it: one checkable statement per item. Splitting long text into atomic facts is what makes automatic fact scoring accurate (FActScore, Min et al. 2023; SAFE, Wei et al. 2024). A sentence with three facts becomes three items.

## Strength, A to E

Adapted from GRADE (Schünemann et al. 2013) and Dybå & Dingsøyr (2008). The rungs for code and docs are this skill's own adaptation; no research was found on grading claims backed only by docs or source code.

| Strength | Means |
|---|---|
| A, strong | Observed in this run: code executed or inspected at the pinned latest version. For empirical claims, an independently replicated finding or a meta-analysis, or two tier 1-2 sources that agree |
| B, moderate | An explicit statement in the official docs for the matching version; or one well-run controlled study with variance reported (several seeds, error bars, code available) |
| C, limited | Docs that only imply it; source code read with an inference step; one study with problems (student subjects, one seed, no baseline); a replication by the same team only |
| D, weak | Practitioner consensus, an industrial self-report, a tier 4 source |
| E, none found | Pairs with `couldnt-verify` |

Move down one level for:

- **Indirectness:** a different version or platform, a toy benchmark, an inference step.
- **Inconsistency:** sources disagree, or docs contradict observed behavior.
- **Imprecision:** one seed, no error bars, a small sample.
- **Risk of bias:** authors evaluating their own tool; internal replications, which confirmed with an odds ratio of 8.64 against external ones in empirical software engineering (Shepperd et al. 2018).

Move up one level for:

- **Independent replication** or reimplementation.
- **A large effect that holds** across settings.
- **Triangulation:** code, docs and execution all agree.

`validate_audit.py` enforces the ceiling the best source allows (AS027): with no command, tier 1 or 2 gives at most B (A needs two tier 1-2 sources), tier 3 C, tier 4 D. It cannot apply the up and down factors above; those are yours to state in the note, and the reviewer checks them.

Recommendations usually top out at C or D unless a controlled study tested that practice itself (Kitchenham & Charters 2007 on fitting evidence to the question).

## Wording the note

One or two sentences: what the evidence shows, which stopping condition ended the search, and any downgrade reason. Mark reasoning that is yours, not a source's, as `[inference]`.
