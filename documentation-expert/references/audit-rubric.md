# Audit rubric and report format

Contents:

- [How to use the score](#how-to-use-the-score)
- [The rubric](#the-rubric)
- [Bands](#bands)
- [Report format](#report-format)
- [Worked example](#worked-example)
- [Sources](#sources)

## How to use the score

The score is a communication device. It compresses an audit into something a
maintainer can act on and compare across documents. It is not the finding.

The finding is the claim ledger from `scripts/verify_claims.py` and the rule hits from
`scripts/lint_prose.py`, both of which cite a file and a line. Lead with
those. A score with no evidence behind it is an opinion wearing a number, and
the reader is right to distrust it.

Score every criterion from its "How to check" procedure, not from impression.
Where the procedure names a command, run the command. Where you could not check
something, score it as unverified and say so rather than guessing.

## The rubric

Weights reflect what developers report actually blocking them: respondents
prioritized every content problem over every presentation problem, and named
incompleteness, incorrectness, and ambiguity as most in need of attention
(Uddin and Robillard 2015). The point values themselves are this skill's
choice, not the paper's.

### Accuracy and grounding (30 points)

| # | Criterion | Points | How to check |
|---|---|---|---|
| 1.1 | Every path, link, and anchor resolves | 8 | `verify_claims.py DOC`; count `missing` and `moved` on kinds `link`, `path`, `anchor` |
| 1.2 | Every documented symbol exists with the documented signature | 8 | Ledger `symbol` claims, then open each definition and compare parameters by hand |
| 1.3 | Every command, flag, and environment variable exists | 6 | Ledger `command`, `flag`, `env` claims |
| 1.4 | Every code example parses and runs | 5 | Ledger `code-block` claims, then execute each example |
| 1.5 | Version and compatibility claims match the manifest | 3 | Compare stated versions against the manifest's declared ranges |

Deduct the full sub-score for a criterion with any unfixed failure. Accuracy is
not partially satisfied: one wrong install command wastes the reader's session
regardless of how right the other nine were.

### Getting started (20 points)

| # | Criterion | Points | How to check |
|---|---|---|---|
| 2.1 | Prerequisites stated with versions, before the install step | 5 | List everything the install assumes; check each appears in the document |
| 2.2 | Install is a copyable code block | 4 | It exists, and it is a block rather than prose |
| 2.3 | A verification step proves the install worked | 3 | A command whose output tells the reader they succeeded |
| 2.4 | First success reachable in under five minutes | 5 | Follow the document literally in a clean directory and time it |
| 2.5 | The failure the reader will actually hit is addressed | 3 | Whatever went wrong in 2.4 either does not happen or is documented |

Criterion 2.4 is the one that cannot be faked. Run it.

### Usage and examples (18 points)

| # | Criterion | Points | How to check |
|---|---|---|---|
| 3.1 | The first example is the common case, not the clever one | 5 | Compare it against what the tests and `examples/` actually do |
| 3.2 | Examples show their output | 4 | Every example either prints a result or is a pure side effect that is described |
| 3.3 | Examples are complete enough to run as written | 4 | Copy into a fresh file, supply nothing extra, run |
| 3.4 | Examples are ordered by how many readers need them | 3 | Common, then configuration, then advanced |
| 3.5 | Elisions use a language comment, not `...` | 2 | `scripts/lint_prose.py` rule CODE002 |

### Coverage of the surface (12 points)

| # | Criterion | Points | How to check |
|---|---|---|---|
| 4.1 | Public names are documented or deliberately excluded | 5 | `scripts/scan_repo.py` gap list; every gap either filled or explained |
| 4.2 | Every environment variable and config key the code reads is documented | 4 | `scripts/scan_repo.py` undocumented env vars |
| 4.3 | Error and failure modes are documented, not only the happy path | 3 | Grep for raised exceptions and non-zero exits; check each is reachable from the docs |

### Structure and scannability (12 points)

| # | Criterion | Points | How to check |
|---|---|---|---|
| 5.1 | Exactly one level-one heading, no skipped levels | 3 | `scripts/lint_prose.py` rules DOC001, DOC002 |
| 5.2 | Headings say what the section contains | 3 | Read the headings alone; can you predict each section |
| 5.3 | Link text is meaningful out of context | 3 | `scripts/lint_prose.py` rules LINK001, LINK002 |
| 5.4 | Images have alt text; no directional references | 3 | `scripts/lint_prose.py` rules A11Y001, A11Y002 |

### Honesty and tone (8 points)

| # | Criterion | Points | How to check |
|---|---|---|---|
| 6.1 | No unfinished markers or unfilled placeholders | 3 | `scripts/lint_prose.py` rules HONEST001, HONEST002 |
| 6.2 | Claims are measurements, not adjectives | 3 | Every performance or ease claim traces to a number or a step count |
| 6.3 | Limits and non-goals are stated | 2 | A reader can tell when this is the wrong tool |

## Bands

| Score | Band | What it means |
|---|---|---|
| 90-100 | Exemplary | A stranger can install, run, and extend without reading source |
| 75-89 | Strong | Complete and accurate; gaps are at the edges |
| 60-74 | Adequate | Works for the common path; readers will hit source for anything else |
| 40-59 | Weak | Real blockers. Sections missing or wrong enough to cost the reader time |
| 0-39 | Poor | A reader is faster reading the code than the docs |

A document with any accuracy failure caps at Adequate no matter what else it
scores. Write the Docs: "Consider incorrect documentation to be worse than
missing documentation." Missing documentation at least sends the reader to the
source instead of down a wrong path.

## Report format

Keep this order. It puts the actionable material first and the score where it
cannot be mistaken for the evidence.

```markdown
# Documentation audit: <document path>

Audited against <commit sha or date>. Scripts run: verify_claims.py, lint_prose.py.

## Blockers

Things that will waste a reader's time today.

1. **<one-line statement>**: `<doc>:<line>`
   Evidence: `<source file>:<line>` shows <what is actually true>.
   Class: removed | renamed | moved | signature changed | undocumented | ambiguous
   Fix: <the specific change>

## Gaps

Present in the code, absent from the docs. Each with a recommendation to fill
or to leave, and the reason.

## Style and structure

Rule ID, location, and the one-line fix. Grouped, not enumerated one by one.

## Not verified

What could not be checked and why. Be specific: "the Docker instructions were
not run because Docker is unavailable here", not "some things were unverified".

## Score

| Category | Score | Notes |
|---|---|---|
| Accuracy and grounding | n/30 | |
| Getting started | n/20 | |
| Usage and examples | n/18 | |
| Coverage of the surface | n/12 | |
| Structure and scannability | n/12 | |
| Honesty and tone | n/8 | |
| **Total** | **n/100** | Band: <band> |
```

The "Not verified" section is not a disclaimer. It is the part a maintainer uses
to decide whether to trust the rest, and omitting it makes the whole report less
useful rather than more confident.

## Worked example

Running the bundled scripts against a real project produces findings of this
shape. From `pallets/click` at commit `2247b35` (2026-10-04):

```text
$ python3 scripts/lint_prose.py click/README.md --min-severity warn
ERROR L1    RM002  README has no section matching install
ERROR L1    RM002  README has no section matching license
WARN  L21   TONE001  'simple' judges the reader's experience; state the step count instead
WARN  L38   CODE001  Fenced code block has no language tag
```

The first two are where judgement starts rather than ends: click's README
deliberately routes install and license to its docs site and `LICENSE.txt`, so
a maintainer may reasonably decline them.

Each of those is a specific edit rather than a judgement, which is the point:
the maintainer can act on all four without agreeing with your taste.

## Sources

- Uddin and Robillard, "How API Documentation Fails", IEEE Software 2015 - https://www.cs.mcgill.ca/~martin/papers/ieeesw2015.pdf
- Write the Docs, documentation principles - https://www.writethedocs.org/guide/writing/docs-principles/
- Diataxis, quality - https://diataxis.fr/quality/
- standard-readme specification - https://github.com/RichardLitt/standard-readme
