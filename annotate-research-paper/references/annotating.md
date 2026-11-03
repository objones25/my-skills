# Annotating pages

Rules for every note, highlight, and page review. Chunk workers and verifiers read all of it; the build (`scripts/annotate.py`) enforces what it can.

## Contents
- What good annotations do
- Coverage: every figure, table, equation, and algorithm gets a decision
- Page-by-page review: every key point gets a decision
- What goes in each note
- Check questions and their answers
- Writing style
- Edge cases

## What good annotations do

Evidence and strength for each rule are in `references/evidence.md`.

1. **Tag each claim, point to its evidence, and state the scope.** Say which
   figure, table, or section supports each load-bearing claim and whether the
   evidence covers the claim as stated. Faculty evaluate claims as they read;
   novices mostly do not.
2. **Lead with the implication, then the detail.** The first labeled part of a
   note is its "so what": what the thing demonstrates, buys, or costs. Description
   comes second, short, and only if needed. Faculty notes are 75% analytic or
   evaluative; student notes are 21%.
3. **Give figures and tables their message, not their layout.** Data figures:
   **Metric → Baseline → Takeaway**. Diagrams: **Point → Why it matters**. Never
   list the rows, columns, or axes.
4. **Name assumptions, especially the unstated ones.** Mark the load-bearing ones
   and say what breaks if they fail.
5. **Define terms and symbols at the point of use**, in one clause, after saying
   why the term matters here.
6. **Open with a reading route, not a linear walk.**
7. **Keep each note to one idea and off the text. Highlight every key point**, on
   the few words that carry it, each with a reason on hover. A highlight with no
   reason is decoration. The band is 6-15% of a page's body text: under 4% means
   key points were skipped, over 25% means nothing stands out, and the build fails
   both.
8. **Make the reader answer before reading the answer.** Check questions demand
   explanation (why, how, what if). The answer, with its reason, sits upside down
   right under the question, so the reader tries first and then checks without
   leaving the page.
9. **Match every critique verdict to its text** (see `references/guide-and-critique.md`).

## Coverage: every figure, table, equation, and algorithm gets a decision

`extract.py` writes `inventory.txt`: every figure, table, numbered display
equation, and algorithm (with page and line range), the section headings, and the
reference-list entries. `annotate.py` refuses to finish until each figure, table,
equation, and algorithm is annotated or listed in `skip` with a reason, and until
each has an `audit` entry. Every annotated figure or table needs one covering note
with `fig_kind` (see "What goes in each note" below).

An annotation covers an object when its anchor touches the object's caption line,
or its `title` (or a `covers` list) names it. A good skip reason says why this
reader needs nothing ("duplicate of Fig. 2 at larger scale"); "self-explanatory"
is not a reason. Skipping an object does not exempt it from the audit.

The extractor's detection is a net: two-column layouts and split captions make it
miss or mangle objects, and it may list a stray line (or an in-text "Figure 1"
reference) as an object. Read the pages and trust your own count; annotate
anything it missed (name it in `title`/`covers`), and skip false positives with
"not a real object: in-text reference". When a stray in-text "Figure 1." sits on
the page next to the real caption, the build ignores a `covers` name match from
the real figure's page for that twin, so the skip entry and the real note do not
collide.

## Page-by-page review: every key point gets a decision

Figure coverage and notes are not enough on their own: a page can carry a
load-bearing claim, a number, or a mechanism that no figure or note touches. The
spec's `page_review` lists, for every page in range, that page's key points, each
with a verbatim quote and a decision.

```json
"page_review": [
  {"page": 6, "section": "exploitation",
   "points": [
     {"quote": "We used AFL, a well-known and highly successful evolutionary fuzzer", "what": "design",
      "decision": "mark", "why": "Everything downstream depends on the crashes this off-the-shelf fuzzer finds."},
     {"quote": "Driller limits the search space of the symbolic execution", "what": "mechanism",
      "decision": "note:How Driller hands inputs to the fuzzer"},
     {"quote": "The crashes we targeted were instruction pointer overwrite", "what": "concept",
      "decision": "note:A fixed menu of crash types"}],
   "not_key": [{"quote": "Here we show a function with a basic stack overflow",
                "why": "introduces the listing; the listing's note covers it"}]}
]
```

- **What counts as a key point** (`what`): a `claim` the argument needs, a
  `result` or `number`, a `mechanism` (how a component works), a `definition` or
  coinage, a `concept` the reader must know, a `limitation`, an `assumption`, or a
  `design` choice. Expect 3-8 on a dense page.
- **Decisions.** `mark`: highlight it, with `why` (at most 25 words) saying why it
  matters, shown on hover; not what it says. `note:<title>`: a gutter note on this
  or an adjacent page explains it. `walkthrough:<id>`: the idea belongs to an
  explainer the map plans; it still needs a `why`, and the page still gets its
  notes. Nothing is routed to an explainer by rule, and explainers do not quote
  routed points.
- **Floors.** The build fails a page below these. They scale with the page's body
  words (reference list excluded) and are minimums, not targets: a dense page
  carries 6-8 key points and 3 notes.

  | Body words | Key points: ceil(words / 150), 2 to 6 | Gutter notes | Highlighted share |
  |---|---|---|---|
  | 150-249 | 2 | 0 | no floor |
  | 250-299 | 2 | 1 | no floor |
  | 300-499 | 2-4 | 1 | at least 4% (aim 6-15%) |
  | 500+ | 4-6 | 2 | at least 4% (aim 6-15%) |

  Plus one check question per 6 reviewed pages. Check questions do not count as
  gutter notes; every other note is at least 15 words, every highlight's `why` at
  least 6, and every `not_key` entry quotes 4+ words with a reason of 5+ words. A
  page that truly carries fewer ideas than its length (a table of contents,
  boilerplate) gets `"thin": "<what is on it and why>"` (5+ words), which lifts
  all three floors for that page; `"nothing_key": "<why>"` does the same for a page
  with no key point. Both draw on one budget of 10% of reviewed pages, paper-wide,
  and the verifier checks each one. Mark a reference list as a section with
  `"references": true` instead.
- **Sections.** A review entry's `section` is an `id` from the spec's `sections`
  (the paper map; SKILL.md, step 3), and the build rejects any other name.
- **Highlighting.** Every key point is highlighted on its own words unless a
  note's highlight already covers them. Quote the load-bearing words (2-30), not
  the paragraph; a 2-word coinage is fine if it occurs once on the page. The build fails a page above 25% of its body text highlighted and
  warns above 15%.
- **The salience net.** The build flags sentences that carry a signal of a key
  point: a percentage or ratio; "we found / designed / chose"; "called / known
  as"; "however / cannot / fails to"; "because / instead of". Each flagged
  sentence must be covered by a key point or a note, or listed in `not_key` with a
  reason. Like the figure detector it is a net, not a judge: mechanism sentences
  often carry no signal word, so your own reading decides what is key.
- The build fails if a page with 150+ words of body text (reference list
  excluded) has no review entry, if a quote is not on that page, or if a
  decision points nowhere.

Density of notes should follow density of ideas. The failure this rule exists
for: a competition retrospective whose densest section (bug finding and
exploitation, a dozen new mechanisms in two columns) got two notes while lighter
sections got five. The floors exist for the opposite failure, a long paper's back
half drifting down to whatever passes: in the 44-page survey of Hard rule 7 (SKILL.md),
every appendix page got exactly 2 key points, the number that silenced the old
warning, and 2-4% of its text highlighted.

## What goes in each note

Short labeled parts, one topic per note. The **first** part is the implication.

| Target | Parts, in order |
|---|---|
| Data figure / results table (`fig_kind: "data"`) | **Metric.** what is measured and which direction is better. **Baseline.** the exact comparison point (which system, which setting). **Takeaway.** the deviation from the baseline and what it means. Optional **Caveat.** (error bars, n, scale) and **Supports?** (does the text's claim follow). Never list rows, columns, or axes. |
| Diagram, photo, architecture, schematic (`fig_kind: "diagram"`) | **Point.** the one design idea the figure makes. **Why it matters.** which claim or result depends on it. |
| A central claim | **Claim.** in plain words. **Evidence.** the figure/table/section. **Scope.** what it does and does not cover. |
| Equation | **Why this form.** what the form buys (and what it trades). Then **Reads as.** term by term, naming every symbol. |
| Code listing / worked example | **Why it matters.** what the example demonstrates about the system, often its limits. Then the shortest trace that makes that point. Never a line-by-line paraphrase. |
| Method / architecture | **Why it works.** or **Trade-off.** first, then **Does.** input to output with shapes. |
| Assumption | **Assumes.** the condition. **Load-bearing?** which result needs it. **If it fails.** what breaks. |
| Term / coinage | **Why it matters here.** then **Means.** in one clause. |
| Subtle point (`subtle`, "Watch out") | **Looks like.** the easy misreading. **Actually.** what the evidence supports. **Matters because.** where it bites. |
| Check question (`check`) | The question, plus **Hint.** The `answer` (20-60 words, with the reason) prints upside down under it. |

The build enforces: every non-check note has an implication part; descriptive
parts (`Means`, `Does`, `Shows`, `Reads as`, or unlabeled text) are at most about
half the note; no descriptive part copies most of its words from the annotated
text; code and equation notes lead with the implication; `fig_kind` notes have
their parts in order. Notes over 45 words need a `short` version (one sentence,
the single most useful point) for when a page overflows.

### Restatement versus unpacking

> Restates (fails): **Does.** `char name[20]` reserves 20 bytes on the stack;
> `read_string(name)` reads input into it with no length limit; `printf` prints
> it.
>
> Unpacks: **Why it matters.** This is the easiest possible case on purpose: the
> overflow lands input bytes directly in the return address, so the crash already
> hands over control. Real bugs rarely line up this neatly, which is the limit the
> authors admit on p. 9 (15 of 82 exploited). **The bug.** An unbounded read into
> a 20-byte buffer.

### Figures: the message, not the layout

> Lists the layout (fails): **Shows.** Table 3 has columns for Team, Score, VDS,
> PRS, SAS, BDL and rows for seven teams.
>
> Gives the message: **Metric.** team score (higher is better), summed over
> challenges. **Baseline.** second place, 219.35. **Takeaway.** the winner scores
> 1.79 times second place, and about 81% of that lead comes from patches and SARIF
> assessments, not discovery; accuracy alone does not buy rank (the most accurate
> team placed fifth). **Caveat.** one competition run, no variance.
>
Compute every number and every pattern in a note from the source. A plausible
pattern written without checking is often false: "accuracy separates the middle
of the table" sounds right for this scoreboard, but the most accurate team placed
fifth and a 44%-accuracy team placed third.

## Check questions and their answers

Each `check` note prints the question and a hint upright, then the answer upside
down and muted directly under it, with a one-line instruction to answer first. This
keeps the attempt-then-feedback loop on the page: attempting an answer before
seeing it helps even when the attempt fails, and feedback that explains why
outperforms a bare correct answer. Write answers that explain (20-60 words, the
build rejects fewer than 15). Refer to places by page, figure, or equation, never
"above". Place a question before, or on a different page from, the note that
explains its answer; the build warns when a note on the same page shares most of
the answer's words.

## Writing style

Plain and specific. Use the paper's own symbols and section numbers. Refer to
places as "p. 7", "Fig. 3", "Eq. (2)". No em dashes, no filler ("It's important
to note", "Essentially", "crucial"), no hype. Hedge only with a reason ("single
run, so..."), never by reflex.

## Edge cases

- **Two-column papers**: keep left-column notes' titles explicit ("Fig. 3").
  Section and citation detection is noisier; trust the page images.
- **Theory papers**: the object set is equations, lemmas, and theorems; use the
  `general` checklist.
- **Retrospectives and magazine articles** (like a team's competition write-up):
  claims are mostly self-reported; the critique's job is to separate measured
  results from the authors' account.
- **Preprints / tech reports**: verify the citation from the arXiv or DOI record.
- **Scanned PDFs**: `extract.py` warns when pages lack text; anchor with `rect`.
- **Already-annotated PDFs**: existing annotations are kept; mention it.

---
