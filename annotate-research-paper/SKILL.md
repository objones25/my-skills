---
name: annotate-research-paper
description: "Annotate a research paper or web article (PDF, URL, or saved HTML) with an expert reader's marginalia written onto the PDF: every page reviewed so each key point is highlighted with a one-line reason, notes that lead with each target's implication, figures read as Metric -> Baseline -> Takeaway, a critique whose verdicts must match their evidence, links labeled by how they were actually checked, check questions with upside-down answers beneath them, and a short explainer PDF for each algorithm or central idea the reader wants to understand (the idea, notation, a diagram, a hand-worked example, a proof outline with only the hardest steps broken down). Use when someone gives a paper or article to understand, review, or present, or says annotate, break down, or help me read this. Annotates only the document given, never the works it cites. For lecture slides use lecture-study-reference; for math-heavy notes use annotate-math-notes."
---

# Annotate a research paper

Produce `<name>_annotated.pdf`: the original document, unchanged underneath, with
a right-hand gutter of numbered notes, a reading-route page at the front, and a
critique page and a key-references page at the back. Each check question carries
its answer upside down, directly beneath it. The notes do on the page what an
expert reader does silently: separate what the paper claims from what it shows,
name the assumptions a result rests on, say what a figure or equation *implies*
(not what it contains), define the paper's own coinages where they appear, and
flag where the evidence does not reach the claim. Every page is reviewed for its
key points, and each key point is highlighted on its own words with a one-line
hover note. Each algorithm or central idea the reader wants to understand gets a
short separate explainer (`<name>_walkthrough_<id>.pdf`, 3 to 5 pages): the idea
first, the notation, the mechanism as a diagram, one example worked by hand, and
an outline of the proof with only its hardest steps broken down.

**The reader** is a strong CS student or grad student reading this paper for the
first, serious time. They can follow the math and the systems. What they cannot
do alone is what an experienced reviewer does: judge whether the evidence
supports the claim, spot the unstated assumption, read the error bars, tell a
real improvement from a weak-baseline artifact, and know which of the field's
standard pitfalls this paper does or does not fall into.

## Hard rules

These override everything else. The build enforces what it can mechanically;
the rest is on you.

1. **Never guess.** Do not state that an event, paper, product, benchmark, person,
   or number exists, does not exist, or is fake unless a search or a source you
   opened this session settles it. Not recognizing something, above all anything
   dated after your training data ends, is never evidence it is not real. If you
   cannot check, write "I could not verify X" and nothing stronger. A note that
   calls something fictional, hypothetical, or nonexistent fails the build unless
   it carries `checked` (what you searched and what came back).
2. **Annotate only what you were given.** The input is the file or URL the user
   handed over. Never fetch and annotate a cited paper or article on your own
   because the document mentions it; cited works appear only as links on the
   Key references page.
3. **Never claim a check that did not happen.** Every link records exactly how it
   was checked (`opened`, `search`, `expected`; see `references/links.md`). Use
   `opened` only when a tool call this session returned that page, and copy the
   title the tool showed into `seen`.
4. **Lead with the implication; never restate.** Every note says why its target
   matters: what it shows, buys, costs, or breaks. A note that paraphrases the
   text beside it fails the build.
5. **Quote only what the paper says; label everything else.** A phrase in
   quotation marks that a note attributes to the paper must be in the paper (the
   build checks every one). Background the paper does not state carries
   `"outside": true` and a link or `checked`. A reading of the method that the
   paper does not spell out is written as an inference ("Inferred: ..."), never
   as the method.
6. **Every page, every key point.** Each key point on each page gets a decision
   and a highlight, and each page meets the floors in
   `references/annotating.md`. Missing a central claim or mechanism is a worse
   failure than a slightly dense page.
7. **No context annotates more than 5 pages.** A paper longer than 5 pages is
   mapped once, cut into chunks of at most 5 pages, and each chunk is annotated
   by its own subagent, then verified by another (see "Workflow"). Every chunk
   passes the build's floors before it is merged. One context working through a
   long paper spends its attention on the front: a 44-page survey annotated in a
   single pass got 22 gutter notes on its first 21 pages and 4 on its last 23.
8. **Explain, do not cover.** An explainer selects: the idea, the mechanism, one
   example, the proof's chain, and at most 3 hard steps. It never paraphrases
   the paper passage by passage, never repeats the margins, never reports errata,
   and never ships without a passing reader review. Walkthroughs built the old
   way (every concept, then every passage) came out 2.5 times the length of the
   paper and taught nothing; `references/walkthroughs.md` has the audit.

## What is in this skill

Read a reference file when its step comes up, not before. Run the scripts; do
not read them, except a docstring when a step says so.

| File | What it holds | Who reads it, when |
|---|---|---|
| `references/annotating.md` | Note types, floors per page, page review, coverage, check questions, style, edge cases | Chunk workers and verifiers, in full (steps 4, 9) |
| `references/spec-format.md` | `map.json`, part files, `globals.json`, verify files, the merged spec | You at step 3; chunk workers at step 4 |
| `references/walkthroughs.md` | Which explainers to plan, the writing protocol, the reader review | You at step 3 ("Planning"); explainer workers and reviewers at step 6 |
| `references/example-explainer.json` | The reference explainer (SimGolf); copy its form | Explainer workers at step 6 |
| `references/guide-and-critique.md` | Reading route, soundness audit, domain checklists | You at step 7; chunk workers for `critique_notes` |
| `references/links.md` | How each link is checked, choosing concept resources, sourcing | Anyone adding a link (steps 4, 7) |
| `references/evidence.md` | The studies behind each rule | Only to justify or change a rule |
| `scripts/extract.py` | Line IDs, object inventory, page images | Step 2 |
| `scripts/parts.py` | Map check and chunking, gated merge, verifier diff | Steps 3, 5, 7, 9 |
| `scripts/annotate.py` | The build: chunk gate (`--part`) and full build | Steps 4, 8 |
| `scripts/walkthrough.py` | Explainer PDFs; its docstring is the explainer spec | Step 6 |
| `scripts/check_links.py` | Optional mechanical link check | Step 7 |
| `scripts/fetch_article.py` | URL, HTML, or verbatim text to PDF | Step 1, for articles |

## Workflow

You, holding this skill, are the orchestrator: you map the paper, plan the
explainers, cut the pages into chunks, then merge, critique, build, and deliver.
Reading pages and writing their annotations happens in subagents with fresh
contexts (general-purpose agents through the Agent tool). Do not read every page
image yourself; the chunk workers and the verifiers do that.

### 0. Install

Copy the skill into the work folder so every subagent uses the same paths
(replace SKILL_DIR with the base directory shown when this skill loaded):

```bash
mkdir -p work/arp && cp -r SKILL_DIR/SKILL.md SKILL_DIR/scripts SKILL_DIR/references work/arp/
pip install pymupdf requests markdown --break-system-packages -q
```

`fetch_article.py` also needs Playwright with Chromium (often preinstalled; do not
download browsers where the environment says not to).

### 1. Get a PDF

A PDF is used directly. A URL or HTML article is converted first, and nothing
else is annotated:

- `python work/arp/scripts/fetch_article.py --url <URL> work/article.pdf` prints
  the live page in headless Chromium (needs network access to that host).
- `--html saved.html --source-url <URL>` renders a saved copy (no network).
- `--text page.md --source-url <URL>` is the fallback when the page cannot be
  printed (blocked host, bot wall, JS-only page). Give it the page's **verbatim**
  text, for example a browser's page-text extraction or text the user pasted.
  Never feed it a summarizer's output; the annotation must be of the author's
  words.

Every converted PDF starts with a provenance header (source URL, retrieval date,
how it was made). For news, blog posts, and essays use the `article` critique
domain.

### 2. Extract

```bash
python work/arp/scripts/extract.py paper.pdf work/ex
```

Outputs `lines.txt` (`p3 L7 [x0 y0 x1 y1] text`), `inventory.txt`, page PNGs in
`pages/`, and `meta.json`.

### 3. Map the paper and plan the explainers

Read the abstract, introduction, and conclusion page images, the section and
object lists in `inventory.txt`, and the first page of each section. Classify the
paper, pick the critique domain, and look up anything you do not recognize (Hard
rule 1). Then read "Planning" in `references/walkthroughs.md`: record the
reader's goal (the user's words, or the default there) and plan one explainer per
algorithm or central idea that goal names, usually 1 to 3, each with its own
goal. A proof appendix is outlined inside the explainer of the result it proves;
dense pages without an explainer are covered by margin notes. Write
`work/map.json` in the format of `references/spec-format.md`, and cut the chunks
(this also checks the map and stops on a gap):

```bash
python work/arp/scripts/parts.py chunks work/map.json work/ex/lines.txt work/chunks.json --max 5
```

A paper of 5 pages or fewer is one chunk: annotate it yourself, with the gate in
step 4.

### 4. Annotate the chunks (parallel workers)

Start one subagent per chunk, all in one message (at most 8 at a time; start the
next wave as they finish). Fill in this brief for each, and do not open a chunk's
page images yourself while its worker runs:

> You are annotating pages {a}-{b} of {paper.pdf}. Other workers have the other pages.
> 1. Read `work/arp/SKILL.md` (Hard rules only), `work/arp/references/annotating.md` and `work/arp/references/spec-format.md` in full, and `work/map.json` (sections, planned explainers, glossary). Read `work/arp/references/links.md` before adding any link, and the "Soundness audit" in `work/arp/references/guide-and-critique.md` before writing `audit` entries.
> 2. For each of your pages, look at `work/ex/pages/p<N>.png` (figures, tables, and math are mangled or missing in `lines.txt`; the image is the source), read its lines in `work/ex/lines.txt`, and list every key point (quote, `what`) before deciding which get a note or a highlight. Objects on your pages are in `work/ex/inventory.txt`. Look up anything you do not recognize before writing about it.
> 3. Write `work/parts/part_{a}-{b}.json` as a part file (`references/spec-format.md`), with `sections` copied from the map unchanged.
> 4. Every page gets its gutter notes and highlights, including pages an explainer covers; the explainers are written separately and do not replace the margins. Tag `what` honestly: a blind verifier tags the same pages. Define a glossary term only on the page the map lists; elsewhere point back to it ("defined on p. 9").
> 5. Gate: `python work/arp/scripts/annotate.py {paper.pdf} work/parts/part_{a}-{b}.json work/parts/out_{a}-{b}.pdf --part --render work/parts/render_{a}-{b}`. Fix every error and rerun until it exits 0, then look at each render.
> 6. Reply with only: the coverage table the build printed, and anything on your pages that contradicts the map. Do not change the map yourself.

Without the Agent tool, do the chunks yourself one at a time, in order: pass the
gate and write the part to disk before opening the next chunk's pages, and never
hold more than one chunk's pages in view.

### 5. Merge

```bash
python work/arp/scripts/parts.py merge work/map.json work/chunks.json work/parts work/spec.json --pdf paper.pdf
```

The merge re-runs every part through its gate. It fails while any chunk is
missing or failing. It writes `work/critique_notes.json`. A note title repeated across chunks usually means
two workers explained the same thing; keep the first.

`spec.json` is generated, so never edit it by hand. Every fix from here on goes
into a part file (through its worker, or yourself) or `work/globals.json`, and
then you merge again. If a worker reports a wrong map decision, change
`map.json`, re-run `parts.py chunks`, and send the affected chunks back.

### 6. Write the explainers (parallel workers), then review them

One subagent per planned explainer, all in one message, with this brief:

> You are writing the explainer "{title}" for {paper.pdf}, pages {pages} (and proof pages {proof pages}, if any). Reader goal for the paper: {reader_goal}. Goal for this explainer: {goal}.
> 1. Read `work/arp/references/walkthroughs.md` in full, the docstring of `work/arp/scripts/walkthrough.py`, and `work/arp/references/example-explainer.json` (the model to follow).
> 2. Follow the protocol in order: read the pages as images; decide the idea, the mechanism's steps, the proof chain, and the at most 3 hard steps (load-bearing and non-obvious) before writing; choose and verify a hand-sized example; then write `work/walkthrough_{id}.json`. Do not paraphrase passages, report errata, or print code. Keep scratch scripts under names that include `{id}`; other workers share the scratch folder.
> 3. Build until it exits 0, then look at every page of the render:
>    `python work/arp/scripts/walkthrough.py paper.pdf work/walkthrough_{id}.json out/<name>_walkthrough_{id}.pdf --verify-runs --render work/render_wt_{id}`
> 4. Reply with only: the build line, the hard steps you chose and why each is load-bearing and non-obvious, and the steps you treated as routine.

When each finishes, start a fresh reviewer for it with the brief in
`references/walkthroughs.md` ("Reader review"), filling in the spec hash from
the writer's build line. Send the review back to the writer, who fixes what it
lists and rebuilds. Each new build gets a new fresh reviewer, until the verdict
is "pass"; then the writer builds with `--review work/review_{id}.json`. An
explainer is done only when that build exits 0: the review passes and its hash
matches the spec, so nothing changed after the review. Read at least the first page of each yourself before
delivery.

### 7. Links, guide, critique

Read `references/links.md` and `references/guide-and-critique.md`. Research the
cited works worth linking and record what you actually did. Write
`work/globals.json` (`guide`, `critique` from `critique_notes.json` and your map,
`cited_works`, and `walkthroughs`: the map's explainer entries plus `spec`, `file`,
and `review`). Merge again; every
later merge also passes `--globals`:

```bash
python work/arp/scripts/parts.py merge work/map.json work/chunks.json work/parts work/spec.json --pdf paper.pdf \
    --globals work/globals.json
```

### 8. Build

```bash
python work/arp/scripts/annotate.py paper.pdf work/spec.json out/<name>_annotated.pdf --render work/render \
    [--link-report work/link_report.json]
```

The build exits non-zero on any rule violation and says how to fix it. It prints
a coverage table (per page: body words, key points, gutter notes, highlighted
share); read it for any stretch thinner than the rest. Renders are named by the
paper's page number (`out_p27.png`, notes pages `out_p27_notes1.png`, plus
`out_guide.png`, `out_critique.png`, `out_refs.png`). Open the guide, critique,
and key-references renders and the sparsest page; the verifiers look at the rest.
A page error in the full build (the paper-wide `thin` budget, for example) goes
back to that chunk's worker; then merge and build again.

### 9. Verify, chunked

One fresh subagent per chunk, all in one message, with this brief:

> You are checking pages {a}-{b} of {paper.pdf}. Do not open `work/spec.json` or `work/parts/` until step 2.
> 1. Read `work/arp/references/annotating.md`. Blind pass: from `work/ex/pages/p<N>.png` and `work/ex/lines.txt`, list every key point on each page, and write them to `work/verify/part_{a}-{b}.json` as a verify file (`work/arp/references/spec-format.md`).
> 2. Then read `work/parts/part_{a}-{b}.json` and your pages' renders (`work/render/out_p<N>.png`, notes pages `out_p<N>_notes1.png`). Report, each with page and quote: notes that restate instead of unpack; figure notes that list layout; claims the page does not support; wrong facts; wrong check answers; `not_key`, `thin`, or `nothing_key` reasons you disagree with; critique verdicts about your pages that do not match the text; links whose title does not match the citation.

Then diff the blind pass against the spec:

```bash
python work/arp/scripts/parts.py verify work/spec.json work/chunks.json work/verify work/ex/lines.txt --pdf paper.pdf
```

It fails while any chunk lacks a verifier output, and lists every verifier point
the spec does not cover or dismissed as `not_key`. Check each finding against the
page. Fix it in its part file, or reject it in `globals.json` `verify_rejections`
(format in `references/spec-format.md`). Act on the verifiers' other findings the
same way. A verifier finding never adds material to an explainer; explainers
change only through their own review. Then merge with `--globals` and build
again, until
`annotate.py` and `parts.py verify` both exit 0. If you cannot start subagents, do
the blind pass yourself one chunk at a time before rereading that chunk's part,
and say in the delivery that the verification was not independent.

### 10. Deliver

Write the PDF beside the paper as `<name>_annotated.pdf`, and each explainer as
`<name>_walkthrough_<id>.pdf`, when a folder is connected; otherwise to the
outputs directory. Reply briefly: counts by note kind; the coverage table
condensed by section (gutter notes and key points per page, highlighted share);
each explainer, its goal, and its length; the critique domain and any
`flaw`/`concern` verdicts (with page); any objects you flagged; any `thin` pages;
and the link-check counts (opened / search only / expected).
