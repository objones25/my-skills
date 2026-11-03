# Explainers (the walkthrough PDFs)

The orchestrator reads "Planning" at step 3. Each explainer worker reads this whole file, the docstring of
`scripts/walkthrough.py`, and `references/example-explainer.json` (the reference example) at step 6.

## Contents
- What an explainer is for
- Planning: which explainers to write
- Writing: the protocol
- Reader review
- What not to do
- Why these rules

## What an explainer is for

An explainer makes one hard idea understandable: an algorithm, a mechanism, or a central result and how its
proof works. It does what margin notes cannot. It gives the big picture first, puts the notation in one place,
draws the mechanism, works one small example by hand, outlines the proof, and breaks down only the steps that
are both load-bearing and hard to see.

It is not a second copy of the paper. It does not walk the section passage by passage, does not cover every
point on every page, and does not repeat what the margin notes say. Errors, slips, and critique stay in the
margins.

A good explainer is short because it is selective: typically 1,500 to 2,400 words, 4 or 5 pages, about 12
minutes, against 16 to 27 pages under the old rules. Most of the words past 1,500 go to defining terms for a
reader new to the field, which reviews showed cannot be skipped; spend them there, not on restating the paper.
`example-explainer.json` (SimGolf) is the model; read it before writing.

## Planning: which explainers to write

1. **Record the reader's goal** in `map.json` as `reader_goal`. Use the user's words if they said what they
   want from the paper. Otherwise use this default: "understand each algorithm or central mechanism
   conceptually, then an outline of how its main proof works: notation clarified and the most complex steps
   broken down; concise."
2. **One explainer per algorithm or central idea the goal names.** This is usually 1 to 3 for a paper, never
   one per dense section. A proof appendix gets no explainer of its own: its outline and hard steps go inside
   the explainer for the result it proves. Dense pages with no explainer are handled by margin notes.
3. **Give each explainer a goal**: what the reader can do after reading it, for example "explain what
   SimGolf does each iteration, why resets get around the double-sampling bias, and the four steps of the proof".
   The reviewer checks the explainer against this goal.
4. More than 4 explainers needs `many_explainers_because` in the map (a survey of several algorithms, say).
   `parts.py chunks` enforces this.

Chunk workers do not route key points to explainers. A point whose idea an explainer covers may use the
decision `walkthrough:<id>` with a `why`, but nothing has to be quoted there.

## Writing: the protocol

Work in this order. Steps A to C happen before any prose is written.

**A. Read and decide.** Read the result's pages and its proof pages as images. Write down, for yourself:
1. The idea in at most three sentences: what the method does and why it works.
2. The mechanism in 2 to 7 steps, each tied to an algorithm line or page.
3. The proof as a chain of 2 to 6 steps, each with the lemma or page where it lives.
4. For each proof step, two questions. *Load-bearing*: does the conclusion fail without it, or is it the
   paper's new idea? *Non-obvious*: does it need an inference the text does not state, a technique this
   reader may not know, or several pieces held in mind at once? A step that is both is a **hard step**.
   Break down at most 3. Every other step gets its one line in the outline. Algebra, union bounds,
   standard concentration used the standard way, and constant bookkeeping are routine.

**B. Choose the example.** Find the smallest concrete case that shows the key insight: the reason the method
works, or the failure it avoids. Use numbers the reader can check by hand (2 states, 2 candidates, K = 8).
Check every number by running code and record the command in `example.verify`. The code is never printed.
Tie the example back to the general statement in its last sentence.

**C. Notation.** List only the symbols the explainer uses, at most 16. Define each in plain words, with what
it means as well as how it is written.

**D. Write the sections** in the docstring's order:
1. **The idea** (40 to 170 words). The whole story in one paragraph. Every term in it is defined in the
   paragraph or is ordinary vocabulary for this reader.
2. **Notation.**
3. **The mechanism**: a diagram (2 to 6 boxes; highlight the novel step), the steps, and why it works.
4. **The example**, plus at most one or two check questions that ask the reader to do something with it.
5. **The proof outline** (if the goal asks for proofs): first the theorem itself, by its number, in plain
   words (what is assumed, what is guaranteed, and the bound), then a chain diagram with the hard steps
   highlighted, one line per step, and how the steps combine into the bound. If the outline's arithmetic
   gives a different constant from the theorem's, say why in one clause.
6. **Hard steps**: 2 to 5 moves each, one inference per move, and a check question whose answer asks for the
   inference itself. Simplify when it helps (a tabular version of a coverability argument, say), and say
   that you did.
7. **Safe to skim**: what on these pages the reader can pass over, and why.

Rules while writing:
- Explain why a step works, not what the paper says. The reader has the paper; never paraphrase it passage
  by passage.
- Define each term where it is first used, or in the notation table. Background this reader already has gets
  at most a notation row, never a section. This includes the field's basic vocabulary when the reader is new
  to the field (policy, return, Q-function), other algorithms you compare against (one clause on what each
  does), the paper's named assumptions or setups, and any named inequality or lemma whose content a move uses.
- State the condition a claim needs. "Each step is a convex problem" needs "when the class is linear";
  "the bound holds" needs the parameter range it holds for.
- Number references must be unambiguous: hard-step moves are "move 2", outline steps are "outline step 2".
- A loop in the mechanism goes in the diagram (`"loop": [from, to]`), not only in its caption.
- No errata, no critique, no production vocabulary (chunk, part file, annotator, flagged). The build rejects
  them.
- No printed code, no em dashes.

## Reader review

Every explainer is reviewed by a fresh subagent that did not write it, before delivery. The review is
required: `walkthrough.py --review` and `annotate.py` both check it. Brief:

> Read `out/<file>.pdf` (render the pages and look at them) as a reader who knows probability and linear
> algebra but not this paper's field (or as the reader the user described). Open the paper's pages if the explainer sends you there, or to check a
> claim or a piece of arithmetic you doubt. Write `work/review_<id>.json`:
> `{"spec_sha": "{sha}",
> "main_idea_after_page_1": "<state the main idea in your own words, from page 1 only>",
> "undefined_terms": ["<every term or symbol used before it is explained>"],
> "restates_paper": ["<any part that paraphrases the paper instead of explaining it>"],
> "unclear": ["<any step you could not follow or claim that is wrong, with where; any layout problem>"],
> "goal_met": "<yes/partly/no, tested against this goal: {goal}>",
> "verdict": "pass" or "revise"}`.
> Use "pass" only if the main idea is clear from page 1, `undefined_terms` is empty, the goal is met, and
> you found no claim that is wrong.
> A term is undefined if this reader would not know it and it is not explained at or before first use (the
> field's vocabulary, named lemmas, the paper's own names); standard math this reader knows (Cauchy-Schwarz,
> Hoeffding, union bounds, martingale concentration) does not count. Hard steps are workable sketches, not
> full proofs: judge whether each move's inference can be followed and where each term of the bound comes
> from, not whether every constant matches the paper. Errors in the paper itself are marked in the margins;
> the explainer may state a step conditionally ("if β ≤ 1") without repairing the paper.

`{sha}` is the spec hash the writer's last build printed ("spec 3f9c0a..."). The build rejects a review whose
hash does not match the current spec, so an explainer edited after its review is reviewed again.

The writer fixes what the review lists and rebuilds. A fresh reviewer (not the one who wrote the last review,
which now knows what the text means) reviews the new build, until the verdict is "pass". Reviews vary: a
later reviewer may find what an earlier one missed, and that is the point. Check each claimed error against
the paper before fixing it. If a third review still says "revise", read it yourself and fix what is real before
the next round, and tell the user if the reviews are not converging. A "pass" may still list
minor items; clarity nits after a pass can stay. A wrong claim is never a nit: fix it and review again.

## What not to do

These rules come from an audit of ten walkthroughs built under the old rules. Those walkthroughs were 16 to 27
pages each, about 80,000 words for a 31,800-word paper, and the reader found them unhelpful.
- Quoting and unpacking every passage. That made them 2.5 times longer than the source and mostly restatement.
- Teaching every prerequisite first. The main idea arrived 5 to 15 pages in.
- Repeating the margin notes. 12 of 12 and 14 of 14 error notes reappeared.
- Printing demo code. It took 16% to 31% of the words, and many demos showed paper errors, not ideas.
- Treating every step equally. A routine lemma got 443 words while the key idea got 167.
- Using terms before defining them. There were 13 to 23 such terms in the first four pages.
- No diagrams.

## Why these rules

| Rule | Evidence (details and sources in `evidence.md`) |
|---|---|
| Select; do not paraphrase every passage | Redundancy effect; line-by-line proof commentary (e-Proofs) did no better at once and worse after two weeks |
| Short, big picture first | Summaries a fifth the length matched or beat full chapters; pre-training and advance organizers help |
| Break down only hard, load-bearing steps | Expertise reversal: support that helps novices costs readers with background; element interactivity |
| Check questions at the hard steps | Self-explanation beats provided explanation (g = .35); for proofs d = 0.95 |
| One hand-sized example, tied back | Concreteness fading; "illustrating with examples" is one dimension of proof comprehension |
| Diagram the mechanism and the proof chain | Multimedia and signaling effects; holistic proof understanding is about modules and how they link |
| Reader review against a goal | Learners' preference does not track learning, so a build gate on form is not enough |
