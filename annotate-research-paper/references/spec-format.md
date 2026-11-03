# File formats

The orchestrator writes `map.json` and `globals.json`; each chunk worker writes one
part file; each verifier writes one verify file; `scripts/parts.py merge` combines
map, parts, and globals into `spec.json`, which `scripts/annotate.py` builds.
Never edit `spec.json` by hand: the next merge overwrites it.

## Contents
- map.json (orchestrator, step 3)
- Part files (chunk workers, step 4)
- globals.json (orchestrator, steps 7 and 9)
- Verify files (verifiers, step 9)
- The merged spec, field by field
- Note text

## map.json (orchestrator, step 3)

```json
{
  "title": "...", "summary": "2-4 sentences: the claim and where the contribution is", "domain": "security",
  "reader_goal": "understand each algorithm conceptually, then an outline of how its main proof works; concise",
  "sections": [
    {"id": "intro", "title": "1 Introduction", "pages": [1, 2]},
    {"id": "simgolf", "title": "3 SimGolf", "pages": [4, 7], "walkthrough": "simgolf"},
    {"id": "simgolf-proof", "title": "C-D Proofs for SimGolf", "pages": [27, 30], "walkthrough": "simgolf"},
    {"id": "refs", "title": "References", "pages": [15, 19], "references": true}
  ],
  "walkthroughs": [
    {"id": "simgolf", "title": "SimGolf, explained", "pages": [4, 7], "proof_pages": [27, 30],
     "anchor": "3.1 Algorithm",
     "goal": "Explain what SimGolf does each iteration, why resets get around the double-sampling bias, and the four steps of its proof.",
     "why": "SimGolf is the paper's first algorithm, and its one new idea (resets) is easy to miss in the margins."}
  ],
  "glossary": [{"term": "covert storage channel", "page": 9}]
}
```

- `sections`: every page belongs to one (a boundary page may belong to two). A
  section an explainer covers may name it with `"walkthrough": "<id>"`
  (optional); a reference list carries `"references": true`. Pages after the
  reference list that sit in an ordinary section (an appendix) are reviewed like
  any other.
- `reader_goal`: what the reader wants from the paper (the user's words, or the
  default in `references/walkthroughs.md`). Required when any explainer is planned.
- `walkthroughs`: the planned explainers, usually 1 to 3 (more needs
  `many_explainers_because`). Each is `{id, title, pages, goal, why, anchor}`,
  optionally `proof_pages`; `goal` (10+ words) says what the reader can do after
  reading it; `anchor` is a verbatim phrase on its first page, where the pointer
  note goes.
- `glossary`: where the paper first defines each coinage, so only that page's
  chunk defines it.
- `pages`: only for a partial request, `[first, last]`; chunking, merging, and the
  build follow it.

`parts.py chunks` checks all of this and stops on a gap.

## Part files (chunk workers, step 4)

`work/parts/part_<a>-<b>.json`:

- `pages`: `[a, b]`, the chunk.
- `sections`: copied from the map unchanged (the merge rejects any edit).
- `annotations`, `page_review`, `skip`, `audit`: for this chunk's pages only, in
  the formats below.
- `critique_notes`: `[{page, claim, evidence, concern}]`, anything the critique
  should weigh.

`scripts/annotate.py paper.pdf <part> <out.pdf> --part` is the chunk's gate.

## globals.json (orchestrator, steps 7 and 9)

- `guide`: `{summary, route, skim}` (see `references/guide-and-critique.md`).
- `critique`: `{domain, items}` (same file).
- `cited_works`: see `references/links.md`.
- `walkthroughs`: the map's explainer entries plus `spec` (the explainer JSON's
  file name), `file` (its PDF's file name), and `review` (the passing reader
  review's file name; default `review_<id>.json` beside the spec). The build checks that the review passes
  and that its `spec_sha` matches the spec as it is now.
- `verify_rejections`: `[{page, quote, why}]` for a verifier point that is wrong;
  `why` is 6+ words.

## Verify files (verifiers, step 9)

`work/verify/part_<a>-<b>.json`: `{"pages": [a, b], "points": [{"page", "quote",
"what"}]}`, with each `quote` 4-20 words copied from `lines.txt` and `what` one of
the labels in `references/annotating.md` ("Page-by-page review").

## The merged spec, field by field

```json
{
  "title": "Dense Passage Retrieval for Open-Domain QA",
  "guide": {
    "summary": "2-4 sentences: what the paper claims and where the contribution is.",
    "route": [
      {"what": "Abstract + Introduction", "page": 1, "read": "pass 1", "why": "the claim and the gap."},
      {"what": "Section 3, the model", "page": 3, "read": "pass 2", "why": "the method and its training."},
      {"what": "Tables 2-3, results", "page": 6, "read": "pass 2", "why": "where the gains are and are not."}
    ],
    "skim": "Optional: what to read lightly and why."
  },
  "annotations": [
    {"page": 3, "quote": "dual-encoder", "kind": "key", "title": "Two encoders, one dot product",
     "note": "<b>Why it works.</b> Passages are encoded once, offline, so search becomes a maximum-inner-product lookup ... <b>Trade-off.</b> ... <b>Does.</b> ...",
     "short": "Encoding passages offline turns retrieval into a fast inner-product lookup; the cost is ...",
     "links": [{"type": "course", "url": "https://...", "title": "...", "status": "opened",
                "seen": "Lecture 9: Dense retrieval", "via": "WebFetch",
                "fit": "45-minute university lecture on exactly this architecture; standalone.",
                "chosen_over": ["Vendor blog post (sells a vector DB; skips training)",
                                "Conference talk (assumes the paper is already read)"]}]},
    {"page": 6, "quote": "Table 2", "kind": "key", "title": "Table 2", "fig_kind": "data",
     "note": "<b>Metric.</b> top-20 retrieval accuracy (higher is better). <b>Baseline.</b> BM25 on the same questions. <b>Takeaway.</b> ... <b>Caveat.</b> single runs ...",
     "short": "..."},
    {"page": 4, "lines": "L10-L18", "mark": "none", "kind": "intuition", "title": "Eq. (2) loss",
     "note": "<b>Why this form.</b> ... <b>Reads as.</b> ..."},
    {"page": 6, "lines": "L4", "mark": "none", "kind": "check",
     "note": "Why train with in-batch negatives instead of random ones? <b>Hint.</b> think about cost per batch.",
     "answer": "Because every other question's positive passage in the batch is a free negative: B questions give B-1 negatives each at no extra encoding cost, so the model sees many more negatives per step than sampling separately would allow."}
  ],
  "sections": [
    {"id": "intro", "title": "1 Introduction", "pages": [1, 2]},
    {"id": "model", "title": "3 Dense Passage Retriever", "pages": [3, 4], "walkthrough": "training"},
    {"id": "results", "title": "5-6 Experiments", "pages": [5, 7]}
  ],
  "skip": [{"item": "Figure 1", "page": 2, "why": "schematic of the pipeline; the model note covers it."}],
  "audit": [
    {"item": "Table 2", "page": 6, "result": "ok", "how": "top-20 numbers in the text match the table; same test sets across systems"},
    {"item": "Eq. (2)", "page": 4, "result": "ok", "how": "one in-batch example: softmax over 1 positive + B-1 negatives; shapes check"}
  ],
  "critique": {
    "domain": "ir",
    "items": [
      {"check": "Are the baselines tuned, at least at the collection's known level?", "verdict": "ok",
       "where": "section 5", "evidence": "section 5 reports standard BM25 settings and compares against published numbers",
       "note": "Uses a standard BM25 and reports it as a strong baseline it sometimes loses to."},
      {"check": "Do gains survive out of domain?", "verdict": "concern", "where": "Table 3",
       "claim": "abstract: 'outperforms a strong BM25 system'", "note": "Only in-domain sets are tested; the claim is stated without that scope."},
      {"check": "Multiple runs and variance reported?", "verdict": "concern", "where": "section 6",
       "claim": "Table 2's margins", "note": "Single-run numbers with no variance, so small margins are not separable from noise."},
      {"check": "Claim scope matched to the data?", "verdict": "ok", "where": "section 6",
       "evidence": "the evaluation sets are the QA sets the claim names", "note": "..."}
    ]
  },
  "cited_works": [
    {"type": "paper", "cite": "Author et al. Title. Venue Year. arXiv:XXXX.XXXXX.", "url": "https://arxiv.org/abs/XXXX.XXXXX",
     "where": "ref [12]", "why": "the method this paper extends", "status": "expected",
     "basis": "arXiv ID printed in the paper's reference list; not opened"}
  ],
  "page_review": [ {"page": 1, "section": "intro", "points": [ ... ], "not_key": [ ... ]}, "... one per page" ],
  "walkthroughs": [
    {"id": "training", "title": "Training with in-batch negatives", "pages": [3, 4],
     "spec": "walkthrough_training.json", "file": "<name>_walkthrough_training.pdf",
     "review": "review_training.json",
     "goal": "Explain how the dual encoder is trained with in-batch negatives and why that makes training cheap.",
     "why": "The training trick is the paper's contribution and needs the loss and the batch picture together.",
     "anchor": "3.2 Training"}
  ]
}
```

Anchors (one per annotation): `"lines": "L7-L12"` from `lines.txt` (most robust;
use for equations and code); `"quote": "exact phrase"` for a tight highlight (add
`"within": "L5-L9"` if it repeats; a caption like `Table 2` is a good anchor);
`"rect": [x0, y0, x1, y1]` for scanned pages. `mark`: `highlight`, `bar`, or
`none`; `key`/`subtle` default to a highlight. Never put line IDs ("L18") in note
text.

## Note text

Note text is short HTML: `<b>`, `<i>`, `<sub>`, `<sup>`, `<br>`, `<code>` and
Unicode math (ℙ, 𝔼, ∈, →, ≤, ∑, γ, ·). LaTeX does not render.
