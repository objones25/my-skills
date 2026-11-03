# Links: concept resources and cited works

Read before adding any link: chunk workers for concept links on their pages, the orchestrator for cited works (SKILL.md, step 7).

## Contents
- Links: concepts and cited works
- Sourcing quick-reference

## Links: concepts and cited works

Two places:

- **`cited_works`**: the handful of references worth reading, rendered on the Key
  references page with a line saying exactly how each link was checked.
- **`links` on a note**: a concept link (docs, a survey, a CWE, a lecture, a
  video) for a term the reader may not know. Its tag reads `[paper]`, or
  `[paper · not opened]` when it was not opened.

### Record how each link was checked

| `status` | Meaning | Required |
|---|---|---|
| `opened` | A tool call this session returned the page. | `seen`: the exact title the tool showed. Optional `via` (tool). |
| `search` | You saw it only in a search-results listing. | `seen`: the title as listed. |
| `expected` | Built from a persistent identifier (arXiv ID, DOI-based open repository) and not opened. | `basis`: where the identifier came from. Not allowed for videos. |

`verified` is retired; the build rejects it. If the title you saw shares few words
with the citation, the build warns: you may have linked the wrong work. A tool can
return wrong content (cached or mis-resolved pages), so compare the title you saw
with what you expected, every time.

**`check_links.py` (optional, mechanical).** Where the environment allows direct
HTTP, run `python work/arp/scripts/check_links.py work/spec.json work/link_report.json`
and pass `--link-report work/link_report.json` to `annotate.py`. Links the script
opened are labeled "Opened by the build script" with the HTTP status and the page
title it read, so no part of that label depends on the model's word. Where the
network is policy-restricted the script reports the block and changes nothing.

### Choosing concept resources

The link is where the learning happens, so pick it as carefully as you write the
note. For every concept link:

1. **Standalone.** It must make sense on its own. No mid-series episodes ("Ep.
   02"), project logs, or case studies that assume earlier parts, unless the note
   says to start at part 1 and links it.
2. **Fit.** It teaches this concept at this reader's level and length: a 10-25
   minute explainer or a focused text section for a term; a full lecture only for
   a central method.
3. **Inspect the content, not the title or channel.** Read the description,
   chapter list, or transcript, or the course's notes. If you cannot tell what a
   video covers, prefer a text source you can read.
4. **Compare at least three candidates.** Never take the first result. Weigh
   authority (university course, standards body, official docs, an established
   educator, the original authors), the audience it is written for, quality
   signals (views and likes for videos, the course or venue for lectures), and
   production quality (audio, slides).
5. **Video and course links** must carry `fit` (one line on why this one teaches
   this concept to this reader, standalone) and `chosen_over` (at least two
   alternatives you compared and why each lost). The build rejects them otherwise.

Sourcing for papers is below, under Sourcing quick-reference (priority order,
per-venue open-text locations, bad-link patterns). Put the persistent id (arXiv, DOI) in the
citation text.

## Sourcing quick-reference

Prefer open full text, and record honestly how you checked each link (status opened / search / expected; see above). Priority for a
paper: (1) arXiv `arxiv.org/abs/<id>`; (2) the venue's open copy (USENIX, NDSS,
ACL Anthology `aclanthology.org/<ID>.pdf`, PMLR, NeurIPS proceedings, OpenReview
`pdf?id=` only if "Published as", IEEE S&P 2021+ in CSDL, ACM
`dl.acm.org/doi/pdf/10.1145/...`: ACM's own page confirms 1951-2000 is open;
reports that the whole library opened in January 2026 come from secondary
sources only, so check each paper; never `/doi/abs/`); (3) an author/lab copy you fetched and confirmed;
(4) a repository copy (PMC, institutional); (5) none: cite with the DOI as text,
no link. Put the arXiv id or DOI in the citation text.

The scripts reject: search-result pages, Google Scholar, URL shorteners,
researchgate/academia/scribd, `export.arxiv.org` and other arXiv mirrors,
sci-hub/libgen, `dl.acm.org/doi/abs`, `sciencedirect.com/.../abs/`, and non-https
links; they warn on likely-paywalled landing pages (ieeexplore `document`,
springer, sciencedirect `pii`, wiley, tandfonline, `doi.org` resolvers,
`dl.acm.org/doi/` landings). Confirm a video's title and channel via its oEmbed record or page; compare them with what you expected, because a tool can return a cached or wrong record. Reference rot is real (about a
quarter of a decade's web pages go dead; Pew 2024), so prefer persistent ids.
