# Judging sources

Which sources may support a verdict, and how to tell. The tier ladder is an adaptation, assembled from the sources cited on each line; no single study defines it.

## Judge a source from outside first

Before reading an unfamiliar source closely, search for what others say about its author, publisher or venue. Professional fact-checkers "read laterally, leaving a site after a quick scan and opening up new browser tabs in order to judge the credibility of the original site", while checklists that inspect only the page "create a false sense of security" (Wineburg & McGrew 2019). A polished design, a `.org` domain, an "about" page and a logo tell you nothing.

## Tiers

| Tier | What | Use |
|---|---|---|
| 1 | The code itself (run or inspected), official specs and the tool's own docs and changelog for claims about that tool; systematic reviews and meta-analyses for empirical claims | Strongest support |
| 2 | Peer-reviewed papers in established venues (author PDFs matching the published version are fine) | Strong support |
| 3 | Preprints, labelled as such. Check whether a published version exists and whether its conclusions changed (Brierley et al. 2022: conclusions changed in 7.2% of non-COVID and 17.2% of COVID preprint abstracts) | Support, one level weaker |
| 4 | Signed, dated practitioner writing: maintainers' issues, RFCs and design docs, conference talks, an engineer's own blog. Rate it on Garousi et al. 2019 (Table 7): authority of the author, stated method, objectivity, date, links to other sources | Weak support; never outranks a contradicting tier 1-2 source |
| 5 | Q&A answers, anonymous wikis, forum posts | Leads to better sources only; never cited as evidence (AS017) |
| 6 | SEO listicles, content farms, AI-generated summaries, predatory venues (Grudniewicz et al. 2019), marketing pages | Excluded; cannot be cited (AS017) |

For tool behavior, the vendor's own official docs are tier 1 even though the vendor sells the tool, because they are the primary source. Code still outranks them.

## Red flags: lower the tier, or exclude

- The author or publisher sells the product being recommended or compared. Garousi's objectivity item asks: "Is there vested interest? E.g., a tool comparison by authors that are working for particular tool vendor."
- No named author, no date, or hidden funding.
- Superlatives with no method or numbers; benchmarks nobody can rerun.
- Conclusions that go further than the data.
- A venue with aggressive solicitation, false claims about indexing or peer review, or an opaque editorial board.

## Conflicts of interest are separate from method quality

Record any stake in `interest` on the evidence entry, even for a well-designed study. Industry sponsorship biases results in a way "that cannot be explained by standard 'Risk of bias' assessments" (Lundh et al. 2017, Cochrane), so good methods do not cancel a conflict.

## Primary over secondary, and more than one source

Cite the primary source, not a summary of it. Check key claims against more than one named source where possible (IFCN Code of Principles). When a verdict rests on a single source, say so in the note; for empirical claims and recommendations that limits strength to C or below unless the source is tier 1 (AS027 enforces this).

## Industrial reports

A company describing its own practice (for example a mutation-testing paper from the company that runs the tool) is a self-report: tier 2 if peer reviewed, but note the interest, and treat its effect sizes as indirect for other settings.
