# Sources

Where this skill's method comes from. Checked means how much was read: Full, Partial, Abstract only, or Installed. "Quote matched" means the quote this skill uses was string-matched with `scripts/quote_match.py` against a raw fetch by `scripts/fetch_source.py` on 2026-11-07. Rows marked "research pass" were read by a research subagent on 2026-11-07 and not re-fetched by the author; treat their figures as abstract-level.

## Search method

| Source | Date checked | Checked | Supports |
|---|---|---|---|
| Lefebvre et al., Cochrane Handbook ch. 4, current version: https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-04 | 2026-11-07 | Partial (§4.2 to 4.5); quotes matched | Two or more databases; concept blocks with OR/AND; stopping documented; strategies not re-typed |
| Rethlefsen et al., PRISMA-S, *Syst Rev* 10:39 (2021): https://pmc.ncbi.nlm.nih.gov/articles/PMC7839230/ | 2026-11-07 | Checklist; quote matched ("copied and pasted exactly as run") | Logging every query as run, with counts |
| Page et al., PRISMA 2020, *BMJ* 372:n71: https://pmc.ncbi.nlm.nih.gov/articles/PMC8005924/ | 2026-11-07 | Checklist, research pass | Full search strategies and dates reported |
| Kitchenham & Charters, EBSE-2007-01 v2.3 (2007): https://legacyfileshare.elsevier.com/promis_misc/525444systematicreviewsguide.pdf | 2026-11-07 | §2.2, §6.1 to 6.3; quotes matched (disconfirming evidence is §2.2, known studies §6.1.1) | Seeking disconfirming evidence; testing queries on known studies; fit evidence to the question |
| Wohlin, snowballing guidelines, EASE 2014: https://www.wohlin.eu/ease14.pdf | 2026-11-07 | §1 to 3; quote matched ("once no new papers are found") | Backward and forward snowballing; its stopping rule |
| Garousi, Felderer, Mäntylä, multivocal reviews in SE, *IST* 106 (2019): https://arxiv.org/pdf/1707.02553 | 2026-11-07 | Guidelines 8, 11, 12 and Table 7; quote matched ("Is there vested interest?") | Grey-literature quality criteria; effort-bounded stopping |
| Booth, stopping rules, *IJTAHC* 26(4) (2010), doi:10.1017/S0266462310000966 | 2026-11-07 | Abstract only, research pass | Stopping rules are little evaluated |

## Judging sources

| Source | Date checked | Checked | Supports |
|---|---|---|---|
| Wineburg & McGrew, "Lateral Reading and the Nature of Expertise", *TCR* 121(11) (2019), preprint: https://purl.stanford.edu/yk133ht8603 | 2026-11-07 | Full text extracted; quotes matched | Lateral reading; checklists give "a false sense of security"; click restraint |
| Lundh et al., industry sponsorship and research outcome, Cochrane MR000033.pub3 (2017), via Europe PMC (PMID 28207928) | 2026-11-07 | Abstract; quote matched | Sponsorship bias not explained by risk-of-bias tools |
| Grudniewicz et al., "Predatory journals: no definition, no defence", *Nature* 576:210 (2019) | 2026-11-07 | Full, research pass; one quote matched | Definition of predatory venues |
| Brierley et al., preprint vs published changes, *PLOS Biol* (2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC8806067/ | 2026-11-07 | Abstract; quote matched | 7.2% and 17.2% of abstracts' conclusions changed |
| IFCN Code of Principles, commitments: https://ifcncodeofprinciples.poynter.org/the-commitments | 2026-11-07 | Commitments section; quotes matched | Primary over secondary sources; more than one named source |

## Evidence strength

| Source | Date checked | Checked | Supports |
|---|---|---|---|
| Schünemann et al. (eds), GRADE Handbook (2013): https://gdt.gradepro.org/app/handbook/handbook.html | 2026-11-07 | About a third (ch. 5), research pass | Four certainty levels; factors that lower and raise certainty |
| Dybå & Dingsøyr, agile SR, *IST* 50 (2008), author PDF | 2026-11-07 | §3.5, Table 3, §5.2, research pass; a short phrase matched | GRADE applied to software engineering |
| Shepperd, Ajienka, Counsell, replication in SE, *IST* 99 (2018), author preprint | 2026-11-07 | Abstract and introduction, research pass; "8.64" matched | Internal replications confirm far more often than external |

## LLM citation and verification failures

| Source | Date checked | Checked | Supports |
|---|---|---|---|
| Walters & Wilder, *Sci Rep* 13 (2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10484980/ | 2026-11-07 | Abstract and results; quote matched | 55% (GPT-3.5) and 18% (GPT-4) of citations fabricated |
| Chelli et al., *J Med Internet Res* 26:e53164 (2024), via Europe PMC | 2026-11-07 | Abstract; quote matched | 39.6%, 28.6%, 91.4% hallucinated references |
| Wu et al., SourceCheckup, arXiv:2402.02008 | 2026-11-07 | Abstract; quote matched | 50 to 90% of LLM responses not fully supported by their cited sources; nearly half of GPT-4-with-RAG responses |
| Li et al., AttributionBench, arXiv:2402.15089 | 2026-11-07 | Abstract; quote matched | About 80% macro-F1 for attribution judgement |
| Min et al., FActScore, arXiv:2305.14251; Wei et al., SAFE, arXiv:2403.18802 | 2026-11-07 | Abstracts, research pass | Atomic facts make automatic fact scoring accurate |
| Dhuliawala et al., Chain-of-Verification, arXiv:2309.11495 | 2026-11-07 | Abstract, research pass | Answer verification questions independently of the draft |
| Sharma et al., sycophancy, arXiv:2310.13548; Panickssery et al., self-preference, arXiv:2404.13076 | 2026-11-07 | Abstracts, research pass | Why the reviewer is blind and ideally a different model |
| Huang et al., ICLR 2024, arXiv:2310.01798; Kamoi et al., TACL 2024, arXiv:2406.01297 | 2026-11-07 | Abstracts and parts of Huang §3, research pass | Self-correction needs reliable external feedback |

## Tools

| Source | Date checked | Checked | Supports |
|---|---|---|---|
| `pdftotext` (poppler), Homebrew build on macOS arm64 | 2026-11-07 | Installed; exercised by `tests/test_fetch_source.py` | PDF to text in `fetch_source.py` |
| `pytest-expert/scripts/mutate_check.py` | 2026-11-07 | Full; copied with its tests | `scripts/mutate_check.py` |

## Not verified

- No primary source was found for grading claims backed only by docs or source code; the A and B rungs for code and docs are this skill's adaptation.
- No study measures a verifier drifting toward a verdict it was shown in a fact-checking setting; the blind reviewer extrapolates from Sharma et al. and Chain-of-Verification.
- The stopping rule (2 empty rounds, 10 hits, 4 queries) is an adaptation; Booth (2010) found stopping rules little evaluated.
