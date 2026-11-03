# Paper digests for pytest-expert

One file per paper or tight paper group, written 2026-11-02 from the PDFs (arXiv, author pages, or open-access proceedings). Every number carries a location tag like [Petrovic21 §III-C] or [ACH Table 8]. "Read" in a header says how much of the paper was read; claims are limited to that. Industrial papers (Google, Meta) are self-reports by the teams that built the tools.

| File | Paper | Read it when |
|---|---|---|
| `petrovic-mutation-practices.md` | Petrović, Ivanković, Fraser, Just, *Does mutation testing improve testing practices?* ICSE 2021 | justifying mutation testing as a practice; why survivors should be shown as individual test goals; statement-deletion operator |
| `google-mutation-at-scale.md` | Petrović et al., *Practical Mutation Testing at Scale: A view from Google* (arXiv 2021, later TSE) | making `mutate_check.py` output actionable: diff-only, arid-line suppression, one mutant per line, "unproductive" mutants |
| `google-coverage.md` | Ivanković, Petrović, Just, Fraser, *Code Coverage at Google*, FSE 2019 | coverage thresholds, diff coverage, why instrumentation causes timeouts and flakiness |
| `coverage-mutation-real-faults.md` | Inozemtseva & Holmes ICSE 2014; Just et al. FSE 2014; Chen et al. ASE 2020 | the evidence for "coverage is a floor, mutation is the signal", and the suite-size controversy |
| `llm-tests-proxy-metrics.md` | Zhao, Zhou, Cohen, *Do Coverage and Mutation Scores of LLM-Generated Test Suites Correlate with Their Effectiveness?* ISSTA 2026 | judging agent-written tests; why coverage fails when the code under test is already buggy |
| `meta-ach.md` | Foster et al., *Mutation-Guided LLM-based Test Generation at Meta* (ACH), FSE Companion 2025 | feeding surviving mutants to the agent as test prompts; equivalent-mutant filtering |
| `meta-testgen-llm.md` | Alshahwan et al., *Automated Unit Test Improvement using LLMs at Meta*, FSE 2024 | filtering LLM-written tests (build, pass 5x, coverage) and why that is not enough |
| `coverup.md` | Altmayer Pizzorno & Berger, *CoverUp*, FSE 2025 | coverage-guided generation loop for pytest; isolation and repetition of generated tests |
| `impossiblebench.md` | Zhong, Raghunathan, Carlini, *ImpossibleBench*, arXiv 2025 | designing the Stop hook against test deletion, special-casing and `__eq__` tricks |
| `escalation-channels.md` | Gomez, *Can escalation channels redirect reward hacking toward defect disclosure?* arXiv 2026 | giving the agent a sanctioned "this test is wrong" exit |
| `pbt-in-practice.md` | Goldstein et al., *Property-Based Testing in Practice*, ICSE 2024 | which property patterns pay off; checking that generators hit the interesting inputs |
| `agentic-pbt.md` | Maaz, DeVoe, Hatfield-Dodds, Carlini, *Agentic Property-Based Testing*, arXiv 2025 | how an agent should derive and triage Hypothesis properties |
| `hypothesis-reducer.md` | MacIver & Donaldson, *Test-Case Reduction via Test-Case Generation*, ECOOP 2020 | why Hypothesis needs no custom shrinkers; strategy design that shrinks well |
| `gruber-flaky-python.md` | Gruber, Lukasczyk, Kroiß, Fraser, *An Empirical Study of Flaky Tests in Python*, ICST 2021 | Python flakiness causes and rerun counts (already cited; this checks the citation) |
| `flaky-survey.md` | Parry et al., *A Survey of Flaky Tests* (TOSEM 2021) and *Systemic Flakiness* (arXiv 2025) | flaky-test causes beyond Python, repair patterns, grouping co-failing flaky tests |
| `pynose-test-smells.md` | Wang et al., *PyNose*, ASE 2021 | mapping test smells to audit codes; candidate new audit checks |

Every paper, with venue, how it was read, and what it supports, is listed in `references/sources.md`. The changes these papers drove are logged in `AUDIT.md`.

## Known caveats

- Not read: Papadakis et al. ICSE 2018 (author PDF link 404; its results are quoted only as summarized by Chen et al. 2020 and Zhao et al. 2026); Panichella et al. EMSE 2022 (not found in the research index; PyNose used instead); Petrović et al. ICSE-SEIP 2018 (the arXiv "Practical Mutation Testing at Scale" covers and extends it); Kochhar et al. (not needed once Chen 2020 was available).
- Abstract or introduction only, cited as such: arXiv:2609.25528 (Shaker on Python flaky tests, 2026) and arXiv:2610.34487 (multi-domain LLM PBT study, 2026).
- Version notes: `google-mutation-at-scale.md` uses arXiv v2, not the TSE version. The ACH arXiv PDF carries a DOI and running header left over from TestGen-LLM; do not cite that DOI for ACH. TestGen-LLM's abstract says "test cases" where §3.3 says "test classes" for the 75/57/25% figures.
- The Just et al. 2014 PDF URL the skill cites (people.cs.umass.edu/~rjust/...) now returns 404; working copies are at cs.ubc.ca/~rtholmes/papers/fse_2014_just.pdf and homes.cs.washington.edu/~mernst/pubs/mutation-effectiveness-fse2014.pdf (the latter seen in search results, not fetched).
- 2026 preprints (Zhao et al. has an ISSTA 2026 venue line; Gomez has none) are recent and unreplicated. Gomez is single-author with 9 problems.
- Most studies are Java or Kotlin. Python-specific evidence: Gruber 2021, CoverUp, Agentic PBT, MacIver & Donaldson, PyNose (unittest only).
