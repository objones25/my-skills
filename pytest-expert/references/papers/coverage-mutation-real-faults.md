# Coverage, mutation score, and real faults: three papers

A combined digest, because the three papers answer one question in sequence: which adequacy number tells you whether a suite finds bugs?

| Tag | Paper | How read |
|---|---|---|
| [Ino14] | Laura Inozemtseva, Reid Holmes (Waterloo). *Coverage Is Not Strongly Correlated with Test Suite Effectiveness.* ICSE 2014 | Full author PDF, cs.ubc.ca/~rtholmes/papers/icse_2014_inozemtseva.pdf |
| [Just14] | René Just, Darioush Jalali, Laura Inozemtseva, Michael D. Ernst, Reid Holmes, Gordon Fraser. *Are Mutants a Valid Substitute for Real Faults in Software Testing?* FSE 2014 | Full author PDF, cs.ubc.ca/~rtholmes/papers/fse_2014_just.pdf (the umass.edu link the skill cites now returns 404) |
| [Chen20] | Yiqun T. Chen, Rahul Gopinath, Anita Tadakamalla, Michael D. Ernst, Reid Holmes, Gordon Fraser, Paul Ammann, René Just. *Revisiting the Relationship Between Fault Detection, Test Adequacy Criteria, and Test Set Size.* ASE 2020. DOI 10.1145/3324884.3416667 | Full author PDF, cs.ubc.ca/~rtholmes/papers/ase_2020_chen.pdf |

The 2026 replication on LLM-written tests has its own digest, `llm-tests-proxy-metrics.md`.

## The idea

[Ino14] generated 31,000 random test suites for five large Java programs and found that once you hold the number of tests fixed, coverage predicts mutant-killing ability only weakly to moderately, and stronger coverage types do not help [Ino14 Abstract]. But [Ino14] used mutants as the stand-in for faults. [Just14] checked that stand-in against 357 real, developer-fixed faults: mutant detection correlates with real-fault detection even with coverage held constant, and 73% of real faults are coupled to common mutants [Just14 Abstract, §1]. [Chen20] then argued that the "hold suite size fixed" design used by [Ino14] and by Papadakis et al. (ICSE 2018) is itself flawed, and that when criteria are compared fairly, mutation-adequate test sets detect more faults than coverage-adequate ones of the same size [Chen20 Abstract, §6].

## Terms

| Term | Meaning |
|---|---|
| Test suite effectiveness | Ability to detect faults. Unknowable directly, so measured by a proxy. |
| Mutation score | Killed mutants divided by (non-equivalent) mutants. |
| Normalized effectiveness | [Ino14]: killed mutants divided by mutants on covered lines only [§3]. |
| Equivalent mutant | A mutant no input can distinguish from the original. [Ino14] treated any mutant the full developer suite missed as equivalent [§6.1]. |
| Kendall τ | Rank correlation; [Ino14] reads 0.4-0.7 as moderate, 0.7-0.9 high (Guilford scale) [§4]. |
| Coupling | A real fault is coupled to a mutant if a test that detects the fault also kills a mutant the old tests missed [Just14 §2]. |
| Triggering test | The developer test added with a bug fix that fails on the buggy version [Just14 §2]. |
| Statement / decision / modified condition coverage | Lines run; branch outcomes run; each condition shown to independently affect its decision. |
| SizeFixed / SizeRandom | [Chen20]'s names for sampling test sets from a pool at a fixed size (stratified) or at a random size [§3]. |
| Probabilistic coupling | [Chen20]: for a fault and a test goal, the probability that a randomly chosen test satisfying the goal also detects the fault [§6.5]. |

## The study designs

**[Ino14].** Five Java systems up to 724,000 lines (Apache POI, Closure, HSQLDB, JFreeChart, Joda Time). From each developer suite, 1,000 random sub-suites at each of several fixed sizes, 31,000 in all. Measure statement, decision and modified condition coverage; measure effectiveness with mutants (raw and normalized); correlate with and without fixing size [Ino14 §3].

**[Just14].** 357 reproducible real faults from Chart, Closure, Math, Time, Lang (the start of Defects4J), 321,000 lines, about 230,000 mutants from the Major tool. For each fault, compare a test suite that passes on the buggy version with one that adds the triggering test, controlling for coverage by only counting mutants in code both suites cover. Also generate suites with EvoSuite and Randoop and correlate their mutation scores with real-fault detection [Just14 §2, Table 1].

**[Chen20].** Review the designs of [Ino14], [Just14], Papadakis et al. 2018 and others; show statistically why fixing size via stratification misleads; propose comparing criteria by building test sets the way a developer would (add a test only if it satisfies a new goal) and comparing at equal size to a random baseline. Data: the 231 Defects4J faults and test pools used by Papadakis et al., Major for mutants [Chen20 §2-§6].

## Key results

- [Ino14] Size vs effectiveness: moderate to very high correlation [Ino14 Answer 1, §4].
- [Ino14] Coverage vs effectiveness, size ignored: moderate to high for many programs [Answer 2]. Size controlled: typically low to moderate; stronger with raw than normalized effectiveness; coverage type made little difference [Answer 3].
- [Ino14] Project spread: Joda Time 0.80-0.85 with size ignored, about zero with size controlled; Apache POI raw-effectiveness correlation 0.94 ignoring size, 0.46-0.85 controlling it [§4].
- [Ino14] Up to 35% of mutants classified equivalent, per program [§6.1, Table 2].
- [Just14] Mutation score rose for 362 of 480 suite pairs (75%); statement coverage rose for only 222 (46%). Where coverage did not rise, mutation score still rose for 153 of 258 (59%) [Just14 §3.1].
- [Just14] 73% of real faults coupled to mutants; 95 of 357 (27%) not. Of those 95: 25 need a stronger operator, 7 a missing operator, 63 have no plausible operator [§3.2, Table 4]. The paper's summary: 10% need new or stronger operators, 17% are inherently not coupled [§1].
- [Just14] Statement deletion was the operator most often needing strengthening (12 of the 25) [§3.2].
- [Just14] Mutation score vs real-fault detection: positive, usually moderate or strong, and significantly stronger than statement coverage vs real-fault detection [§1, §3.3, Fig. 8]. With coverage controlled, the suite containing the triggering test had a significantly higher mutation score for 21/22 (Chart), 27/32 (Closure), 76/80 (Math), 18/18 (Time), 40/42 (Lang) faults [Table 5].
- [Chen20] Test set size is "neither a confounding variable ... nor an independent variable that should be experimentally manipulated" [Abstract]. Size is a consequence of the test objective, not a cause.
- [Chen20] Papadakis et al. (as summarized by [Chen20]) found mutant-fault correlation moderate to strong ignoring size and negligible to weak with size fixed; [Chen20] attributes the drop to the SizeFixed design, not to mutation [§2.4, §5].
- [Chen20] At equal size, selecting tests by coverage gives fault-detection probability as high as or higher than selecting by mutation, "at least until coverage is satisfied"; both beat random. Mutation-based selection is "most effective when employed after coverage has exhausted its usefulness" [§6.4, Fig. 5; §8]. (Corrected 2026-11-09: an earlier version of this digest had the comparison reversed.) When reducing a fault-detecting test set, mutation-based reduction mostly keeps the fault-detecting test; coverage-based reduction often loses it [§6.4, Fig. 7].

## What it means for a pytest suite

- Keep coverage as a floor, never a target. All three papers support hard gate 26-27: coverage finds untested code, but at fixed suite size it says little about whether tests check anything [Ino14 Answer 3].
- Mutation is the better signal to optimize, which is why workflow step 5 (`mutate_check.py`) exists. [Just14] and [Chen20] are the evidence; cite [Chen20] alongside [Just14] since it answers the "but size confounds it" objection.
- Statement-deletion mutants matter: they were the most common "weak operator" gap in [Just14] and the most fault-coupled at Google (`petrovic-mutation-practices.md`). Add a delete-statement operator to `mutate_check.py`.
- If you prune a suite, prune by mutation score, not by "fewest tests with equal coverage": mutation-based reduction mostly kept the fault-detecting test, coverage-based reduction often lost it [Chen20 §6.4].
- About a sixth of real faults have no mutant analogue [Just14 §1]. A 100% mutation score is not proof of correctness; property tests and integration tests cover a different slice.

## Threats to validity / caveats

- All Java, mostly libraries; Python behavior may differ.
- [Ino14] effectiveness rests on mutants, and its "anything the full suite misses is equivalent" rule can bias which mutants count [§6.1]. Correlational only.
- [Just14] faults are those that could be reproduced and isolated in five well-tested projects; coupling is to mutants from one tool (Major).
- [Chen20] uses a 231-fault subset of Defects4J for comparability, and the "add a test if it satisfies a new goal" procedure is one model of how developers work.

## Disagreements

- [Ino14] vs [Just14] are not in conflict: one says coverage is a weak proxy, the other says mutation is a decent proxy.
- [Ino14] and Papadakis et al. 2018 vs [Chen20]: the size-control design. [Chen20] (which shares authors with [Ino14]) argues stratifying by size is unrealistic and produces misleadingly weak correlations. The skill's current wording ("only a low to moderate correlation ... once suite size is controlled for") is a faithful quote of [Ino14] but should be paired with [Chen20]'s critique.
- [Zhao26] (`llm-tests-proxy-metrics.md`) finds that for LLM-generated tests suite size barely matters, and coverage and mutation track real-bug detection when comparing models on bug-free code, but not when the code under test is already buggy.
