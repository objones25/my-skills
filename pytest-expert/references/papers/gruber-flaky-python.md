# An Empirical Study of Flaky Tests in Python

Martin Gruber (BMW Group), Stephan Lukasczyk, Florian Kroiß, Gordon Fraser (Passau). ICST 2021. arXiv:2101.09077. Read in full from the arXiv PDF. Already cited by the skill (`flaky-tests.md`, `rules.md`); this digest checks and extends those citations. Tag [Gruber21].

## The idea

The first large flakiness study for Python: run the test suites of 22,352 PyPI projects 400 times each (200 in fixed order, 200 shuffled), find every test that both passed and failed, and classify why. Order dependence dominates, infrastructure problems come second, and the rest is mostly network and randomness. Finding flaky tests by rerunning takes far more runs than people use [Gruber21 Abstract, §III-§IV].

## Terms

| Term | Meaning |
|---|---|
| Flaky test | Passed at least once and failed at least once across the reruns [§III-D]. |
| Order-dependent (OD) | Outcome depends on which tests ran before it [§II-B]. |
| Victim / polluter | A victim passes alone but fails after a polluter test changes shared state [§II-B]. |
| Brittle / state-setter | A brittle fails alone and passes only after a state-setter test runs first [§II-B]. |
| Non-order-dependent (NOD) | Flaky even in a fixed order: network, randomness, time, I/O, concurrency and similar [§II-A]. |
| Infrastructure flakiness | Caused outside the project, such as pip failing to install a dependency and producing `ModuleNotFoundError`; new category in this paper [§I, §II-A]. |
| FlaPy | The authors' tool that runs suites in isolation and in many orders [§III-D]. |
| n(t, p) | Reruns needed so that test t's flakiness would show with probability p, from its observed failure rate: U(n) = 1 − (1 − P(fail))^n [§III-D3]. |

## The study design

1. **Dataset.** 22,352 projects sampled from the 284,112 on PyPI (as of January 2021); 876,186 test cases [§III-A].
2. **Execution.** 200 runs in a fixed order and 200 with project-level random order, each in isolation [§III-D1].
3. **Classification.** Trace each flaky test's execution and search for API calls characteristic of each category (Table IV: for example `requests` for network, `random` for randomness, `sleep` for async wait, `__hash__` and `set.__contains__` for unordered collections), then classify by hand; OD tests split into victims and brittles by running them alone [§III-C].
4. **Degree.** Estimate per-test failure probability and derive rerun counts [§III-D3].

## Key results

- 7,571 flaky tests (0.86% of tests) in 1,006 projects (4.5% of projects) [Gruber21 §IV-A].
- Stable projects contained flakiness more often (5.3%) than pre-alpha ones (3.4%) [§IV-A, Table V].
- Causes: order dependency 59% (4,461; 3,168 victims, 738 brittles, 555 unanalyzed), infrastructure 28% (2,158), NOD 13% (952) [§IV-B, Table VII].
- In a random sample of 100 NOD tests: network 42, randomness 37, I/O 7, time 4, async wait 3, concurrency 3, resource leak 2, test-case timeout 1, unordered collections 1 [Table VII, Table IX].
- Async wait and concurrency, leading causes in earlier Java studies, were rare here (3 each of 100) [Table IX].
- Checking whether one failure is flaky: for most NOD tests, 1 rerun gives 95% confidence; for OD tests, 3 [§IV-C (d)].
- Checking whether a passing test is free of flakiness: 170 reruns for 95% confidence (NOD); 31 random-order runs for OD [§IV-C (a)].
- Ten reruns find on average no more than 33% of NOD and 54% of OD flaky tests [§IV-C (b)].
- Finding 80% of NOD flakiness needs on average 110 reruns, or 472 for 95% confidence; for OD, 49 and 209 [§IV-C (c)].
- With 200 reruns, the authors estimate they found at least 61% of NOD flakiness at 95% confidence [§IV-C].
- Hypothesis-based tests can be flaky too; the paper shows one from `humansort` [§IV-B, Fig. 6].

## What it means for a pytest suite

- The skill's numbers are right: 876,186 tests, 22,352 projects, 7,571 flaky, 59/28/13% [`flaky-tests.md`]. Two refinements:
  - "about 170 reruns for 95% confidence that a passing test is not flaky" applies to NOD flakiness only; OD needs 31 random-order runs [§IV-C (a)]. `flake_hunt.py --runs 30` therefore matches the OD figure, which is the common case, but says little about NOD.
  - "Two runs prove almost nothing" is true for proving a test is clean, but one or three reruns are enough to tell whether a single observed failure was flaky [§IV-C (d)]. The flaky-tests reference can say both.
- Order dependence is the main Python cause, so pytest-randomly in every CI run (already in `tests.yml`) and the module-state audit (`MUTABLE_MODULE_STATE`) target the largest slice.
- The 28% infrastructure share argues for pinned, cached dependencies in CI (lockfiles, `pip install --require-hashes`) and for treating `ModuleNotFoundError` in CI as infrastructure, not a test failure.
- The trace keywords in Table IV are a cheap classifier. `flake_hunt.py` could tag a flaky test's likely cause by grepping its source and imports for `requests`, `socket`, `random`, `time.sleep`, `datetime.now`, iteration over `set`.

## Threats to validity / caveats

- Machines were similar, so platform-dependent flakiness was likely under-detected [§III-E].
- The paper calls its 22,352-project sample "only about 10%" of available PyPI projects (presumably those with usable tests; 22,352 of the 284,112 total is about 8%) [§III-A, §III-E].
- 400 reruns still miss some flakiness [§III-E, §IV-C].
- The NOD breakdown comes from a 100-test sample, so the per-category counts carry wide uncertainty.

## Disagreements

- With Java studies (Luo et al. 2014, Eck et al. 2019, as compared in Table IX and in Parry et al.'s survey) that rank async wait and concurrency first; in Python these were minor [Table IX]. See `flaky-survey.md`.
- A 2026 preprint (arXiv:2609.25528, Leal et al., "Evaluating Shaker for Flaky Test Detection in Python Projects"; abstract and introduction read only) reran 137 NOD flaky tests from Gruber21's ground truth 100 times each on independent hardware and found fewer than half reproduced as flaky at all, and those that did were mostly network and randomness. This does not contradict Gruber21's counts, but warns that a flaky label is tied to the environment where it was observed.
