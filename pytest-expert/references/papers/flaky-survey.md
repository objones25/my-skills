# A Survey of Flaky Tests, and Systemic Flakiness

Two papers by the same group (Owain Parry, Gregory M. Kapfhammer, Michael Hilton, Phil McMinn; Sheffield, Allegheny, CMU).

| Tag | Paper | How read |
|---|---|---|
| [Parry21] | *A Survey of Flaky Tests.* ACM TOSEM, 2021, DOI 10.1145/3476105, 71 pages | Accepted-manuscript PDF from philmcminn.com/publications/parry2021.pdf. Read: methodology, the eight "Conclusion for RQ" boxes, Table 12 on repairs. Body sections skimmed. |
| [Parry25] | *Systemic Flakiness: An Empirical Analysis of Co-Occurring Flaky Test Failures.* arXiv:2504.16777 (v1, Apr 2025); venue not printed on the arXiv PDF | Read in full from the arXiv PDF. |

## The idea

[Parry21] is a systematic review of 76 papers on flaky tests, organized by causes, costs, detection, and mitigation and repair [Parry21 Abstract, §2]. [Parry25] challenges the assumption that flaky failures are independent: in 24 Java projects, three quarters of flaky tests fail together in clusters that share a root cause, mostly network trouble or an unstable external dependency, so one fix can repair many tests [Parry25 Abstract, §1].

## Terms

| Term | Meaning |
|---|---|
| Flaky test | Passes and fails with the test code and code under test unchanged [Parry21 §1]. |
| Order-dependent test | Outcome depends on test execution order [Parry21 §3.2]. |
| Async wait | A test that waits a fixed time for an asynchronous result instead of waiting for the event [Parry21 §3.1]. |
| Systemic flakiness | Flaky tests whose failures co-occur in the same runs because of a shared cause [Parry25 §1]. |
| Jaccard distance | Here, 1 − (shared failing runs / union of failing runs) for two tests; the clustering distance [Parry25 §2.2]. |
| Agglomerative clustering | Repeatedly merge the two closest groups; cut the tree where the silhouette score (cluster separation) is best [Parry25 §2.2, §4.1]. |
| FlakeFlagger dataset | 10,000 test-suite runs of 24 Java projects with 810 flaky tests [Parry25 §2.1]. |

## Study designs

**[Parry21].** Search and snowballing collected 76 papers; four RQs answered in §3-§6, each ending in a "Conclusion for RQ" box [§2].

**[Parry25].** (1) For each project, cluster flaky tests by Jaccard distance over their failing-run sets [§2.2]. (2) Train models (extra trees and others) to predict that distance from static features: test-name similarity, code similarity, distance in the class hierarchy [§2.3]. (3) Four authors read stack traces of 45 clusters to code causes and possible repairs [§2.4].

## Key results

[Parry21]
- Across commit, bug-report and survey studies, async-related timing is the leading cause; one study found it in up to 78% of cases. But the Python trace study (Gruber et al.) found it a minority category [Parry21 RQ1.1 box].
- Platform dependency accounted for up to 34% of flakiness bug reports in one study [RQ1.1 box].
- For order-dependent tests, one study traced 61% to static fields (global variables) and found 76% depend on only one other test [RQ1.2 box].
- Only 50 of 107 flaky tests found by rerunning whole suites could be reproduced as flaky when run in isolation [RQ2.1 box].
- Firefox builds with at least one flaky failure had a median of 234 production crash reports, against 291 for builds with any failing test and 2 for fully passing builds: ignoring flaky failures is not free [RQ2.1 box].
- At Google, 41% of test targets that had both passed and failed were flaky; at Microsoft, flaky failures appeared in 26% of sampled builds [RQ2.2 box].
- Coverage-based detection (DeFlaker) found 96% of the flaky tests that repeated runs found [RQ3.1 box].
- Repairs: for async wait, adding or changing a wait-for-event construct was the most common fix (57% and 86% in two studies); adding or changing a `sleep` 27% in one [RQ4.2 box, Table 12]. For order dependence, fixing setup/teardown or removing the dependency [RQ4.2 box].

[Parry25]
- 606 of 810 flaky tests (75%) belong to a cluster; mean cluster size 13.5; 10 of the 22 projects with flaky tests have at least one cluster [Parry25 §3.1].
- Extra trees predicted pairwise failure distance with mean R² = 0.74 and classified with MCC 0.74; class-hierarchy distance was the most useful feature, and name-based features beat code-based ones [§3.2].
- Most common cause theme across 45 clusters: networking; second: external dependency [§3.3, Table 4].
- Most common repair theme: "look before you leap" (check a directory exists, a server is up); second: mocking the external service or the system clock [§4, Table 5].

## What it means for a pytest suite

- When `flake_hunt.py` reports several flaky tests, group them by the runs in which they failed before fixing them one by one. Tests that fail in the same seeds and runs likely share a cause [Parry25 §3.1]. Adding a co-failure grouping to the script's output is cheap: it already records which seeds failed for each test.
- Network and external services are the usual shared cause [Parry25 §3.3]. This backs hard gate 14 (no network in unit tests) and the `NETWORK` audit code; for integration tests, add a readiness check in the fixture (look before you leap) rather than a retry.
- Never fix async flakiness with a longer `sleep`; wait on the event or condition [Parry21 RQ4.2, Table 12]. Hard gate 15 (`SLEEP`) already pushes this way.
- Do not dismiss flaky failures; the Firefox data links ignored flaky failures to more crashes [Parry21 RQ2.1]. Supports `flaky-tests.md`'s "never retry in a merge gate".
- A flaky failure seen in the full suite may not reproduce in isolation [Parry21 RQ2.1]; when debugging, rerun with the same `--randomly-seed`, not the test alone.

## Threats to validity / caveats

- [Parry21] summarizes other studies, mostly Java; its numbers are those studies' numbers.
- [Parry25] uses Java projects only and one dataset; co-occurrence is assumed to mean a shared cause; cause coding was done by authors unfamiliar with the projects [§2.5, §4].
- The survey's coverage ends at its 2021 publication.

## Disagreements

- Leading cause: Java studies say async wait; Gruber et al. say order dependence for Python [Parry21 RQ1.1 box; `gruber-flaky-python.md`]. The skill's Python-first ordering (order dependence first) is the right one for its users.
- [Parry25] vs the default "each flaky test is independent" model behind per-test rerun statistics, including Gruber et al.'s rerun numbers, which treat tests separately.
