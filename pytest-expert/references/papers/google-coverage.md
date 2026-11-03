# Code Coverage at Google

Marko Ivanković, Goran Petrović (Google), René Just (UW), Gordon Fraser (Passau). ESEC/FSE 2019, industry track. DOI 10.1145/3338906.3340459. Read in full from the author PDF at homes.cs.washington.edu/~rjust/publ/google_coverage_fse_2019.pdf. Industrial self-report.

## The idea

Coverage is cheap only if someone builds and maintains the infrastructure. Google computes line coverage (and branch coverage for C++, Java, Python) daily for about a billion lines in seven languages and shows it inside code review, per changelist. The paper reports adoption over five years and a developer survey, and gives deployment advice. It is a usability paper, not an effectiveness study [Coverage Abstract, §1].

## Terms

| Term | Meaning |
|---|---|
| Line coverage | Share of instrumented lines (no blanks or comments) executed by tests. At Google it tracks statement coverage closely because of enforced style guides [§2.2]. |
| Branch coverage | Share of conditional-branch outcomes executed; measured for C++, Java, Python [§2.2]. |
| Project coverage | Line coverage of a project's tests over its code, computed daily [§2.2]. |
| Changelist (delta) coverage | Coverage of the lines added or changed in one code review [§2.2]. |
| Failed coverage computation | A coverage run that errors although tests pass without instrumentation [§4.3]. |

## The study design

1. **Infrastructure description** built on existing tools (gcov, JaCoCo, coverage.py and others) behind one interface [§3].
2. **Adoption data**, 2012 to 2018: about 13,000,000 daily project measurements and 14,000,000 changelist measurements [§4].
3. **Survey**, February 2019: 3,000 randomly chosen developers, 512 responses (17%), three Likert questions plus free text [§4.4].

## Key results

- Only 0.5% of developers run coverage manually on a given day, while 33.5% run tests from the command line daily; coverage is consumed when it is automatic [Coverage §4.1, Table 1].
- Google sets no global threshold. A voluntary five-level scheme is used; for example Level 4 is project coverage ≥ 75% and changelist coverage ≥ 80%, Level 5 is ≥ 90% for both [§4.2, Table 2].
- Main causes of failed coverage runs: instrumentation slows tests into timeouts and bloats binaries into out-of-memory errors; second, flakiness, because instrumented code is slower and sleep-based tests break [§4.3].
- When authoring, 45% use changelist coverage "Often" or "Very often" and 25% "Sometimes"; when reviewing, 40% and 28%; 10% never use it [§4.4, Fig. 11].
- Of 286 free-text answers: 60% positive, 24% neutral, 16% negative [§4.4, Table 3].
- Safety-critical projects use mutation testing on top of coverage [§4.2].
- Recommendations: measure automatically at key workflow points, show it in tools developers already use, budget for instrumentation-caused errors, reuse established libraries, one interface for all languages [§6].

## What it means for a pytest suite

- Make coverage automatic and per-change. The skill's CI already enforces a floor; adding a diff-coverage report (lines changed in the PR) matches what Google developers actually used [§4.1, §4.4].
- Expect `--cov` to change timing. Instrumented runs are slower, so timeouts and sleep-based tests fail under coverage first [§4.3]. This backs hard gate 15 (no `time.sleep`) and suggests running the flake hunt under coverage too.
- Thresholds are a team choice, and the scheme Google publicized tops out around 75-90%. That is consistent with hard gate 27 ("do not chase 100%").
- Coverage is the prerequisite for mutation, not the goal. Google's own mutation system only mutates covered lines (`google-mutation-at-scale.md` §2.1).

## Threats to validity / caveats

- Adoption counts are percentages only; absolute project counts are confidential [§4.1 footnote].
- Median project coverage over time is plotted (Fig. 8) but not tabulated; no number is quoted here.
- Survey response rate 17%, self-reported, and includes non-engineers [§4.4].
- Says nothing about whether coverage improves fault detection.

## Disagreements

- Not in conflict with Inozemtseva and Holmes (coverage is a weak effectiveness signal); this paper only claims developers find it useful for spotting untested code, which Inozemtseva and Holmes also concede.
