# PyNose: A Test Smell Detector for Python

Tongjie Wang (UC Irvine), Yaroslav Golubev, Oleg Smirnov (JetBrains Research), Jiawei Li (UC Irvine), Timofey Bryksin (JetBrains Research), Iftekhar Ahmed (UC Irvine). ASE 2021. arXiv:2108.04639. Read in full from the arXiv PDF. Substitutes for Panichella et al. "Test smells 20 years later" (EMSE 2022), which the research index did not return and was not read.

## The idea

Test smells (patterns in test code associated with poor maintainability or reliability) had been studied in Java and Scala but not Python. The authors picked 17 language-agnostic smells from the literature, mined commit histories for a Python-specific one (Suboptimal Assert), built a PyCharm plugin to detect all 18 in `unittest` code, and measured prevalence: 98% of projects and 84% of test suites had at least one smell [PyNose Abstract, §III-§V].

## Terms

| Term | Meaning |
|---|---|
| Test smell | A test-code pattern that suggests a design or maintenance problem; not necessarily a bug [§I]. |
| Test suite / test case | In `unittest`: a `TestCase` class / one `test_*` method [§V]. |
| Assertion Roulette | More than one assertion without a message, so a failure does not say which check broke [Table II]. |
| Conditional Test Logic | `if`, `for`, or `while` inside a test [Table II]. |
| Magic Number Test | An assertion with a bare numeric literal as an argument [Table II]. |
| Unknown Test | A test with no assertion [Table II]. |
| Suboptimal Assert | A general assertion where a specific one exists: `assertTrue(X == Y)` instead of `assertEqual`, `assertEqual(X, False)` instead of `assertFalse(X)`, `assertTrue(X <= Y)` instead of `assertLessEqual` [§III-B]. |
| Sleepy Test | `time.sleep()` with no comment [Table II]. |
| Redundant Assertion | `assertEqual(X, X)` or `assertTrue(True)` [Table II]. |
| Exception Handling | `try/except` or `raise` in a test [Table II]. |
| General Fixture / Test Maverick | `setUp()` builds fields not all tests use / a test uses none of them [Table II]. |

## The method

1. **Literature mapping** to select smells definable for Python `unittest` [§III-A].
2. **Change-pattern mining** over 450 projects to find frequent test edits that fix or introduce a smell; 159 patterns studied, 70 (44%) were assertion changes, yielding Suboptimal Assert [§III-B].
3. **Tool**: PyCharm plugin using the PSI syntax tree, with the rules in Table II [§IV-A].
4. **Validation**: two raters labeled 8 projects; precision, recall, F1 per smell [§IV-B, Tables III-IV].
5. **Prevalence**: Primary dataset of 450 projects (248 with `unittest` tests: 9,158 files, 16,681 suites, 96,736 test cases), plus a Secondary dataset of 239 smaller projects as a check [§V].

## Key results

- Tool accuracy: weighted precision 94.0%, recall 95.8%, F1 94.9%; per-smell F1 from 81.5% (Lack of Cohesion) to 100% [PyNose §IV-B, Table IV].
- Two Unknown Test false positives were tests using pytest-style asserts, which the tool does not support [§IV-B].
- 5 of 248 projects (2%) had no smells, all tiny; 84% of suites had at least one [§V-C].
- Share of projects / of test suites with the smell (Primary): Assertion Roulette 89.9% / 52.4%; Conditional Test Logic 89.5% / 31.9%; Magic Number Test 79.8% / 26.9%; Unknown Test 81.5% / 22.5%; Duplicate Assertion 78.2% / 17.4%; Suboptimal Assert 70.6% / 15.4%; Exception Handling 64.9% / 8.6%; Sleepy Test 23.8% / 1.4%; Redundant Print 37.9% / 2.9% [§V-C, Fig. 4].

## What it means for a pytest suite

- Most of PyNose's smells already map to `audit_tests.py` codes: Conditional Test Logic → `BRANCHING`; Unknown Test / Empty Test → `NO_ASSERT`; Exception Handling → `SWALLOWED_FAILURE` (the audit is narrower: it flags swallowing, not every `try`); Sleepy Test → `SLEEP`; Redundant Assertion → `ALWAYS_TRUE`; Redundant Print → `PRINT`; Ignored Test → `SKIP_NO_REASON`. This is external support for those rules.
- Smells PyNose finds that the audit does not:
  - **Suboptimal Assert, pytest flavor.** pytest rewrites plain `assert`, so `assertEqual` vs `assertTrue` does not apply, but the analogue does: `assert (a == b) is True`, `assert bool(x) == True`, `assert x == None`. ruff's PT rules cover some; an audit warning would cover the rest.
  - **Duplicate Assertion** (same assertion twice in one test): cheap AST check, present in 78.2% of projects [Fig. 4].
  - **Magic Number Test** is debatable for pytest; the skill's rule 10 asks for exact expected values, which are often literals. Do not adopt it.
  - **Assertion Roulette** matters less in pytest because assertion rewriting shows both sides; skip it.
- Expect smells in almost every existing suite [§V-C]. When auditing a legacy suite, report counts per code and fix the error-level ones first rather than blocking on all findings at once.

## Threats to validity / caveats

- `unittest` only; pytest-style tests were out of scope and caused false positives [§IV-B]. Most modern Python projects use pytest, so prevalence numbers describe `unittest` code.
- Validation on 8 projects with few instances of some smells (one example each for three smells) [§IV-B, Table IV].
- Smells are associations with poor quality, not demonstrated defects; the paper does not measure fault detection.

## Disagreements

- None among the papers read. Panichella et al. (EMSE 2022) was not read, so no comparison is made.
