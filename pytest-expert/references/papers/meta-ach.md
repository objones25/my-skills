# Mutation-Guided LLM-based Test Generation at Meta (ACH)

Christopher Foster, Abhishek Gulati, Mark Harman, Inna Harper, Ke Mao, Jillian Ritchey, Hervé Robert, Shubho Sengupta (Meta; alphabetical, Harman corresponding). arXiv:2501.12862 (v1, Jan 2025); the PDF names FSE Companion '25, Trondheim. Read in full from the arXiv PDF. Note: the PDF's running header ("Harman, Sengupta") and DOI line (10.1145/3663529.3663839) appear to be left over from the TestGen-LLM template; do not cite that DOI for ACH. Industrial self-report by the tool's builders.

## The idea

Instead of asking an LLM to "add tests" or "raise coverage", first ask it to write a realistic bug of a chosen kind (here, privacy leaks) that no existing test catches, filter out mutants that are equivalent to the original, then ask it to write a test that fails on that bug and passes on the real code. The surviving mutant is the prompt. Engineers accepted 73% of the resulting tests in two test-a-thons, and about half of the tests that killed a mutant added no line coverage [ACH Abstract, §1, §6].

## Terms

| Term | Meaning |
|---|---|
| Mutant / simulated fault | An LLM-written version of a class with one injected bug per method, marked with `// MUTANT <START>` / `<END>` comments [Table 1]. |
| Equivalent mutant | A mutant no test can distinguish from the original. |
| Equivalence detector | An LLM-as-judge agent asked whether two class versions always behave the same [§2, Table 1]. |
| Hardening | The new test catches a fault no existing test catches [§1]. |
| Assured LLMSE | Meta's term for wrapping LLM output in checks that give guarantees: here, builds, passes consistently, and kills a mutant [§1]. |
| Test-a-thon | A focused event where engineers write and review tests for one product area [§4]. |
| Fashion following | New tests copy the style of the existing test class [§1]. |

## The method

1. **Issue text in.** Free text about a concern (past privacy faults here) [§2].
2. **Make a fault.** An LLM gets the class, its test class, and the concern, and writes a mutated class. Keep only mutants that build and that pass the existing tests (so they are unkilled) [§2, Fig. 1].
3. **Drop equivalents.** Remove syntactically identical mutants, then ask the equivalence detector [§2, §5.2].
4. **Make a test to catch the fault.** Prompt with original class, mutated class and existing tests; ask for an extended test class with tests that fail on the mutant and pass on the original [Table 1].
5. **Assure.** Keep tests that build, pass on the original consistently (non-flaky), and fail on the mutant; report coverage gained when there is any [§1, Fig. 1]. This stage is a modified TestGen-LLM (`meta-testgen-llm.md`).
6. **Review.** Diffs with a summary and test plan go to engineers [Fig. 1].

All agents used one model, Llama 3.1 70B, with the simple prompts in Table 1 and no fine-tuning [§2].

## Key results

- 10,795 Kotlin classes on 7 platforms; 9,095 mutants built and passed; 4,660 (51%) judged non-equivalent; 571 tests generated [ACH §3, Table 2, Table 8].
- 25% of mutants were syntactically identical to the original, against typical 10-15% equivalent rates for rule-based tools [§3, Table 2].
- Of the 571 tests, 277 (49%) added no line coverage and would have been discarded under a coverage-only filter [§1, Table 8].
- Equivalence detector: precision 0.79, recall 0.47 (counting "unsure" as equivalent); with lexical dedup plus stripping comments, 0.95 and 0.96 [§5.2, Table 6]. 61% of equivalent mutants differed only by a misleading comment [§5.2, Table 5].
- Initial trial: 27 of 30 diffs accepted [§4.1, Table 3].
- Test-a-thons (Dec 2024): 191 tests reviewed, 140 accepted (73%); 175 rated for privacy relevance, about 36% at levels 4-5. WhatsApp 50/91 accepted, Messenger 90/100 [§4.2, Table 4].
- Engineers valued tests that were not privacy-relevant for other reasons (corner cases, coverage) and found the example fault helpful for understanding each test [§4.1].
- Compared with TestGen-LLM on the same platforms: TestGen-LLM produced a test for 32% of classes against ACH's 5.3%, but killed only 2.0-2.4% of the non-equivalent mutants against ACH's 12-15% (ranges are believed vs likely-actual non-equivalent counts) [§6, Table 8].
- 70% of mutants left unkilled by TestGen-LLM were in classes with no TestGen-LLM test at all [§6, Table 8].

## What it means for a pytest suite

- Use surviving mutants as test prompts. When `mutate_check.py` reports `SURVIVED line 49: <= -> <`, the agent's next prompt should include the original line, the mutated line, and the existing tests, and ask for a test that passes on the original and fails on the mutant. That is ACH's step 4 almost verbatim [Table 1].
- Then verify mechanically: re-run `mutate_check.py` and require the specific mutant to be killed, and require the new test to pass on unmutated code several times. These are ACH's assurances 1-3 [§1].
- Do not filter agent-written tests by "adds coverage". Half of ACH's useful tests added none [Table 8]. The skill already ranks mutation over coverage; this is industrial support for that.
- Strip comments and whitespace before deciding a mutant is equivalent; an LLM-written mutant often changes nothing but a comment [§5.2]. A rule-based `mutate_check.py` does not have this problem, but an LLM-proposed mutant step would.
- Target mutants at a concern. For security-relevant code, ask for mutants that remove a check (a missing `..` rejection, a dropped auth test) rather than generic operator swaps; pair with `security-testing.md` payload tables.

## Threats to validity / caveats

- Acceptance is engineers at a test-a-thon reviewing tests already filtered to pass; it says tests were acceptable, not that they would catch future regressions.
- Kotlin/Android only; one model; privacy concern only.
- Relevance ratings are subjective Likert scores by the reviewing engineers [§4.2, Fig. 2].
- Equivalence-detector accuracy reflects the easy mutants this generator makes, as the authors say, not general skill at program equivalence [§5.2].
- Self-reported by the team that built the tool; no external replication read here.

## Disagreements

- With coverage-guided generation (CoverUp, `coverup.md`; TestGen-LLM): ACH argues coverage is the wrong filter, since 49% of its tests add none [§6]. CoverUp optimizes coverage and reports mutation scores only as a secondary check.
- Consistent with Google (`google-mutation-at-scale.md`) that raw mutants need heavy filtering to be worth a human's time.
