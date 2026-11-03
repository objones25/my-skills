# Agentic Property-Based Testing: Finding Bugs Across the Python Ecosystem

Muhammad Maaz (MATS, Anthropic), Liam DeVoe (Northeastern), Zac Hatfield-Dodds (Anthropic), Nicholas Carlini (Anthropic). arXiv:2510.09907 (v1, 10 Oct 2025); no venue printed. Read in full from the arXiv PDF (main text; the prompt in Appendix A not read). Code: github.com/mmaaz-git/agentic-pbt. Hatfield-Dodds and DeVoe are Hypothesis maintainers.

## The idea

A prompt for Claude Code tells the agent to read a Python module and its docs, propose properties grounded in that evidence, write Hypothesis tests, run them, triage failures against a rubric, and write bug reports only for failures it believes are real. Run across 100 popular packages (933 modules), it produced 984 reports; on manual review of a sample, 56% were real bugs and 32% worth reporting upstream. Several led to merged fixes, including in NumPy [APBT Abstract, §2-§4].

## Terms

| Term | Meaning |
|---|---|
| Property | A rule the code should satisfy for all valid inputs, here derived from code, docstrings and callers [§2]. |
| Hypothesis | The Python PBT library; tests use `@given` with strategies [§1]. |
| Triage rubric | Instructions for deciding whether a failing property is a real bug or a false alarm (wrong property or invalid input) [§2]. |
| False alarm | A failing property that reflects a mistaken property or input domain, not a bug [§2, §5]. |
| Intent ambiguity | The agent cannot tell an intentional design choice from a bug (example: `requests`' `LookupDict` is not meant to act like `dict`) [§5]. |

## The method

The agent is a Markdown prompt with six steps [§2]:

1. **Analyze the target** (file, module, or function).
2. **Understand it**: read docs, signatures, source.
3. **Propose properties** supported by evidence; examples given include invariants, round trips, metamorphic relations. Study callers to learn the real input domain.
4. **Write Hypothesis tests.**
5. **Execute and triage** with pytest. False alarm: refine and go back to step 4. Passing test: check that it is meaningful.
6. **Report** as Markdown: summary, the failing PBT, a minimal reproducer, why it is a bug, optionally a patch.

Design choices: a to-do list for progress; emphasis on few, well-evidenced properties to cut false alarms; broad autonomy over targets [§2]. Corpus: 15 stdlib modules, 15 hand-picked PyPI packages, 70 random from the top 5,000 PyPI by downloads; main module plus one level of submodules [§3]. Model: Claude Opus 4.1 [§3].

## Key results

- 984 bug reports from 933 modules; at least one in 786 modules (84.2%) [APBT §4].
- Cost $5,474.20 total, $5.56 per report, about $9.93 per valid bug; 136.6 hours of agent time, about 110 turns per run [§4, §5].
- Sample of 50 reports: 56.0% valid (95% CI 42.2-69.8%), 32.0% valid and worth reporting; inter-rater κ = 0.31 before consensus [§4].
- Top-scoring 21 under the final rubric: 18 valid, 17 worth reporting [§4].
- Reported upstream: NumPy `random.wald` returning negative samples through catastrophic cancellation (patch merged); `aws-lambda-powertools` `slice_dictionary` repeating its first chunk (merged); `tokenizers` HSL string missing a parenthesis (merged); a `cloudformation-cli` hash where every list hashed to `hash(None)` because of in-place `.sort()` (patch submitted); `dateutil.easter` Julian dates not on Sunday (judged intended) [§4].
- False discovery rate 95% CI 30.2-57.8% [§5].

## What it means for a pytest suite

- Ground properties in evidence the agent can cite: docstring text, type hints, or caller usage. The paper attributes its low false-alarm rate to this [§2]. The skill's property-writing guidance can require a one-line source for each property (for example a comment `# docstring: "returns a non-negative ..."`).
- Look at callers to define the strategy's domain; the signature alone is not enough [§2 step 3]. This avoids both invalid inputs (false alarms) and over-narrow strategies (the `allow_nan=False` trap the skill already warns about).
- After a property passes, ask whether it could fail. The agent is told to verify passing tests are meaningful [§2 step 5]; `mutate_check.py` is the mechanical version.
- When a property fails, produce a minimal reproducer as a plain test plus the Hypothesis `@example`, matching the skill's "Turning findings into permanent tests" section [§2 step 6].
- Treat agent-found "bugs" in third-party behavior skeptically: roughly half were not real [§4]. For a user's own code, a failing property should become a question to the user when the docs are ambiguous, not an automatic code change.

## Threats to validity / caveats

- Only 50 of 984 reports manually reviewed, with low initial agreement (κ = 0.31) [§4, §5].
- "Valid" was judged by the authors, not maintainers, except for the five reported bugs.
- One model, one prompt; packages are popular and well tested, so bugs found are hard ones but the rate may not transfer to a user's private code.
- Authors include Hypothesis maintainers and Anthropic staff evaluating an Anthropic model.

## Disagreements

- None with papers read in full. A 2026 follow-up seen only as an abstract (arXiv:2610.34487, "A Multi-Domain Study on the Effectiveness of LLMs at Writing Property-Based Tests") reports Agentic-PBT generalizing to databases, compilers and ML libraries, with mature domain fuzzers covering more of compilers and PostgreSQL; not read, so not relied on.
