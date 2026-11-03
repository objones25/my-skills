# Property-Based Testing in Practice

Harrison Goldstein, Joseph W. Cutler (Penn), Daniel Dickstein (Jane Street), Benjamin C. Pierce, Andrew Head (Penn). ICSE 2024, Lisbon. Read in full from the author PDF at harrisongoldste.in/papers/icse24-pbt-in-practice.pdf. Interview study at one company that uses OCaml; Python transfer is by analogy.

## The idea

Thirty experienced PBT users at Jane Street were interviewed about where property-based testing pays off. Most successful uses fell into a few high-leverage shapes, above all comparing code to a simpler reference implementation. The main pains were writing properties and generators, shrinking, and knowing whether a passing property test actually tested anything: over a third admitted they did not think hard enough about that [PBTP Abstract, §4.3, §4.6].

## Terms

| Term | Meaning |
|---|---|
| Property-based test (PBT) | A test that states a rule for all inputs; a harness generates many inputs and checks the rule [§2]. |
| Generator | Code that produces random inputs of a type, often built from combinators [§2]. |
| Derived generator | A generator synthesized automatically from a type definition (OCaml's `ppx_quickcheck`; Hypothesis's `st.builds`/`st.from_type` play a similar role) [§4.4]. |
| Shrinker | Code that turns a failing input into a smaller one that still fails [§4.5]. |
| Expect test | Jane Street's example-based test style: log output, compare to a saved snapshot. Used here as a stand-in for ordinary unit tests [§4.2]. |
| Differential (model-based) property | Compare the system to a reference implementation or abstract model [§4.3]. |
| Round-trip property | `decode(encode(x)) == x` and similar inverse pairs [§4.3]. |
| Catastrophic-failure property | Only checks that nothing crashes or raises unexpectedly; fuzzing-like [§4.3]. |

## The study design

1. Semi-structured interviews with 30 Jane Street developers who use PBT [§3].
2. Qualitative coding of transcripts into themes; counts reported as n/30 [§3, §4].
3. Observations (OB) and research opportunities (RO) drawn from the themes [§5].

## Key results

- High-leverage idioms: differential properties 17/30 (the most common), classical algebraic properties 11/30, round-trip 11/30, catastrophic-failure 7/30 [PBTP §4.3].
- One participant's summary: PBT is "most useful when ... you have a really good abstraction with a complicated implementation" [§4.3].
- Derived generators were used by 19/30; 7/30 combined derived and hand-written ones [§4.4].
- Shrinking was called necessary, but writing shrinkers by hand was disliked and error-prone (a shrinker can break an input invariant and produce a false failure) [§4.5].
- 11/30 said they did not think enough about whether their generators exercised the code; 3 of them stopped improving properties after the first bugs found [§4.6].
- How the rest checked their tests: mutation testing (planting bugs) 7/30, inspecting sample generated values 8/30, code coverage 2/30, coverage of the property code itself 1/30 [§4.6].
- About a third said PBT found bugs other methods had missed, and about a third said it raised their confidence [§4.2].
- 12/30 used properties as documentation; reviewers sometimes asked for properties in code review [§4.1, §5.1 OB2].
- No universal preference over example tests; both belong in the toolkit [§4.2].
- Research opportunities include time budgets for PBT runs (RO1), better support for differential testing (RO2), better shrinking interfaces (RO5), and tools for judging test effectiveness (RO6, RO7) [§5.2].

## What it means for a pytest suite

- Lead with the differential pattern. The skill's property table lists "Oracle (differential)" among seven; this paper suggests putting it first when a slow, obvious reference implementation exists, and stating the precondition (a simple model) so the agent looks for one [§4.3].
- Check what the generator produces. Add a step to the Hypothesis workflow: run `strategy.example()` a few times, or use `hypothesis.event()` / `--hypothesis-show-statistics`, to confirm the interesting region is hit. 8/30 practitioners did this by eye [§4.6].
- Mutation-test property tests too. `mutate_check.py` on a function covered only by a property test tells you whether the generator reaches the mutated branch [§4.6]. This is the most direct answer to "is my property test vacuous?"
- Prefer type-driven generation (`st.builds`, `st.from_type`) and change types to narrow inputs (an `Enum` instead of an `int` flag) rather than hand-filtering [§4.4].
- Never hand-write a shrinker in Python; Hypothesis shrinks generically (see `hypothesis-reducer.md`). The pain reported here is specific to QuickCheck-style frameworks [§4.5].
- Keep example tests next to properties; a property alone can miss behaviors no property states [§4.6].

## Threats to validity / caveats

- One company, one language (OCaml), a strong PBT culture; self-selected volunteers.
- Counts are how many interviewees mentioned something, not how often it happens.
- Interview data, no measurement of bug-finding.

## Disagreements

- None with the other PBT papers read. Agentic PBT (`agentic-pbt.md`) is consistent: its high-value property list (invariants, round-trip, metamorphic) mirrors these idioms, and its false-alarm problem echoes this paper's point that properties need a clear specification.
