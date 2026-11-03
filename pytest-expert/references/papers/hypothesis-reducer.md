# Test-Case Reduction via Test-Case Generation: Insights from the Hypothesis Reducer

David R. MacIver, Alastair F. Donaldson (Imperial College London). ECOOP 2020, Tool Insights paper. DOI 10.4230/LIPIcs.ECOOP.2020.13. Read in full from the open-access LIPIcs PDF (drops.dagstuhl.de). MacIver created Hypothesis.

## The idea

Most reducers shrink a failing test case directly (delete lines, simplify values), which often produces inputs the generator could never have made, so they fail for a different reason. Hypothesis instead records the sequence of random choices the generator consumed, shrinks that sequence, and regenerates. Any reduced case is therefore one the generator could produce, and every generator gets shrinking for free, with no user-written shrinker [HR Abstract, §1, §7].

## Terms

| Term | Meaning |
|---|---|
| Test-case reduction (shrinking) | Turning a large failing input into a smaller one that still fails [§1]. |
| External reduction | Reducing the produced test case itself (C-Reduce, delta debugging, QuickCheck shrinkers) [§1]. |
| Internal reduction | Reducing the generator's choice sequence and regenerating [§1, §2.2]. |
| Choice sequence | The stream of random decisions (bits) a generator reads; replaying it reproduces the value exactly [§2.2]. |
| Shortlex order | Shorter sequences first, then lexicographically smaller; the order the reducer minimizes in [§2.2]. |
| Test-case validity problem | A reduced case violates the input's preconditions and triggers a different failure [§1]. |
| SUT | System under test. |

## The method

1. **Generators as parsers.** A generator is treated as a deterministic function from a choice sequence to a value [§2.2].
2. **Shortlex optimization.** Search for the shortlex-smallest choice sequence whose regenerated value still triggers the failure [§2.2].
3. **Passes.** In Hypothesis 5.15.1, 15 passes: six that delete contiguous regions, one that replaces a region with a sub-region, one that zeroes a region, four lexicographic passes, three mixed passes; one lexicographic pass is a special case for floats [§3.1].
4. **Generator-directed reduction.** Use the structure the generator exposes to choose edits that keep validity [§3.2].

## Key results

- Python autoformatter bugs (differential testing of yapf and black with a custom program generator): Hypothesis, Picire (delta debugging) and C-Reduce gave reduced cases of similar size; differences not significant [HR §4.1.2, Fig. 8].
- C compiler bugs (Csmith programs): C-Reduce and Picire produced substantially smaller programs than Hypothesis [§4.1.2, Fig. 8].
- Cost in SUT calls: Hypothesis 762 vs Picire 3,138.9 and C-Reduce 3,968 on Csmith (about four times fewer than Picire); on the Python formatting study 1,284.6 vs Picire 529.2 and C-Reduce 863.3 (about 50% more than Picire) [§4.1.2, Fig. 12].
- Overhead of going through the generator: generator evaluations per SUT call, geometric mean 95% CI 2.78-2.94 for Csmith [§4.1.2].
- On SmartCheck's five synthetic QuickCheck benchmarks, Hypothesis was worse than QuickCheck only on "binheap", and only slightly [§4.3].
- Hypothesis limits choice sequences to 8KB by default; all experiments kept that limit [§5].

## What it means for a pytest suite

- Do not write custom shrinkers for Hypothesis strategies; the skill should state that generic shrinking is the design, and that strategies built from `st.builds`, `st.composite` and `flatmap` all shrink [§7].
- Prefer generating valid inputs by construction (`st.composite`, `st.builds`) over generating broadly and calling `assume()`. Shrinking works on the choice sequence, so a strategy that builds valid values directly shrinks to valid small values; heavy filtering gives the reducer fewer valid neighbours. This is an inference from §2-§3, not a measured result.
- Shrunk counterexamples are minimal "for this generator", not globally minimal; for parsers or compilers a format-aware reducer may do better [§4.1.2, §7]. When a Hypothesis failure is still large, write the regression test from the shrunk value and simplify by hand.
- `@example` replays a fixed value outside the choice-sequence machinery; that is why the skill's rule to pin known special values with `@example` works regardless of shrinking.

## Threats to validity / caveats

- Authors built the tool; evaluation is described as "mostly descriptive" to show adequacy, not superiority [§4].
- Inputs capped at 8KB of choices; larger cases may behave differently [§5].
- Hypothesis version 5.15.1 (2020); the skill pins 6.168.3, and the shrinker has changed since. The design principle is stable; pass counts are not.

## Disagreements

- Goldstein et al. (`pbt-in-practice.md`) report practitioners hating hand-written shrinkers in QuickCheck-style tools; this paper is the design answer to that complaint. No conflict.
