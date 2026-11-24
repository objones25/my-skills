# Audit, 2026-11-24 (version 1.1.0)

## Errors found

| Claim | Evidence | Fix |
|---|---|---|
| EPO beat SAPG "35.8 vs 2.2" on two-arm reorientation; seed variance "20% vs 38%" | EPO (2503.19037) text lines 495-496 compare 35.8 with SAPG's 28.58; its Table 1 row M:TR gives SAPG 4.4 and 3.9; 2.2 is PBT on M:ST. No percentage for seed variance in the paper | Corrected in `large-batch-on-policy.md` and `papers.md`; the step-count disagreement behind 28.58 (5e10 in EPO, 2e10 in SAPG) stated in place |
| gymnax ORs the time limit into `done` (Non-negotiable 1) | gymnax 1.0.1 source: `step_env`'s done is natural termination only; `is_truncated` is separate | Scoped to gymnax < 1.0; `AutoReset` fixed to apply the time limit itself |
| gymnax MinAtar has 1-2 start states | Freeway: 2,000 distinct resets of 2,000 keys on gymnax 0.0.9 and 1.0.1 | Claim limited to Breakout, SpaceInvaders and Asterix |
| ABS, SGS, PQN, FlashSAC claims without their conditions | Citation audit against the paper texts | Qualifiers added (see `CHANGELOG.md`) |

## How it was checked

- Citation audit: one agent checked every claim the skill made about the corpus papers it already cited. All SKILL.md defaults-table numbers and all Zheng et al. exponents and validation numbers were correct.
- New content: an independent reviewer that wrote none of it checked the v1.1.0 diff against the paper texts (about 55 claim groups confirmed; 1 wrong, 2 miscited, 3 overstated, 3 missing a qualifier, 1 unsupported). All fixed after the coordinator re-checked them, except one PACE item: the reviewer's numbers (0.694, 0.618) could not be found in the paper text, so the Table 5 values stay.
- gymnax behaviour was checked by reading the installed source and running resets, not from docs.
- The test suite was run on jax 0.11.2 + gymnax 1.0.1 and jax 0.12.0.1 + gymnax 0.0.9.

## Not verified

- Repo file:line facts dated 2026-10-27 were not rechecked against newer commits.
- Papers in the corpus that were only triaged contribute no numbers; claims from figures that did not survive text extraction (e.g. DRATS Fig. 13, the Kinetix Appendix L result, Moalla et al.'s env counts) were not re-verified.
- The trigger evals have not been run. Output evals: one blind run per case (`CHANGELOG.md`).
