# Audit, 2026-11-16 (version 0.4.0)

## Fact-check

audit-skill run of 2026-11-16, risk scope. Report: `audits/robot-learning-expert/2026-11-16.md` in the My_Skills repo (gitignored, local only); run folder beside it. Coverage line, verbatim: "Checked 124 of 2460 items (5%): 97 confirmed, 8 wrong, 10 stale, 8 disputed, 1 could not be verified. Independent review: stopped at the 3-round cap; 8 item(s) disputed."

Every wrong and stale item was fixed on 2026-11-16, listed by item id in `CHANGELOG.md`. The 8 disputed items:

- I-W1-0305 (DQN target refresh C): the paper and DeepMind's released code disagree; `concepts/q-learning.md` now states both.
- I-LEAD-0006 ("no arXiv version" of DQN Nature): reworded as a dated negative search.
- I-LEAD-0010 (SAC alpha learning rate as a cause of entropy collapse): tagged [inference].
- I-W6-0091 (G1 sweep CSV provenance): stated as reported by the run's author, not recorded in the CSV.
- I-W6-0105, I-W6-0106, I-LEAD-0007 (Pendulum per-seed records with no environment): replaced by the dated rerun with versions.
- I-LEAD-0004 (default-run eval and the bug's effect): covered by the same rerun.

Mutation check after the new tests (audit-skill `mutate_check.py`, each script against its own test file, 2026-11-16): lqr.py 41/80 killed (was 24/80), run_stats.py 36/71 (was 17/55). Survivors that remain are in `lqr.dare`'s iteration and input checks and in the scripts' `_self_check` functions, which the standalone subprocess test runs but the per-file mutation run does not.

## Not verified

- The 2336 items the audit did not check. 32 files were not opened at all, among them `references/concepts/{bandits,continuous-trajopt,imitation,mdp-dp,model-based,resets-and-exploration}.md`, `references/notation.md`, `references/sources.md` and its errata, `references/papers/{bellman-eluder,coverage,mcts-spectral,theory-map,README}.md`, `references/training/{diagnostics,evaluation,off-policy-at-scale,reward-and-observation-design}.md`, the assets, `evals/evals.json`, and `scripts/{batch_math,local_sim,mdp_tools}.py` beyond running their tests. The audit's workers flagged the theorem bounds in `papers/power-of-resets.md` (read from page images) and the numbers in `papers/massively-parallel-survey.md` as the riskiest of the remainder. Nothing here vouches for them.
- The audit's 7 suggestions (arXiv v2 versions of supermimic and icrl-transformers, UGTS venue, PQN "up to 50x" wording, rsl_rl's `kl_mean > 0` condition, mjlab `--num-envs` flag, MLPModel `distribution_cfg` default) were not acted on.
- Facts in `simulators-and-tooling.md` other than versions, upload dates and dependency pins were not rechecked against the new releases (mujoco-warp 4.0, MJX 4.0, playground 1.0 feature and limit lists; the playground 1.0 release notes).
- The Pendulum measurement is one machine and one set of versions; runs are not bitwise repeatable.
- Neither the evals (including the two new ones) nor the trigger evals have been run.
- No independent review of the 0.4.0 changes themselves.

## Mechanical checks, 2026-11-16

- `check_skill.py` before the changes: 1 error (`.DS_Store`), 5 warnings (no negative scope, no root CHANGELOG, no trigger evals, no test-command, no AUDIT.md). After: see `CHANGELOG.md`.
- Tests before, with every optional package installed (numpy 2.5.5, scipy 1.18.1, gymnasium 1.4.0, torch 2.15.0, pytest 9.2.0; macOS arm64, Python 3.13): 188 passed, 1 skipped (the course-framing check skipped the CHANGELOG by design).
