# Changelog

## 2026-11-16, v0.4.0: fixes from the 2026-11-16 audit

Fixes the findings of the audit-skill report of 2026-11-16 (risk scope, 124 of 2460 items checked; see `AUDIT.md`). Item ids are the report's.

Broken behaviour
- `scripts/run_stats.py` with no arguments exited 2 (argparse needed a CSV), contradicting SKILL.md's "each runs standalone" (I-W1-0030). It now runs a self-check. New test `test_every_script_runs_standalone` runs every script in `scripts/` as a subprocess and fails on a non-zero exit or empty output; it failed on run_stats before the fix.

Wrong claims
- Time-out bootstrapping on Pendulum (I-W1-0042, I-W1-0086, I-W6-0106, I-W6-0107, I-LEAD-0004): the old "small and noisy" record did not reproduce. Rerun 2026-11-16 (macOS arm64, Apple M4 Pro, Python 3.13, torch 2.15.0, gymnasium 1.4.0, numpy 2.5.5; 80k steps, gamma 0.99, seeds 0-2): correct -110.4 / -137.2 / -124.3 (mean -124.0); terminal-treated -1117.1 / -762.6 / -150.6 (mean -676.8). Docstring, SKILL.md and `training/diagnostics.md` now say so, with versions; the Nov 2 record is kept in the docstring as unreproduced. New eval `timeout-terminal-pendulum-effect`.
- Run time: "about 75 s on 2 CPU cores" replaced by the measured 23 s (2 torch threads) and 28 s (all cores) on an Apple M4 Pro (I-W6-0109, I-LEAD-0003).
- `scripts/requirements.txt`: numpy floor lowered from 2.0 to 1.26 (I-W6-0020); the suite passes on numpy 1.26.4 (results below).
- Untested functions (I-LEAD-0001, I-LEAD-0002): new tests for `lqr.rollout_cost` (hand-computed, time-varying gains, seeded noise, DARE value), `lqr.estimate_AB`, `lqr.dare_residual`, `lqr.is_stable`, `run_stats.prob_improvement_ci` and `run_stats.curve_band`. Mutation check (audit-skill `mutate_check.py`, each script against its own test file): lqr.py 24/80 killed before, 41/80 after; run_stats.py 17/55 before, 36/71 after (the new `_self_check` adds mutants). In the four functions the audit named, the only survivors are changed default values of `reps` and `seed`.

Stale facts
- `training/simulators-and-tooling.md`: every version in the quick pick table rechecked on PyPI 2026-11-16. Updated mjlab 1.6.1, mujoco-warp and mujoco-mjx 4.0.1, playground 1.0.1 (base deps now torch + MuJoCo Warp, no JAX or Brax; "when to pick" now says to pin `playground<1` for the JAX + Brax stack), genesis-world 1.4.6 (Python `>=3.12,<3.15`) (I-W2-0189, 0191, 0193, 0195, 0207, 0216, 0275, 0277, 0282, 0316). The other six rows were unchanged on PyPI. mjlab 1.6.1 keeps `rsl-rl-lib==5.4.2` and `mujoco-warp~=3.11.0`, so the `[mjlab 1.6.0]` config facts still apply.

Disputed items, resolved by stating the disagreement
- DQN target refresh C (I-W1-0305): `concepts/q-learning.md` now says the paper's table gives 10,000 while DeepMind's released code counts agent steps (2,500 SGD steps).
- "No arXiv version" of DQN Nature (I-LEAD-0006): reworded as a dated negative search result.
- SAC entropy collapse (I-LEAD-0010): the alpha-learning-rate cause is now tagged [inference].
- G1 sweep CSV provenance (I-W6-0091): `examples/analyze_sweep.py` says the versions, date and hardware are as reported by the run's author and are not recorded in the CSV.
- I-W6-0105 and I-LEAD-0007 (old per-seed numbers with no environment): superseded by the dated rerun above.

Skill standard (skill-builder)
- `CHANGELOG.md` moved from `references/` to the skill root; `AUDIT.md` added; `evals/trigger_evals.json` added (10 should-trigger, 9 near misses); control eval `control-correct-gae` added; frontmatter gains `metadata.version` and `metadata.test-command` (`uv run --no-project --with numpy --with pytest python -m pytest tests`, so the packager can run the tests without a prepared environment); the description now names what the skill is not for. `test_skill_structure.py` checks the metadata, the negative scope and the trigger evals, and the em-dash and path checks now cover `CHANGELOG.md` and `AUDIT.md`.

- Final gate, 2026-11-16: `check_skill.py --strict` 0 errors, 0 warnings (before: 1 error, 5 warnings); `package_skill.py` packaged 63 files and the tests passed in the unpacked copy.

### Test results by machine

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-16 | macOS arm64 (Apple M4 Pro) | Python 3.13.2, numpy 2.5.5, scipy 1.18.1, gymnasium 1.4.0, torch 2.15.0, pytest 9.2.0 | 212 passed, 0 skipped |
| 2026-11-16 | macOS arm64 (Apple M4 Pro) | Python 3.12.16, numpy 1.26.4, pytest 8.0.0 (no scipy, gymnasium, torch) | 203 passed, 9 skipped (5 need scipy, 3 gymnasium, 1 torch) |
| 2026-11-16 | macOS arm64 (Apple M4 Pro) | Python 3.13.2, numpy 2.5.5, scipy 1.18.1, pytest 9.2.0 (no gymnasium, torch) | 208 passed, 4 skipped (3 gymnasium, 1 torch) |
| 2026-11-03 | macOS arm64 | Python 3.13 | 184 passed, 5 skipped (gymnasium not installed) |
| 2026-11-02 and earlier | machine not recorded | not recorded | see the entries below |

## 2026-11-02, v0.3: general robot learning, not one course
- Removed the course framing: the assignment tutoring rule, "lecture notes outrank everything", course logistics (`course.md`), Owen-specific project notes (`projects.md`), course folder paths, lecture dates and pre-lecture tags.
- Lecture notes kept as one cited source (key `Notes N`) alongside AJKS, Sutton and Barto, and BBC, defined in the new `references/sources.md`, which also holds every errata table (Notes 1 to 7, previously split between notation.md and q-learning.md).
- New `references/concepts/resets-and-exploration.md`, generalized from the RVFS project plan (access protocols, methods, how to run a reset experiment).
- The G1 num_envs sweep is now a case study in `training/on-policy-at-scale.md`, with its data in `assets/`.
- `notation.md` keeps general symbols and tabular / rollout array conventions; assignment-specific shapes dropped.
- Assets: `experiment_plan.md` replaces the course proposal outline; the write-up skeleton no longer names the course.
- Evals: three course-specific cases replaced (Q-learning chattering, LSVI-UCB beta, PPO vs FastTD3 experiment design).
- New test `test_skill_is_not_framed_around_one_course` fails if course-specific framing comes back.
- Test result, rerun 2026-11-03 on macOS arm64, Python 3.13: 184 passed, 5 skipped (4 need gymnasium, not installed; 1 skip is by design). Earlier counts in this file: machine not recorded.

## 2026-11-02, v0.2: renamed cs291k-toolkit to robot-learning-expert
- Audit: the attached zip and the installed skill were byte-identical (built Oct 22). Its 34 tests passed (3 skipped without gymnasium; machine not recorded). Gaps found: Notes 5 to 7 (Q-learning, published Oct 29) missing, the project file still described RVFS while the group project had become massively parallel G1 walking, no training-practice material, no PG/AC/IL beyond seeds, three new PDFs undigested.
- Scope: RL concepts for robot learning plus how robot policies are actually trained. Course material kept and reorganized under references/concepts and references/course.md.
- New concept file q-learning.md from Notes 5 to 7, with a typo table; two claims checked against the original papers (Jin et al. 2018: T is total steps; Jin et al. 2020: targets recomputed each episode, beta applied once, beta = c d H sqrt(iota)).
- New pre-lecture files: policy-gradient.md (with rsl_rl 5.4.2 PPO details read from source), actor-critic.md, imitation.md. Old upcoming-units.md retired.
- New training references: on-policy-at-scale (G1 sweep lessons), off-policy-at-scale, reward-and-observation-design (mjlab 1.6.0 G1 task read from source), sim-to-real, simulators-and-tooling, diagnostics (PPO health table), evaluation.
- New paper notes: supermimic, rl-scaling-laws, massively-parallel-survey, canon-algorithms and canon-robotics (verified against arXiv and proceedings pages).
- projects.md replaces project.md: group G1 project plus the RVFS thread.
- New scripts rl_math.py and batch_math.py; examples ppo_pendulum.py, analyze_sweep.py, exploration_chain.py; assets/g1_sweep_results.csv; tests for all of them plus test_skill_structure.py (every referenced path exists, frontmatter, no em dashes, every script tested); evals/evals.json with 10 cases.
- Independent fact-check pass (separate agent, sources only): fixed 7 errors (IQM with 3 seeds equals the mean, not the median; big-N runs are worse per sample, not better; DQN's SGD steps per FQI backup are C = 10,000, not small; missing transpose in the LSVI-UCB eigen view, now also a Notes 6 typo row; the d factor in beta comes from covering the value class, not a missing H; SAPG's and Beukman's explanations for PPO saturation stated separately from the update-count hypothesis; ppo_clip_objective mask now False when A = 0) and 14 imprecisions (PPO clip bounds neither ratio nor KL, gamma^t in the PG theorem, asymmetric-critic bias caveat, symbol definitions).
- Measured time-out bootstrapping on Pendulum (80k steps, gamma 0.99, 3 seeds): correct -130.8 mean vs terminal-treated -146.7 mean; small and noisy, consistent with Andrychowicz et al.; docs now say when truncation bias matters.
- Fixed local_sim.frontier_search_lock docstring: expected steps are about H(A+1)/2, not H*A/2 (measured by examples/exploration_chain.py).

## 2026-10-22, v0.1
- Built from the CS291K folder: syllabus (rev. A), Notes 0 to 4, Assignments 1 and 2 (handouts, skeleton docstrings, tests), 11 papers, AJKS 2026 draft and Sutton and Barto TOCs.
- Covered: course facts, notation and code layout, MDPs and exact DP, LQR and trajectory optimization, bandits, model-based RL.
- Pre-lecture seeds: tabular/linear/deep Q-learning, policy gradient (foundations, PPO, theory), actor-critic, learning from demonstrations.
- Paper notes for all 11 PDFs plus a theory map.
- Scripts: mdp_tools.py, lqr.py, local_sim.py, run_stats.py, with 37 passing tests.
- Next: when Notes 5+ and the A3/A4 handouts arrive, rewrite the matching sections of upcoming-units.md and add their array conventions to notation.md.
- Fact-checked the course references against the PDFs; fixed 3 errors (Thm 3 constant, LinUCB ellipsoid norm, A2 grading criteria) and expanded the notes-typo table.
