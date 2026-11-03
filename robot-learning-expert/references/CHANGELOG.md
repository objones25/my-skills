# Changelog

## 2026-11-02, v0.3: general robot learning, not one course
- Removed the course framing: the assignment tutoring rule, "lecture notes outrank everything", course logistics (`course.md`), Owen-specific project notes (`projects.md`), course folder paths, lecture dates and pre-lecture tags.
- Lecture notes kept as one cited source (key `Notes N`) alongside AJKS, Sutton and Barto, and BBC, defined in the new `references/sources.md`, which also holds every errata table (Notes 1 to 7, previously split between notation.md and q-learning.md).
- New `references/concepts/resets-and-exploration.md`, generalized from the RVFS project plan (access protocols, methods, how to run a reset experiment).
- The G1 num_envs sweep is now a case study in `training/on-policy-at-scale.md`, with its data in `assets/`.
- `notation.md` keeps general symbols and tabular / rollout array conventions; assignment-specific shapes dropped.
- Assets: `experiment_plan.md` replaces the course proposal outline; the write-up skeleton no longer names the course.
- Evals: three course-specific cases replaced (Q-learning chattering, LSVI-UCB beta, PPO vs FastTD3 experiment design).
- New test `test_skill_is_not_framed_around_one_course` fails if course-specific framing comes back.

## 2026-11-02, v0.2: renamed cs291k-toolkit to robot-learning-expert
- Audit: the attached zip and the installed skill were byte-identical (built Oct 22). Its 34 tests passed (3 skipped without gymnasium). Gaps found: Notes 5 to 7 (Q-learning, published Oct 29) missing, the project file still described RVFS while the group project had become massively parallel G1 walking, no training-practice material, no PG/AC/IL beyond seeds, three new PDFs undigested.
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
