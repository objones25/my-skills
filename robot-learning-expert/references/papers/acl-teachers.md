# Automatic curriculum teachers: what to default to

Read it when choosing or debugging a task or level sampler (terrain, command range, goal). Mostly small benchmarks, not thousands of parallel envs; verified only against the cited lines, everything else is from the findings report (unverified).

## TeachMyAgent (Romac, Portelas, Hofmann, Oudeyer; arXiv 2103.09815)

- Benchmark: teachers pick procedural-environment parameters once per episode; students train 20M steps; five challenges, each in no, low and high expert-knowledge setups [§5.1, lines 349-356]. Seed count and the 6-teacher list were not re-read.
- Findings [lines 415-430]: ALP-GMM, with no expert knowledge, reached 80% final performance versus 68% for an Oracle teacher (p < 0.02), and beat Covar-GMM and RIAC. ADR and GoalGAN fell below Random: ADR's expansion can get stuck behind very hard or unfeasible subspaces (significant for ADR, p < 0.05), and GoalGAN is too inert to follow a student that regresses (forgetting). All teachers failed on the Parkour environment. [Parkour result: abstract line 40.]
- Teachers that need an easy initial distribution (ADR, GoalGAN) depend on expert knowledge [lines 419-421].

## Other teachers

- TSCL (1707.00183): bandit over tasks by learning progress; windowed or sampled estimates, K = 10 (`concepts/bandits.md`).
- Graves et al. (1704.03003): Exp3.S over curriculum items.
- Uniform is a strong baseline in several of these benchmarks (findings report); gains appear when many tasks are unfeasible or trivial, or exploration is hard.
- At large N: rules defined in episodes ("refit every 150 episodes", "last 250 pairs") have no tested mapping to thousands of envs; define clocks in optimizer iterations [inference]. UED-style samplers at scale: massively-parallel-rl.
