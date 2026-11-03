# Exploration with simulator resets

Online exploration can need exponentially many episodes (a random policy reaches the end of an H-step combination lock with probability A^−H). A simulator that can be restored to states you have already visited changes that: checkpoint the deepest state reached and explore from there, and the same lock takes about H(A+1)/2 steps (`examples/exploration_chain.py`). Robot learning nearly always trains in simulators, so throwing this access away is a choice, and Notes 7 §3.2 calls exploiting it under-researched. Theory: Mhammedi, Foster, Rakhlin, "The Power of Resets in Online RL" (NeurIPS 2024), digested in `papers/power-of-resets.md`; how its assumptions relate to coverability and Bellman-Eluder dimension: `papers/theory-map.md`.

## Access protocols

| Protocol | You may | Example | Exploration cost |
|---|---|---|---|
| online (episodic) | reset only to the start distribution μ | real robot, standard RL benchmarks | can be exponential in H without strong structure |
| local simulator | restore any state visited before | MuJoCo/mjlab state save and restore, game emulators | polynomial under coverability-type assumptions (RVFS, SimGolf) |
| generative model | query any (s, a) for a next-state sample | tabular theory, Notes 4 to 6 | no exploration problem; cost is samples per (s, a) |

`scripts/local_sim.py` implements all three behind one interface with query counting, budgets, checkpoint and restore tokens, observation-noise models (identity, one-hot plus Gaussian, Block-MDP emission, aliasing), and a Gymnasium wrapper that reseeds the env RNG on restore.

## Methods

- RVFS (Recursive Value Function Search): keeps core sets of states per layer, estimates values by Monte Carlo rollouts from restored states, and recurses when a confidence test says a value estimate is unreliable; the result is a non-executable policy that needs the simulator, then behavior cloning gives an executable one. Pseudocode, implementation (data structures, the linear closed-form confidence test, hand-picked constants, costs) and PDF ambiguities: `papers/power-of-resets.md` §5, §7, §9.
- GowU / Uncertainty Guided Tree Search (`papers/ugts.md`): practical reset-based exploration with experiments; a baseline and testbed source.
- Go-Explore (Ecoffet et al. 2019): archive of visited cells, return to promising ones, explore from there.
- Reset distributions in large-scale robot RL: return-based reset sampling in SuperMimic (`papers/supermimic.md`), success-guided sampling of start states (`papers/massively-parallel-survey.md`), and reference-state initialization in motion imitation (DeepMimic, `papers/canon-robotics.md`).

## Running a reset experiment

Start tabular, add difficulty one axis at a time:
1. Tabular layered combination lock, one-hot (tabular) value functions, no noise. Ground truth from `mdp_tools.finite_horizon_dp` (optimum) and `mdp_tools.finite_horizon_eval` (exact return of the learned policy). Success: learned return near optimal, small core sets, query counts far below A^H.
2. Observation noise through `local_sim.TabularSim(obs_fn=...)`: `block_obs` (decodable, inside RVFS's theory), `onehot_gaussian_obs` and `aliased_obs` (not decodable). Sweep the noise level.
3. Measure the simulator-backed policy and the behavior-cloned executable policy separately; the analysis in `papers/power-of-resets.md` §8.3 to 8.4 [inference] predicts the cloned policy degrades roughly like H times the per-step confusion rate.
4. Then a continuous environment through `local_sim.GymLocalSim`, or a neural value class.

Decisions that change results and must be stated:
- What a restore restores: the latent simulator state (what real simulators do) or an observation-keyed state.
- Whether a restore emits a fresh noisy observation or the stored one (`resample_obs_on_restore`).
- Whether the RNG is fresh after restore (it must be for Monte Carlo backups; `GymLocalSim` reseeds by default).
- Hand-picked constants versus the theory's values, which are astronomically large (`papers/power-of-resets.md` §7.5).

Hygiene:
- Count simulator queries, resets and restores, not just episodes; protocols differ in what a "sample" costs (`local_sim.Counters`).
- Baselines: exact DP (ground truth), an online method without resets (ε-greedy Q-learning or DQN), and a reset-based method without a value function (GowU).
- Seeds and intervals as in `training/evaluation.md`.
