# Exploration with simulator resets

Online exploration can need exponentially many episodes (a random policy reaches the end of an H-step combination lock with probability A^−H). A simulator that can be restored to states you have already visited changes that: checkpoint the deepest state reached and explore from there, and the same lock takes about H(A+1)/2 steps (`examples/exploration_chain.py`). Robot learning nearly always trains in simulators, so throwing this access away is a choice, and Notes 7 §3.2 calls exploiting it under-researched. Theory: Mhammedi, Foster, Rakhlin, "The Power of Resets in Online RL" (NeurIPS 2024), digested in `papers/power-of-resets.md`; how its assumptions relate to coverability and Bellman-Eluder dimension: `papers/theory-map.md`.

## Access protocols

"Reset" means something different in each row, so say which one you have. Weakest to strongest [Access §2.2; reset-free rows from `reset-free-and-goal-conditioned.md`]:

| Protocol | You may | Example | What is provably possible (conditions) |
|---|---|---|---|
| reset-free (non-episodic) | no reset to μ; manual or learned resets | real robot with no operator | method-dependent; irreversible states are a hazard (`reset-free-and-goal-conditioned.md`) |
| online (episodic) | reset only to the start distribution d1 | real robot, standard RL benchmarks | can be exponential in H without strong structure |
| μ-resets (Kakade-Langford) | also roll out from an exploratory reset distribution μ_h | simulator with a state sampler (grasp sampler, demo states) | PSDP needs policy completeness, then poly in C_conc [Access Thm 3 and App. B]; agnostic Π: 2^Ω(H) with C_conc = 6 [Access Thm 3]; realizable Π: 2^Ω(H) with C_all <= 4, but exp(Θ(√H)) with bounded pushforward concentrability [Reset-μ Thm 1 to 3, single-author Aug 2026 preprint that discloses GPT-assisted proofs] |
| local simulator (RLLS) | restore any state visited before | MuJoCo/mjlab state save and restore, game emulators | poly in coverability needs value-class realizability (Q*; [PoR Thm 3.1]); for agnostic policy classes it is NOT sufficient: 2^Ω(H) with C_cov = 2 and policy completeness [Access Thm 2]; minimax is Θ(C_span(Π)), exponential for e.g. linear policies [Access §3.1, citing JLR+23] |
| hybrid resets | local simulator plus μ-resets | both of the above | agnostic Π tractable for Block MDPs with bounded pushforward concentrability and any Π: poly(C_push, S, A, H, 1/ε, log|Π|) [Access Thm 4]; paper notes the factorized-μ assumption is removable (App. F.1); Block MDPs only |
| generative model | query any (s, a) for a next-state sample | tabular theory, Notes 4 to 6 | no exploration problem for model estimation and planning; for policy search over a restricted class Thm 2 still gives 2^Ω(H) samples (proper deterministic learner, policy completeness, C_cov = 2) |
| known model | dynamics given | planning | computation only |

Source (tags [Access] and [Reset-μ] above): Krishnamurthy, Li, Sekhari, "The Role of Environment Access in Agnostic RL" (arXiv 2504.05405, Apr 2025; theory only, no experiments found in the text) [Access]; Reset-μ = arXiv 2608.07772 (Aug 2026). RLLS (reset to any visited state) and μ-resets are different access models [Access §2.2]. Reading for robots [inference]: a grasp or state sampler that feeds resets is μ-resets, and the 2504.05405 lower bounds say a good μ alone does not make policy search explore.

`scripts/local_sim.py` implements all three behind one interface with query counting, budgets, checkpoint and restore tokens, observation-noise models (identity, one-hot plus Gaussian, Block-MDP emission, aliasing), and a Gymnasium wrapper that reseeds the env RNG on restore.

## Methods

- RVFS (Recursive Value Function Search): keeps core sets of states per layer, estimates values by Monte Carlo rollouts from restored states, and recurses when a confidence test says a value estimate is unreliable; the result is a non-executable policy that needs the simulator, then behavior cloning gives an executable one. Pseudocode, implementation (data structures, the linear closed-form confidence test, hand-picked constants, costs) and PDF ambiguities: `papers/power-of-resets.md` §5, §7, §9.
- GowU / Uncertainty Guided Tree Search (`papers/ugts.md`): practical reset-based exploration with experiments; a baseline and testbed source.
- Go-Explore (Ecoffet et al.; cite the Nature version, arXiv 2004.12919, as the 1901.10995 header says): archive of visited cells, return to a promising cell, explore from there. It names two failure modes of earlier exploration: detachment (losing track of promising areas) and derailment (exploring before returning to the state) [2004.12919 abstract, lines 33, 66-67]. The "go" step restores the simulator state; a robustification phase then turns brittle trajectories into a policy with a modified backward algorithm (Salimans-Chen) [2004.12919 lines 120, 206]. Cell weight in the Atari settings is 1/sqrt(Cseen+1) [2004.12919 Extended Data Table 1]. Cheap deep-RL instantiations: restart from a buffer of visited states with probability λ (Go-Exploit; RGSC uses λ = 0.5 in AlphaZero-style self-play [2602.20809 lines 185, 862]). Selection-rule variants (Intelligent Go-Explore, TopoExplore) were not opened (unverified).
- Start-state distributions in practice, three families [inference grouping; papers opened as noted]:
  - reverse curricula from a demo or goal (Salimans-Chen, Backplay, RCG, RFCL, BaRC);
  - demo-state reset sets (Salimans-Chen, X-Reset);
  - generated or visited-state reset sets (OmniReset, Go-Exploit).
  Advance rules reported: Salimans-Chen moves the start back when at least ρ = 20% of workers match the demo return [1812.03381 lines 303-304]; RCG keeps starts with return in [Rmin, Rmax] = [0.1, 0.9] [1707.05300 line 583]; RFCL advances a demo's start when the last m rollouts from it succeeded [2405.03379 line 297]. Always evaluate on the nominal start distribution, not the curriculum's [practice].
- Cautions on when each fails: Backplay is "not a universal strategy" when s0 is randomized [1807.06919 lines 357-359]; RCG needs approximately reversible dynamics [1707.05300 line 65]; Salimans-Chen used 128 GPUs for 2 weeks and did poorly on Gravitar and Pitfall [1812.03381 lines 343, 354-355]; OmniReset depends on grasp-sampler quality [2603.15789 lines 685-686]. For the 1M-env version of reset sampling see massively-parallel-rl (its large-batch-on-policy reference).
- Real-robot variant: when an operator places objects the initial state is a decision; a Mulligan-style coverage and failure-promotion sampler is one answer [2610.05882, from the findings report; not re-opened here (unverified)].
- Reset distributions in large-scale robot RL: return-based reset sampling in SuperMimic (`papers/supermimic.md`), success-guided sampling of start states (`papers/massively-parallel-survey.md`), and reference-state initialization in motion imitation (DeepMimic, `papers/canon-robotics.md`).
- Asymmetric self-play (Alice/Bob) is a reset-based curriculum that needs a reversible or resettable environment (abstract only read, 1703.05407; unverified).

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
