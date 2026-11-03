# On-policy RL with very many environments

What happens to PPO and A2C as the number of parallel environments N grows from
thousands to a million, and what to do about it. Every claim cites the paper it comes
from; the papers were read in full or in their arXiv HTML.

## Contents
1. The plateau and its two explanations
2. Recipe: scaling N without stalling
3. Fixes that change what the batch contains
4. Choosing among them

## 1. The plateau

Plain PPO stops improving well below the optimum once N is large, and adding
environments past that point does not help. Two diagnoses exist and they suggest
different fixes.

**Duplicate data** ([SAPG, Singla et al., ICML 2024](https://arxiv.org/abs/2407.20230)).
A Gaussian policy samples actions near its mean, so tens of thousands of environments
largely repeat each other. A bigger batch adds samples but little information.
Evidence: PPO barely learns the hard AllegroKuka manipulation tasks at 24,576 envs.

**Step size too large for the noise** ([Beukman et al., RLJ 2026](https://arxiv.org/abs/2603.06009);
all four Zheng et al. authors are co-authors). Treat PPO's outer loop (collect,
update, repeat) as noisy stochastic optimization. If each policy change is large
relative to the noise in its estimate, the policy bounces around an optimum instead
of settling. The plateau is under-regularization, not a lack of exploration or
capacity. Two levers: shrink the outer step (stronger trust region, fewer epochs) or
collect more data per update (more envs). Tuning the learning rate cannot fix a bad
outer step: the two act through different mechanisms.

The diagnoses are compatible: duplicates make the effective sample count lower than
N x T, which is the noise side of the second account.

## 2. Recipe: scaling N without stalling

From Beukman et al. 2026, who ran over 1M environments on 128 GPUs:

1. Hold minibatch size and learning rate fixed. Increase the number of minibatches
   per update (num_minibatches = N x T / minibatch_size), so more environments mean
   more optimizer steps on lower-noise data.
2. Increase minibatch size, with a scaled learning rate, only when hardware
   utilization is the bottleneck. Larger minibatches sometimes destabilized training
   and lowered the plateau.
3. As the total interaction budget grows, the best outer step size shrinks, so the
   ratio of data collected to policy change allowed per update should rise.
4. Weaker regularization plateaus sooner as epochs increase; stronger regularization
   partly offsets it.

Results: standard PPO configs plateau before about 10B interactions; scaled past 1M
envs, PPO improved monotonically to 1 trillion steps. On Isaac Gym tasks, reverting
the minibatch from 96k to 16k improved PPO everywhere and narrowed its gap to SAPG.
For Kinetix they used a compromise for wall-clock speed: 32x the minibatches, each 16x
larger, with a 4x larger learning rate. For the small Kinetix distribution, 65k envs
was enough.

Tension with the scaling-law paper: Zheng et al. 2026 grow the minibatch with compute
(Craftax "batch size" is observations per minibatch). Under the Beukman recipe the
same budget would instead add minibatches. Nobody has compared the two at matched
compute; when a user's batch-size law is driving a minibatch size up, mention that
the alternative exists.

`scripts/ppo_jax.py` implements the recipe: it takes `--minibatch-size` and derives
the number of minibatches from N x T.

## 3. Fixes that change what the batch contains

| Method | Where it acts | Mechanism | Reported gain |
| --- | --- | --- | --- |
| [SAPG](https://arxiv.org/abs/2407.20230) (ICML 2024) | who collects data | Split N envs into M blocks, each with its own policy (shared backbone, small per-policy latent). A leader trains on its own data plus importance-weighted, subsampled follower data; followers get different entropy bonuses to stay diverse. 24,576 envs, M = 6. | Beats DexPBT by 12% to 66% on three AllegroKuka tasks after 2e10 samples; about 2x PBT on two-arm reorientation (28.6 vs 14.5 successes) |
| [EPO](https://arxiv.org/abs/2503.19037) (Wang, Su, Gupta, Pathak 2025) | who collects data | Population of agents sharing one actor-critic, each conditioned on a latent "gene"; only the genes evolve (elites kept, crossover by averaging, Gaussian mutation). Master agent aggregates population data SAPG-style. 64 agents x 384 envs. | Two-arm reorientation 35.8 vs SAPG 2.2 successes (SAPG paper itself reported 28.58 after 5e10 steps); seed variance 20% of mean vs 38% for best baseline |
| [Success Guided Sampling](https://arxiv.org/abs/2610.12465) (Zhang et al., Oct 2026) | which tasks start | Uniform resets waste a growing share of the batch on configurations already solved or not yet attemptable. Track per-configuration success over a recent window; sample moderate-success configurations more (beta-shaped weights), never zero. Up to 2^20 envs. | 13-terrain quadruped at 1M envs: 73% vs 54% for Prioritized Level Replay; Franka nut-and-bolt 70% vs 6% uniform |
| [Adaptive Batch Scaling](https://arxiv.org/abs/2605.21557) (Park 2026) | data per update | Grow the batch (via rollout length) as the policy stabilizes, measured by the share of up to 2,048 states whose greedy action changed since the last check. Forward passes only, unlike gradient-noise-scale methods. | On PQN, a fixed 16,384 batch hurt larger models while scheduling 2,048 to 16,384 helped; PPO gains shown only as curves |

### What the ABS code actually does (daisophila/ABS 660b0c1)

PyTorch on CleanRL: PQN + ABS on Atari via envpool, a PQN + gradient-noise-scale
baseline, and PPO + ABS on MuJoCo; no BTR code. The metric is called `policy_change`.
- PQN: every 50 iterations, the fraction of greedy actions that flip on up to 2,048 states
  resampled from the latest rollout (not a fixed reference set). That rate maps
  log-linearly onto rollout length 16 to 64 between thresholds 0.05 and 0.95, smoothed
  with a 0.9/0.1 moving average; more change gives a shorter rollout; epochs are rescaled
  with the rollout (algos/pqn_atari_envpool_ours.py:84-92, :392, :436-440).
- PPO: the same scheme with Gaussian KL, thresholds 0.01 and 0.1, rollout 1,024 to 8,192,
  every 10 iterations (algos/ppo_continuous_action_ours.py:83-91, :177, :219-234).
- As published, the PQN path builds its snapshot network with three arguments
  (`QNetwork(self.envs, self.args, self.device)`, line 415) while the class takes one
  (line 361), so it should fail at the first adaptation step. The repo has no license
  file, so treat the code as reference only.

### A built-in outer-step controller: rsl_rl's adaptive KL

rsl_rl, the PPO used by Isaac Lab and mjlab, already adjusts the learning rate every
minibatch to hold KL near a target: above 2 x 0.01 it divides lr by 1.5, below 0.005 it
multiplies by 1.5, within [1e-5, 1e-2] (rsl_rl ppo.py:240-266). It acts on the step-size
side of Beukman et al.'s account, though the paper notes the learning rate and the outer
step act through different mechanisms. Its default minibatch is large: 4096 envs x 24
steps / 4 minibatches = 24,576.

## 4. Choosing among them

- Plateau with high data duplication (continuous control, narrow initial-state
  distribution): SAPG or EPO; both need an off-policy correction and more code.
- Plateau with a wide task or level distribution (procedural levels, terrains, assembly
  variants): change the reset sampler first; it composes with any learner.
- Plateau that moves when you add minibatches: the Beukman recipe; cheapest to try,
  no new machinery.
- Early training unstable, late training noisy: a growing batch schedule.

Before any of these, check that the plateau is not a bug: truncation treated as
termination, stale observation normalization, or advantage normalization over the
wrong axis all produce plateaus too (see `debugging.md`).
