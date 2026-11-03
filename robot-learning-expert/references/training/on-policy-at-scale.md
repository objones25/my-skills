# Massively parallel on-policy training (PPO on GPU simulators)

GPU simulators step thousands of robots in lockstep, so data stops being the bottleneck and the questions become: how to split a fixed batch between envs and steps, how to keep PPO stable with huge batches, and how to compare runs fairly by wall-clock. This file is the recipe plus a case study (Unitree G1 walking at 64 to 16384 envs). Sources: `papers/legged-gym.md`, `papers/sapg.md`, `papers/rl-scaling-laws.md`, `papers/massively-parallel-survey.md`, rsl_rl 5.4.2 / mjlab 1.6.0 source, and the case-study run's results (`assets/g1_sweep_results.csv`).

## Batch arithmetic (state it whenever it changes)

| Quantity | Formula | G1 shipped config |
|---|---|---|
| batch per iteration B | N envs × T steps per env | N × 24 |
| minibatch size | B / num_mini_batches | B / 4 |
| gradient steps per iteration | epochs × num_mini_batches | 5 × 4 = 20 |
| env steps per iteration | B | |
| sim time per iteration | T × policy dt | 24 × 0.02 s = 0.48 s |
| total env steps | B × iterations | |

`scripts/batch_math.py` prints this table and a cost estimate (iterations/minute × minutes × price) for any config.

With num_mini_batches fixed, growing N grows the minibatch: at 16384 envs each minibatch is 98,304 samples. Gradient steps per iteration stay at 20, so more envs buy a lower-noise gradient, not more updates. One hypothesis for why returns per wall-clock minute flatten past some N [inference]: the update count, not the data, becomes the limit. The papers give other explanations: SAPG argues one Gaussian policy makes the extra samples redundant (a diversity problem) and splits envs into leader and follower policies; Beukman et al. argue the outer step is too large relative to the update noise and hold minibatch size fixed while adding minibatches (`papers/sapg.md`, `papers/massively-parallel-survey.md`).

## The recipe (Rudin et al. 2021, as shipped in rsl_rl and mjlab)

1. Everything on the GPU: sim, observations, rewards, policy, storage, update. No per-step `.cpu()` or `.item()` in the rollout (each forces a device sync).
2. Short rollouts, long episodes: T ≈ 24 steps per env per iteration while episodes last 20 s (1000 policy steps), so one episode spans many updates. Legged_gym found PPO struggled below about 25 consecutive steps (0.5 s) because GAE needs a few steps of rewards.
3. Huge minibatches (tens of thousands of samples), few epochs.
4. Bootstrap time-outs, not falls (`concepts/policy-gradient.md` §4 for exactly how rsl_rl does it).
5. Adaptive learning rate from the KL to the old policy (desired_kl 0.01).
6. Randomize initial episode lengths so envs do not all time out on the same iteration.
7. Curriculum on command range or terrain difficulty, advanced by performance or step count.

## Case study: a G1 num_envs sweep (Oct 29 to 30, 2026)

Setup: `Mjlab-Velocity-Flat-Unitree-G1`, shipped PPO config unchanged except num_envs and seed; N ∈ {64, 256, 1024, 4096, 16384} × 3 seeds × 45 training-clock minutes on one A100 80GB; offline evaluator on 6 fixed commands. Data: `assets/g1_sweep_results.csv`; reproduce the table with `python examples/analyze_sweep.py`.

Seed-mean vel_error_xy (m/s, lower is better) by training-clock minute:

| N | 6 min | 12 min | 24 min | 45 min | iterations by 45 min |
|---|---|---|---|---|---|
| 64 | 0.80 | 0.63 | 0.49 | 0.50 | ~4,300 |
| 256 | 0.61 | 0.50 | 0.51 | 0.48 | ~3,800 |
| 1024 | 0.50 | 0.51 | 0.44 | 0.14 | ~3,350 |
| 4096 | 0.50 | 0.32 | 0.10 | 0.082 | ~2,570 |
| 16384 | 0.49 | 0.51 | 0.081 | 0.077 | ~1,280 |

What it shows:
- 0.50 is "standing still": the mean commanded speed over the 6 evaluation commands is (0.5 + 1.0 + 0.5 + 0.5 + 0.5 + 0)/6 = 0.5 m/s, so a robot that balances but never walks scores 0.50. Small-N runs learn not to fall (no_fall_fraction reaches 1.0) and sit in that local optimum; larger N leaves it sooner [inference from results.csv].
- 16384 envs matches 4096 at 45 min with half the iterations; past ~4096 the extra data mostly buys smoother gradients. Per wall-clock minute the two are equal; per sample 16384 is about 2× worse (about 502M vs 253M env steps to reach the same 45-min error).
- Training reward misleads across N: compare with a fixed offline evaluator (same commands, seeds, horizon, env count for every checkpoint).

Fairness rules the sweep adopted (worth reusing in any scaling comparison):
- Time only rollout + returns + update, CUDA-synchronized before each start and stop; checkpoint I/O, logging, eval and rendering are off the clock (`mprl/clock.py`).
- Disclose step-count curricula: mjlab's `commands_vel` curriculum advances on `env.common_step_counter`, which ticks once per env.step regardless of N, so small-N runs (more iterations per minute) reach harder command stages sooner. That favors small N, so the measured speed-up of large N is if anything understated.
- Fixed per-step overheads (rsl_rl's per-step episode bookkeeping on the CPU) weigh more on small N, which biases the other way; say so.
- A curve point at mark m uses the checkpoint that crossed m (first saved at or after m), the same rule everywhere.
- Re-randomize episode lengths on every resume: a checkpoint does not store `episode_length_buf`, and after a resume all envs timed out together, making Episode_Reward alternate between full and near zero (bug found Oct 30, fixed in 68a30e8).

## What to log (every iteration unless noted)

rsl_rl's own `Loss/*`, `Policy/*`, `Perf/*`, `Train/mean_reward`, `Train/mean_episode_length`, per-term `Episode_Reward/*`, and `Curriculum/*`. rsl_rl 5.4.2 does not return approximate KL, clip fraction or explained variance; recompute them yourself off the training clock every few iterations (the case-study code did every 10):
- approx KL with Schulman's k3 estimator mean((ρ − 1) − log ρ), ρ = π_new/π_old on rollout samples. It includes observation-normalizer drift, as PPO's own ratio does.
- clip fraction mean(|ρ − 1| > clip_param).
- explained variance 1 − Var(R − V)/Var(R), undefined when returns are constant.
Healthy ranges and what to do when they are not: `diagnostics.md`.

## Scaling science

Zheng et al. 2026 (`papers/rl-scaling-laws.md`) fit scaling laws for RL compute, data and model size; a survey doc indexes its 35 references and the recent massively parallel work (`papers/massively-parallel-survey.md`). Use them to frame "how to spend extra data" questions; the G1 sweep above is a one-axis (N) slice of that question.
