# Debugging massively parallel RL

Find the symptom, run the check, then apply the fix. Most "the algorithm stopped
scaling" reports are one of the first five rows.

## Contents
1. Symptom table
2. The three-run triage
3. Throughput problems
4. Checks to run on any new setup

## 1. Symptom table

| Symptom | Likely cause | Check | Fix |
| --- | --- | --- | --- |
| Return plateaus below what smaller runs reached; adding envs does not help | Outer PPO step too large for the update noise (Beukman et al. 2026) | Hold minibatch size fixed and double N (more minibatches). Plateau rises: this was it | Add minibatches, not minibatch size; consider fewer epochs or a tighter clip as the budget grows (`large-batch-on-policy.md`) |
| Plateau on tasks with narrow start states, continuous actions | Duplicate data across envs (SAPG), or training that revisits the same narrow region of state space (OmniReset) | Action std collapsed; many envs in near-identical states; success per reset family, not only from the default start | SAPG/EPO-style diversity, or a validated reset bank (`reset-and-task-sampling.md` section 3); more envs help only once resets are diverse |
| Plateau on procedural levels or terrains | Batch spent on solved or impossible configurations | Per-level success rates: many at 0% or 100% | Success Guided Sampling or SFL-style learnability sampling, p(1-p), with an easy-start control; check the signal is not blind to far-from-solved tasks that are flat, and keep a uniform floor (`reset-and-task-sampling.md` section 4) |
| Return or value-loss spikes with period about H/K updates, or instability that worsens as N grows; fixed-horizon task, short rollout K | All envs reset together, so each batch holds the same slice of the horizon (2511.21011) | Histogram of episode step counts across envs at each update: one narrow peak means synchronized | Stagger initial offsets (NB about H/K groups, offset step K); locomotion with falls desynchronizes already and showed no gain |
| Return collapses or plateaus late in a long run, policy entropy high, pre-activation norm rising, dead units rising | Plasticity or feature-rank collapse (Moalla et al. 2405.00662; single-task Atari and MuJoCo PPO, not massively parallel; env count not checked) | Penultimate-feature rank (99% of variance), dead-unit fraction, pre-activation norm, ratio excess over the clip range | Candidates from that paper: fewer epochs (more epochs sped up the growth of pre-activation norm, their Fig. 1), a penalty on the actor's pre-activations (PFO), resetting Adam's moments after each batch or a faster second-moment decay (reduced feature norm and ratio excess, but not sufficient to prevent collapse). Applicability to large N is untested [inference]; Mayor et al. 2506.03404 saw dormant neurons fall as N_envs rose on Atari-10 |
| Same seed, different curve on a GPU simulator | GPU-batched physics is not reproducible at the precision GPUSimBench reports, even with fixed seeds: GPUSimBench Type 1 (Isaac Lab, ManiSkill) varies between envs inside a run, Type 3 (Madrona, MuJoCo Warp) varies both between envs and between runs, Type 4 (Playground) between runs, Type 2 (Genesis, MJX) neither (2607.13059 §IV; one RTX 5070, single toy scenes, Isaac Lab 2.2.1, MJX 3.4.0, Playground 0.1.0, Genesis 0.3.11, ManiSkill 3.0.0b22) | Repeat a run with fixed seed and randomization off | Average over seeds and do not attribute a single-run gap to the algorithm; MuJoCo Warp also runs on CPU for determinism (its docs, not GPUSimBench; `pytorch-gpu-sim.md` section 1) |
| Value loss stays high on time-limited tasks; returns lower than published | Time-limit truncation treated as termination | Does `done` include the time limit? Does the GAE mask use it? (It does in PureJaxRL, rejax, purejaxql, Craftax_Baselines; rsl_rl bootstraps with the pre-step value, an approximation) | `AutoReset` in `scripts/mprl_common.py` + reward folding, or Brax-style masking (`jax-stack.md` section 4) |
| Training breaks right after resets; values spike | Bootstrapping from the obs returned on a done step, which is the next episode's reset obs | Inspect `next_obs` on done steps | Use the pre-reset `final_obs`; if the env cannot return it (Isaac Lab), do not bootstrap there |
| Logged episode return sits near 0 at times | Averaging over steps where no episode finished | Count finished episodes per log window | Average only finished episodes; log NaN or skip when none finished |
| Results shift when N changes even at the same frames | N also changed batch size, minibatch count and update count | Print batch, minibatches, updates for both runs | Change one knob at a time; with fixed minibatch size, N changes only the minibatch count |
| Learning-rate annealing ends early or late | Schedule counted in updates, optimizer steps counted in minibatches x epochs | Compare schedule length to `num_updates * epochs * minibatches` | Use optimizer-step counts (purejaxrl ppo.py:73-79 divides by minibatches x epochs) |
| NaN after many updates with big networks | No normalization in a deep or wide critic | Grad norms, activation scale per layer | LayerNorm after hidden layers (PQN, scaling-crl), weight projection (FlashSAC), grad clipping |
| Recurrent agent worse than feed-forward | Hidden state not reset at episode boundaries, or sequences shuffled across time | Inspect the reset mask and the shuffle axis | Reset with `last_done`; shuffle the env axis only (`jax-stack.md` section 5) |
| Big off-policy critic worse than small at the same batch | Batch too large for a small critic (TD-overfitting) or too small for a big one | Validation TD error vs training TD error | Scale batch with model size (Fu et al. 2025) |
| Scaling-law fit looks good, extrapolation misses badly | Mean return used as the metric, or a non-monotonic regime inside the fit range | Plot raw isoFLOPs curves; refit with Elo | Elo league; drop budgets below any model-size crossover (`scaling-laws.md` section 5) |
| Elo intervals look tight but rankings flip between leagues | Evaluation instances are copies of one or two games (deterministic env, greedy policy) | `elo_league.py` prints how many instance columns differ; the scaffolds warn at eval time | Sticky actions (`--sticky-prob 0.1`, MinAtar's default) or a stochastic eval policy |
| Different runs disagree on the "best" config | Too few seeds; single argmax of noisy cells | Seed variance vs gap between configs | IQM over 5+ seeds and bootstrap competitive configs |

## 2. The three-run triage

When a scaled-up run underperforms, run three short jobs at the same frame budget
before changing anything else:

1. The small configuration that worked.
2. The large-N configuration with minibatch size held at the small run's value (so
   only the minibatch count grew).
3. The large-N configuration as the user ran it.

If 2 matches or beats 1 and 3 lags, the minibatch size or learning rate scaling is the
problem. If 2 also lags, look at data diversity (rows 2 and 3 above) or at a bug that
only appears with many envs (reset handling, normalization statistics). Use 3 or more
seeds per arm and compare IQMs.

## 3. Throughput problems

| Symptom | Cause | Fix |
| --- | --- | --- |
| JAX run is 10-100x slower than expected | JAX fell back to CPU | `python scripts/check_env.py`; install the CUDA jaxlib |
| First iteration takes minutes | Compilation of the whole training program | Persistent compile cache (`JAX_COMPILATION_CACHE_DIR`); Kinetix/Craftax compiles are known to be long |
| Steps/s stops growing as N doubles | Hardware saturated | `scripts/throughput_bench.py`; pick N at the knee unless bigger batches are wanted for learning reasons |
| Craftax steps are slow | Auto-reset generates a world every step | Optimistic resets (purejaxql `OptimisticResetVecEnvWrapper`, ratio 16) |
| PyTorch GPU sim loop slow despite a fast simulator | Host syncs: `.item()`, `.cpu()`, NumPy env I/O | Keep tensors on device; log every K steps (`pytorch-gpu-sim.md` section 5) |
| OOM in JAX | Replay buffer or rollout too large; preallocation | Reduce num_envs or buffer; `XLA_PYTHON_CLIENT_MEM_FRACTION`; JaxGCRL scripts say to reduce `--num_envs` first |
| OOM in FastTD3 | GPU replay buffer | README order: buffer_size, then batch_size, then num_envs |
| Results differ between GPUs | TF32 matmuls on Ampere | `JAX_DEFAULT_MATMUL_PRECISION=highest` (Playground README) |

## 4. Checks to run on any new setup

1. `python scripts/check_env.py`: the framework sees the accelerator.
2. `python scripts/throughput_bench.py --env <env> --n 256 1024 4096 16384`: where
   throughput saturates.
3. One seed of the scaffold at small N reaches a known score (CartPole 500 for PPO or
   PQN) before any scaling run. A single low PQN seed is not proof of a bug: see
   "Reference runs" below.
4. On a time-limited env, compare the default run against `--truncation-as-terminal`
   once, so the size of the truncation effect is known for that task.
5. Three seeds before any comparison, IQM for the summary.

## Reference runs

These results are specific to the machine and versions shown. Compare the spread, not
the exact numbers.

`pqn_jax.py --env CartPole-v1`, greedy evaluation per seed, on 2026-11-03 on macOS arm64
(Apple Silicon, JAX CPU backend) with JAX 0.12.0.1, Flax 0.12.12, Optax 0.2.8:

| Timesteps | Seeds | Scores |
|---|---|---|
| 500,000 | 8 | 338.9, 500, 500, 500, 500, 500, 184.3, 449.5 |
| 1,000,000 | 2 | 500, 53.7 |
| 1,000,000 | 6 | 500, 53.7, 500, 500, 398.9, 500 |
| 1,000,000 | 8 | 500 for all 8 |
| 2,000,000 | 8 | 500 for all 8 |

In the 2-seed run, the low seed reached 500 at the halfway point before collapsing. Why
it collapses was not determined. A seed's result also depends on `--num-seeds`: seed index
1 scored 53.7 with 2 and 6 seeds and 500 with 8. So judge a run from several seeds, and do
not compare single seeds between runs of different widths.

An earlier version of the test required both of 2 seeds to finish above 400. This skill's
earlier "Tested on" note does not record the machine or that test's result. The current
test requires a median above 400 across 8 seeds, with at most 2 below 100.
