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
| Plateau on tasks with narrow start states, continuous actions | Duplicate data across envs (SAPG) | Action std collapsed; many envs in near-identical states | SAPG/EPO-style diversity, or a wider reset distribution |
| Plateau on procedural levels or terrains | Batch spent on solved or impossible configurations | Per-level success rates: many at 0% or 100% | Success Guided Sampling or SFL-style learnability sampling, p(1-p) |
| Value loss stays high on time-limited tasks; returns lower than published | Time-limit truncation treated as termination | Does `done` include the time limit? Does the GAE mask use it? (It does in PureJaxRL, rejax, purejaxql, Craftax_Baselines; rsl_rl bootstraps with the pre-step value, an approximation) | `scripts/mprl_common.AutoReset` + reward folding, or Brax-style masking (`jax-stack.md` section 4) |
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
   PQN) before any scaling run.
4. On a time-limited env, compare the default run against `--truncation-as-terminal`
   once, so the size of the truncation effect is known for that task.
5. Three seeds before any comparison, IQM for the summary.
