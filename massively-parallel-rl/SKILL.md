---
name: massively-parallel-rl
description: Massively parallel reinforcement learning - thousands to millions of GPU/JAX environments stepped in lockstep - grounded in 2024-2026 papers and verified against 22 repos (PureJaxRL, PQN, Craftax, Kinetix, Brax/Playground, Isaac Lab, rsl_rl, mjlab, FastTD3, FastSAC, FlashSAC, SAPG, PufferLib and others). Use when writing, scaling, tuning or debugging RL that runs many vectorized environments (jax.vmap env.step, gymnax, Craftax, Brax, MJX, Isaac Lab, num_envs in the thousands), when choosing num_envs / batch size / minibatches / learning rate as compute grows, when PPO plateaus or stops improving with more environments, when fitting or applying RL scaling laws (compute-optimal model size, isoFLOPs, Elo instead of mean return), or when picking between PPO, PQN, FastTD3 and FlashSAC for fast training. Ships runnable JAX PPO/A2C and PQN scaffolds, a single-agent Elo league, a scaling-law fitter and throughput/FLOPs tools. Not for distributed training of one large model across GPUs.
license: MIT
compatibility: Python 3.10+. Scripts need jax, flax, optax, distrax, gymnax, numpy, scipy, pandas (craftax optional). Tested on JAX 0.12.0.1 / Flax 0.12.12 / Optax 0.2.8 / gymnax 0.0.9 on CPU. Reference files cite repo commits read on 2026-10-27.
metadata:
  author: owenbeckettjones
  version: '1.0.0'
  verified-against: "jax 0.12.0.1, flax 0.12.12, optax 0.2.8, distrax 0.1.9, gymnax 0.0.9, torch 2.14.1"
---

# Massively parallel RL

## Start here

**Step 1 - know the hardware.** Run `python scripts/check_env.py`. It prints versions
and the accelerator JAX and PyTorch actually see. A JAX install that fell back to CPU,
or a Mac (no GPU simulators; JAX Metal is experimental), changes every recommendation
about N and wall-clock. Quote its output.

**Step 2 - route.** Read only the files the task touches.

| Read | When the task is |
| --- | --- |
| `references/jax-stack.md` | writing or reading jitted JAX RL (PureJaxRL loop, gymnax, Craftax, Kinetix, Brax/Playground), reset and truncation semantics, RNN rollouts, multi-device, shipped defaults per repo |
| `references/pytorch-gpu-sim.md` | Isaac Lab, rsl_rl, mjlab, MuJoCo Warp, FastTD3, FastSAC (holosoma), FlashSAC, PufferLib, keeping a PyTorch loop on the GPU, off-policy scaling results |
| `references/large-batch-on-policy.md` | PPO/A2C with very large N, plateaus, SAPG, EPO, curriculum resets, batch schedules |
| `references/scaling-laws.md` | predicting hyperparameters, model size or performance at larger compute; Elo league; reading Zheng et al. 2026 |
| `references/debugging.md` | any symptom: plateau, NaN, slow steps, OOM, results that change with N |
| `references/papers.md` | which paper or repo says what, with verified links and licenses |

**Step 3 - start from a scaffold, not from scratch.**

| Script | What it does |
| --- | --- |
| `scripts/ppo_jax.py` | PPO or `--algo a2c` (Zheng et al. setting), discrete or continuous gymnax/Craftax envs, whole run in one `jit(vmap(train))` |
| `scripts/pqn_jax.py` | PQN: Q(lambda) with LayerNorm, no replay buffer, no target network |
| `scripts/mprl_common.py` | truncation-aware auto-reset wrapper; fixed-instance evaluator |
| `scripts/throughput_bench.py` | env steps/s vs N; finds where the hardware saturates |
| `scripts/flops_per_frame.py` | FLOPs per frame (FPF), the model-size unit for RL scaling laws |
| `scripts/elo_league.py` | single-agent Elo from per-instance returns (Bradley-Terry, random = 0) |
| `scripts/fit_scaling_laws.py` | `hparams` / `isoflops` / `predict`: the two-stage method of Zheng et al. |

Both scaffolds write `--results-csv` rows for `fit_scaling_laws.py` and `--eval-csv`
rows for `elo_league.py`, so a sweep, a fit and a prediction chain together. Run
`python -m pytest tests/` after editing any script.

## Mental model

N environments each take T steps per rollout, so one update sees a batch of N x T
transitions. Three budgets trade against each other: **frames** (environment steps, the
sample-efficiency axis), **FLOPs** (C = FPF x frames, the compute axis) and
**wall-clock** (set by hardware throughput, which saturates at some N). Say which one
the user is optimizing: the Craftax record of Zheng et al. is compute-optimal and used
135B frames against 4B for the previous best.

On-policy methods throw each batch away after a few passes, so N buys lower-noise
gradients and more optimizer steps. Off-policy methods keep a replay buffer, so N buys
fresh data per step and the update-to-data ratio (UTD, gradient steps per new
transition) becomes the knob.

## Non-negotiables

Each of these, when violated, produces a run that trains but quietly learns less.

1. **Separate termination from truncation.** A time limit is not the end of the task.
   PureJaxRL, rejax, purejaxql, gymnax, Craftax and Kinetix all OR the time limit into
   `done`, which zeroes the bootstrap. Bootstrap truncated steps from the pre-reset
   observation (`mprl_common.AutoReset`), or mask them out as Brax does. The obs returned
   on a done step belongs to the next episode; never bootstrap from it.
2. **Scale N without changing three things at once.** More envs change batch size,
   minibatch count and update count together. Default recipe (Beukman et al. 2026):
   hold minibatch size and learning rate fixed and let the number of minibatches grow.
   Grow the minibatch only for hardware utilization, and say so.
3. **Keep the loop on the device.** JAX: jit the entire `train` function, vmap seeds,
   no Python inside the scan. PyTorch: no `.item()`, `.cpu()` or NumPy env I/O per step;
   GPU replay buffer; log every K steps.
4. **Seeds: vmap at least 3, report the IQM** (mean of the middle 50%). Comparisons that
   decide anything need 5 or more and a bootstrap or confidence interval.
5. **Count compute in FLOPs.** Report C and FPF next to frames when comparing configs
   or predicting larger runs (`flops_per_frame.py` states its accounting assumption).
6. **Fit scaling laws on Elo, not mean return.** Evaluate every agent on the same fixed
   instances, build one league including a random policy, anchor random at 0. Mean
   return extrapolated poorly in Starpilot and did not even fit the fitting range in Craftax.
   The instances must actually differ: gymnax MinAtar has 1-2 start states and
   deterministic steps, so add sticky actions (`--sticky-prob 0.1`) there.
7. **Normalize what grows.** LayerNorm after hidden layers for PQN and deep critics;
   observation normalization (running mean/var, outside the vmap) for continuous control.
8. **Never quote a default, throughput or benchmark number from memory.** Use the
   tables in `references/` (each cites file:line or a paper), or measure with
   `throughput_bench.py`. Repos pin old JAX versions; check before promising compatibility.

## Defaults to reach for

| Decision | Default | Reference |
| --- | --- | --- |
| Discrete actions, JAX env | PPO (`ppo_jax.py`) or PQN (`pqn_jax.py`, cheaper per frame, no replay) | `jax-stack.md` |
| Legged robots on Isaac Lab or mjlab, on-policy | rsl_rl PPO with the shipped velocity config (4096 envs x 24 steps, adaptive KL) | `pytorch-gpu-sim.md` section 2 |
| Continuous control on a GPU simulator, wall-clock matters | FastTD3; FastSAC (holosoma) for humanoids; FlashSAC when a bigger network is wanted | `pytorch-gpu-sim.md` |
| Craftax baseline to beat | Craftax_Baselines PPO (1024 x 64, lambda 0.8); Transformer-XL PPO for memory | `jax-stack.md` section 3 |
| Continuous control in JAX (Brax/Playground) | Brax PPO with Playground's per-task config | `jax-stack.md` section 10 |
| Rollout length T | 16 to 128 for PPO (Zheng et al. A2C: 16; PureJaxRL: 128); 32 for PQN | `jax-stack.md` |
| Minibatch size when N grows | fixed; derive minibatch count from N x T | `large-batch-on-policy.md` |
| Learning rate as compute grows | nearly constant (Zheng et al. exponents -0.023 and -0.082 per decade) | `scaling-laws.md` |
| Batch size as compute grows | fit it; published laws give x1.8 to x2.5 per 10x compute for A2C | `scaling-laws.md` |
| Model size as compute grows | fit it; published FPF laws give about x2.7 per 10x compute | `scaling-laws.md` |
| Optimizer | Adam, eps 1e-5, global-norm clip 0.5 (PPO); RAdam, clip 10 (PQN MinAtar) | `jax-stack.md` |
| Performance metric for scaling | single-agent Elo league | `scaling-laws.md` section 3 |
| PPO plateau at large N | add minibatches, then diversity (SAPG/EPO) or reset curricula | `large-batch-on-policy.md` |

## Debugging protocol

A scaled-up run underperforms: run the **three-run triage** before touching anything
else (small config that worked; large N with the small minibatch size; large N as run),
3+ seeds each, compare IQMs. `references/debugging.md` maps each outcome and each
symptom to a fix.

## Writing the answer

- Lead with the recommendation, then the numbers that justify it.
- Deliver what was asked before building more. A study plan is a plan with budget
  arithmetic and commands that call the bundled scripts; write it first, and build a new
  pipeline only if the user asks. Keep verification runs to a few minutes on CPU and say
  what they do and do not show.
- Show batch arithmetic explicitly: N, T, N x T, minibatch size, minibatch count,
  updates, total frames, C.
- Name the source of every hyperparameter: a paper table, a repo file:line, or a
  measurement you ran. Say when a number is a starting point to tune rather than a result.
- Define each non-obvious term on first use (truncation, IQM, FPF, UTD, isoFLOPs, Elo).
- Run the scaffold or script and quote its real output instead of describing what it
  would print.
- Shell commands meant to be pasted carry no `#` comment lines.
