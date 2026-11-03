# The JAX stack: end-to-end jitted RL

Environment, rollout buffer and learner all live in one compiled XLA program on the
accelerator. Facts below were read from the repos at the commits listed at the end;
file:line references are into those clones.

## Contents
1. Loop anatomy
2. Running many seeds and hyperparameters in one program
3. Environments and their reset semantics
4. Termination vs truncation
5. Recurrent policies
6. Normalization wrappers
7. Multi-device
8. Logging and checkpointing inside jit
9. Performance and precision settings
10. Shipped defaults
11. Version traps

## 1. Loop anatomy

Every repo studied (PureJaxRL, rejax, purejaxql, Kinetix) uses the same shape:

```
train = make_train(config)            batch math done in Python, before tracing
jax.jit(train)(rng)
  scan over NUM_UPDATES
    scan over NUM_STEPS               rollout: policy(obs) -> vmap(env.step) over NUM_ENVS
    reverse scan                      GAE (PPO) or lambda-returns (PQN)
    scan over UPDATE_EPOCHS
      permute batch, reshape to (NUM_MINIBATCHES, MB, ...)
      scan over minibatches           value_and_grad -> apply_gradients
```

- `NUM_UPDATES = TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS`,
  `MINIBATCH_SIZE = NUM_ENVS * NUM_STEPS // NUM_MINIBATCHES` (purejaxrl/ppo.py:63-68).
  Shapes must be static, so these are Python ints computed before `jit`.
- Env step is `jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, states, actions, params)`:
  one key per env, params broadcast (purejaxrl/ppo.py:117-127).
- The GAE scan uses `reverse=True` and `unroll=16` (purejaxrl/ppo.py:143-164).
- Everything that varies per step (rng, env state, obs, train state) is threaded through
  the scan carry. Python side effects inside the traced function run once, at trace time.
- `scripts/ppo_jax.py` and `scripts/pqn_jax.py` are complete single-file versions of this
  loop with the fixes in sections 4 and 6.

## 2. Many seeds and hyperparameters in one program

- Seeds: `jax.jit(jax.vmap(make_train(config)))(jax.random.split(rng, S))`
  (purejaxrl walkthrough notebook :410-411; purejaxql pqn_minatar.py:459-461; rejax
  train.py:34-35). The README of PureJaxRL says 2048 PPO agents train in about half the
  time of one PyTorch PPO agent on one GPU (README.md:28).
- Hyperparameters: make the value an argument, `train(ent_coef, rng)`, and vmap twice,
  inner over seeds, outer over the value (walkthrough notebook :688). rejax marks tunable
  scalars `pytree_node=True` so they can be vmapped (algorithm.py:25-36).
- What can be vmapped: scalars such as learning rate, entropy coefficient, clip epsilon,
  gamma. What cannot: anything that changes an array shape (num_envs, num_steps, minibatch
  count, network width). For a scaling-law grid, vmap seeds and learning rates inside one
  program and loop over batch size and width in Python.
- Exception: purejaxql's Atari path cannot vmap seeds because envpool runs on the CPU
  (pqn_atari.py:457-460).

## 3. Environments and reset semantics

| Library | What it offers | Notes |
| --- | --- | --- |
| gymnax | classic control, MinAtar, bsuite in JAX | MinAtar games have 1-2 start states and key-independent steps (tested: Breakout 2, SpaceInvaders 1, Asterix 1, Freeway 256), and gymnax omits MinAtar's default sticky actions (p = 0.1); use `AutoReset(..., sticky_prob=0.1)`. `Environment.step` auto-resets: it runs `step_env` and `reset_env` every step, then selects per leaf with `lax.select` (environment.py:37-47). The returned obs on a done step is the reset obs. |
| Craftax 1.7.0 | Craftax and Craftax-Classic, symbolic or pixels | `make_craftax_env_from_name(name, auto_reset)`; names `Craftax-Symbolic-v1`, `Craftax-Pixels-v1`, `Craftax-Classic-Symbolic-v1`, `Craftax-Classic-Pixels-v1` (craftax_env.py:1-55). Symbolic obs is a flat [0,1] vector; achievements logged as `info["Achievements/<name>"]`, nonzero only on the done step (envs/common.py:5-11). `max_timesteps` 100000 (Craftax), 10000 (Classic). |
| Kinetix | procedurally generated 2D physics tasks (Jax2D) | done also fires on reward != 0 and on NaN state (env.py:120-141); long compile, use `train_levels=dummy env.dummy_env=True` to debug shapes (README.md:335) |
| MuJoCo Playground | MJX locomotion, manipulation, DM Control | `wrap_for_brax_training` = vmap (or domain-randomization vmap) -> Brax EpisodeWrapper -> its own `BraxAutoResetWrapper` (wrapper.py:87-132). Default auto-reset restores a cached first state and does not reset `info`; `full_reset=True` calls `env.reset` and is slower (:161-248). |
| Brax envs | physics in JAX | `EpisodeWrapper` sets `info["truncation"]` when the step count hits the limit (brax/envs/wrappers/training.py:110) |

### Craftax specifics (Craftax_Baselines 7ce36fa, transformerXL_PPO_JAX d914d55; both MIT)

- Official baselines: `ppo.py`, `ppo_rnn.py`, `ppo_rnd.py` (plus ICM and E3B flags) share
  one set of defaults: 1024 envs x 64 steps (batch 65,536), 4 epochs, 8 minibatches
  (8,192 each), lr 2e-4 annealed, gamma 0.99, gae_lambda 0.8, clip 0.2, ent 0.01, vf 0.5,
  grad norm 1.0, 1e9 steps, width 512 (Craftax_Baselines ppo.py:670-707). The README
  reports no results or timings.
- The `--activation` flag is parsed but never passed to a model: the symbolic
  actor-critic uses its class default tanh and the pixel CNN and RNN hard-code relu
  (models/actor_critic.py:132). PPO-RNN is Dense(512) -> GRU(512) -> 2 x Dense(512) heads
  (ppo_rnn.py:40-119).
- Optimistic resets: `num_envs // reset_ratio` resets per step (ratio 16, so 64 for 1024
  envs); finished envs are matched to distinct reset slots at random, and any finished env
  left over shares a slot with another, so two envs can start the same world
  (wrappers.py:83-157). The latest commit adds a deterministic fallback when fewer envs
  finished than there are slots; TrXL's copy predates it (transformerXL_PPO_JAX
  wrappers.py:124-133).
- Achievements: logged per finished episode as percentages, plus their sum as the
  expected number of achievements per episode (logz/batch_logging.py:17-23).
- Transformer-XL PPO (Hamon 2024), the best published Craftax result before Zheng et al.:
  1024 envs x 128 steps, gamma 0.999, gae_lambda 0.8, ent 0.002, otherwise the baseline
  PPO settings; 2 layers, embedding 256, 8 heads, GTrXL gating (bias 2), relative
  positions, memory window 128, gradient window 64 (train_PPO_trXL_pmap.py:11-37). Memory
  carries across rollouts and is masked, not zeroed, at episode boundaries. README:
  18.3% at 1e9 steps, 20.6% at 4e9 (the figure Zheng et al. compare against), no
  significant gain at 8e9, about 6.5 h per 1e9 steps on one A100 (README.md:11, 40, 53, 60).
  Multi-GPU is single-host `pmap`.

Reset cost: gymnax-style auto-reset pays for `reset_env` on every step. Craftax
generates a whole world per reset, which is why purejaxql uses an
`OptimisticResetVecEnvWrapper`: only `num_envs // reset_ratio` resets are computed per
step and handed to the envs that finished (craftax_wrappers.py:83-148; ratio 16 in
pqn_craftax.yaml:25-26). Its docstring notes the trade: a higher ratio is faster but
more likely to hand two envs the same reset.

## 4. Termination vs truncation

None of PureJaxRL, rejax, purejaxql, gymnax, Craftax or Kinetix separates them: the time
limit is ORed into `done` (gymnax cartpole.py:124-125; Craftax game_logic.py:4-9; Kinetix
env.py:141), and GAE stops bootstrapping wherever `done` is set (purejaxrl ppo.py:150-153).
Where episodes end by time limit, the critic learns that the last steps are worth zero.

Correct handling seen elsewhere:
- Brax PPO masks the truncated step: `deltas *= 1 - truncation` and the GAE accumulator is
  cut there (brax/training/agents/ppo/losses.py:68-99). The truncated transition drops
  out of the loss.
- rl_games adds `gamma * V * time_outs` to the reward when `value_bootstrap` is on
  (sapg/rl_games/common/a2c_common.py:880-881).
- FastTD3 and holosoma's FastSAC bootstrap through time-outs with
  `bootstrap = truncations | ~dones` and swap in the pre-reset observation (FastTD3
  train.py:420-425, :607-616; holosoma base_task.py:503-558).
- rsl_rl (Isaac Lab, mjlab) adds gamma x V(s_t) on time-outs, where s_t is the state
  before the last step: an approximation, because the env hands back the post-reset
  observation (rsl_rl ppo.py:153-158). Isaac Lab's opt-in `compute_final_obs` provides
  the true one, but the rsl_rl wrapper does not use it (`pytorch-gpu-sim.md` section 2).
- Craftax_Baselines and Transformer-XL PPO treat Craftax's 100k-step limit as
  termination like the rest; with episodes that long it rarely matters.

`scripts/mprl_common.py` wraps any gymnax-style env, reports `terminated` and
`truncated` separately, returns the pre-reset `final_obs`, and the scaffolds fold
`gamma * V(final_obs)` into the reward on truncated steps. Never bootstrap from the obs
returned on a done step: after auto-reset it belongs to the next episode.

Measured with `scripts/ppo_jax.py` on gymnax Pendulum-v1 (every episode ends by the
200-step limit), 3 seeds, 3M steps each, CPU, 2026-10-27:

| Handling | Train return at 50% | Final deterministic eval per seed |
| --- | --- | --- |
| truncation bootstrapped (default) | -157, -160, -145 | -163, -158, -159 |
| `--truncation-as-terminal` | -159, -183, -200 | -165, -271, -178 |

Command: `python scripts/ppo_jax.py --env Pendulum-v1 --total-timesteps 3000000
--num-envs 64 --num-steps 64 --update-epochs 10 --minibatch-size 512 --lr 1e-3
--gamma 0.95 --ent-coef 0.0 --norm-obs --reward-scale 0.1 --num-seeds 3`. The effect grows
with gamma (a longer horizon puts more value past the cut) and shrinks when
episodes rarely reach the limit.

## 5. Recurrent policies

- Store the done flag from before the step (`last_done`) and reset the hidden state
  where it is set, with `jnp.where` inside an `nn.scan`-wrapped cell (purejaxrl
  ppo_rnn.py:17-36, :169-171).
- Shuffle only the env axis so each sequence stays intact (ppo_rnn.py:258-262). Kinetix's
  `full_minibatch_shuffle` asserts the model is not recurrent (Kinetix ppo.py:430-451).
- Save the rollout's initial hidden state and replay from it during the update
  (ppo_rnn.py:175, :291).
- PQN's RNN variant recomputes lambda-returns inside each minibatch from stop-gradient
  Q-values of the fresh forward pass (purejaxql pqn_rnn_craftax.py:334-382).

## 6. Normalization

- Observation: running mean and variance over the batch axis, starting count 1e-4,
  output `(obs - mean) / sqrt(var + 1e-8)` (purejaxrl wrappers.py:226-292). It must sit
  outside the vmap because it reduces over the env axis. Store the normalized
  observation in the rollout, so the update sees exactly what the policy saw.
- Reward: divide by the running std of the discounted return, no mean subtraction
  (wrappers.py:304-348).
- Network: PQN puts LayerNorm (or BatchNorm) after every hidden layer, never on the
  output head (purejaxql pqn_atari.py:39-88). Input normalization there is BatchNorm or
  BatchRenorm. scaling-crl's residual block is 4 x (Dense, LayerNorm, swish) plus an
  identity skip (train.py:110-125).

## 7. Multi-device

- Kinetix: `jax.jit(jax.shard_map(_update_step, mesh, ...))`; train state replicated, env
  state, obs, done, hidden state and rng sharded on a `"devices"` axis; gradients and loss
  averaged with `jax.lax.pmean` (ppo.py:605-627; train_utils.py:126). `num_train_envs` is
  the total, split across devices (README.md:332).
- JaxGCRL's PPO/SAC/TD3 use `jax.pmap` and assert `num_envs % device_count == 0`
  (ppo.py:182, :362).

## 8. Logging and checkpointing inside jit

- Metrics: return per-step `info` stacked by the scan, or call `jax.debug.callback` with
  a function that prints or calls `wandb.log` (purejaxrl ppo.py:255-261; Kinetix
  ppo.py:584-591). Average episode metrics only over finished episodes:
  `(x * returned_episode).sum() / returned_episode.sum()` (Kinetix ppo.py:489-492);
  report "no episode finished" rather than 0 when the sum is 0.
- Checkpoints: Kinetix runs an outer scan over checkpoint indices and saves through
  `jax.debug.callback` on process 0 (ppo.py:656-686). Formats seen: Orbax (Playground
  notebooks), safetensors (purejaxql), pickle (JaxGCRL, scaling-crl).

## 9. Performance and precision settings

| Setting | Where seen | Why |
| --- | --- | --- |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | Playground train_jax_ppo.py:55-59 | let several processes share a GPU |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=.95` | JaxGCRL scripts/train.sh:15 | give JAX nearly all GPU memory |
| persistent compile cache (`jax_compilation_cache_dir`) | Playground notebooks; Kinetix README `JAX_COMPILATION_CACHE_DIR` | Kinetix and Craftax compiles take tens of seconds or more |
| `JAX_DEFAULT_MATMUL_PRECISION=highest` | Playground README.md:93-95 | Ampere GPUs default to TF32 matmuls, which hurt reproducibility |
| `precision=jax.lax.Precision.HIGHEST` on InfoNCE logits | JaxGCRL losses.py:35-44 | default GPU matmul gave about 0.4% relative error there |
| `--xla_gpu_triton_gemm_any=True` | Playground train_jax_ppo.py:55-59 | GEMM backend choice |

`donate_argnums`, `remat` and `shard_map` were not used in the PPO or PQN files of the
repos studied (only purejaxrl's DQN donates its buffer, dqn.py:77).

Measure environment throughput before choosing N: `scripts/throughput_bench.py` steps N
envs with random actions in a jitted scan and reports where steps/s stops growing.

## 10. Shipped defaults

| Source | N envs | T steps | epochs | minibatches | lr | gamma / lambda | other |
| --- | --- | --- | --- | --- | --- | --- | --- |
| purejaxrl PPO CartPole (ppo.py:277-294) | 4 | 128 | 4 | 4 | 2.5e-4 annealed | 0.99 / 0.95 | clip 0.2, ent 0.01, 2x64 tanh |
| purejaxrl PPO MinAtar (brax_minatar notebook :331-347) | 64 | 128 | 4 | 8 | 5e-3 annealed | 0.99 / 0.95 | relu |
| purejaxrl PPO Brax hopper (ppo_continuous_action.py:292-309) | 2048 | 10 | 4 | 32 | 3e-4 | 0.99 / 0.95 | ent 0, obs and reward norm, 2x256 tanh |
| rejax PPO defaults (mixins.py:102-104; ppo.py:38-64) | 64 | 64 | 8 | 16 | 3e-4 | 0.99 / 0.95 | swish (64,64); `max_grad_norm=inf` and `optax.clip`, not global-norm clipping |
| Kinetix PPO (ppo-base.yaml) | 2048 | 64 (256 for SFL) | 8 | 32 | 5e-5 constant | 0.995 / 0.9 | grad norm 1.0, 5x128 tanh |
| Craftax_Baselines PPO / PPO-RNN (ppo.py:670-707) | 1024 | 64 | 4 | 8 | 2e-4 annealed | 0.99 / 0.8 | ent 0.01, grad norm 1.0, width 512, optimistic resets 16 |
| Transformer-XL PPO, Craftax (train_PPO_trXL_pmap.py:11-37) | 1024 | 128 | 4 | 8 | 2e-4 annealed | 0.999 / 0.8 | ent 0.002, 2-layer TrXL, memory 128 |
| rsl_rl PPO, Isaac Lab / mjlab velocity (`pytorch-gpu-sim.md`) | 4096 | 24 | 5 | 4 | 1e-3 adaptive-KL | 0.99 / 0.95 | desired KL 0.01, [512,256,128] ELU |
| purejaxql PQN MinAtar (pqn_minatar.yaml) | 128 | 32 | 2 | 32 | 5e-4 decayed | 0.99 / 0.65 | RAdam, grad norm 10, eps 1 -> 0.05 over 10% |
| purejaxql PQN Craftax RNN (pqn_rnn_craftax.yaml) | 1024 | 128 | 4 | 4 | 3e-4 | 0.99 / 0.5 | LSTM 512, input norm, optimistic resets 16 |
| Playground Brax PPO locomotion base (locomotion_params.py:30-53) | 8192 | 20 | 4 | 32 | 3e-4 | 0.97 | batch 256, ent 1e-2, obs norm; G1/T1 use 200M steps, ent 0.005 |
| Playground Brax SAC DM Control (dm_control_suite_params.py:113-137) | 128 | n/a | 8 grad steps per env step | batch 512 | 1e-3 | 0.99 | buffer 4.2M, LayerNorm on Q |
| JaxGCRL CRL (crl.py:133-163) | 256 | 62 | n/a | batch 256 | 3e-4 | 0.99 | replay 10k per env, on-device buffer |
| scaling-crl (train.py:44-79) | 512 | 62 | n/a | batch 256 | 3e-4 | 0.99 | depth 4 to 1024 via `--critic_depth`, 800 SGD batches per step |
| Zheng et al. A2C Craftax (paper App. A.2) | from batch law | 16 | 1 | 4 (V-trace for staleness) | from lr law, decayed to 0 | 0.999 | ent 0.001, Adam (0.9, 0.99), grad norm 0.5 |

## 11. Version traps

- `jax.tree_map` is gone in current JAX (`hasattr(jax, "tree_map")` is False on 0.12);
  2022-era code that calls it fails. The gymnax GitHub clone studied (last commit Dec 2022)
  calls it; the PyPI release 0.0.9 ran fine on JAX 0.12 here.
- purejaxql pins `jax>=0.4.16,<=0.4.38` (pyproject.toml:27-28).
- scaling-crl pins jax 0.4.23 and brax 0.10.1 and requires two hand patches to brax
  (`mjx.ncon` removed after MuJoCo 3.1.5; an `rgba` comparison in brax/io/json.py)
  (README.md:28-88). Humanoid runs with deep nets may need up to 80 GB of GPU memory
  (README.md:92).
- FlashSAC's README warns MuJoCo Playground has issues with JAX newer than 0.5.2
  (README.md:90-91).
- Craftax's PPO baselines live in a separate Craftax_Baselines repository, not in the
  Craftax package (Craftax README.md:96); its requirements are unpinned except
  orbax-checkpoint 0.5.0. transformerXL_PPO_JAX pins jax 0.4.26, flax 0.8.2, craftax
  1.4.3, gymnax 0.0.7.

Commits read: purejaxrl, rejax, gymnax 39b3ba0, Craftax 1.7.0, Kinetix, purejaxql 47af6d7,
JaxGCRL a2740ce, scaling-crl 17acb51, mujoco_playground 12ad7f3, brax c860529 (all
shallow clones of the default branch on 2026-10-27).
