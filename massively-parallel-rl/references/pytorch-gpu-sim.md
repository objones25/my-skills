# PyTorch with GPU simulators, and fast off-policy learners

The second stack: a GPU physics simulator hands batched tensors to a PyTorch learner
without leaving the device. Repo facts carry file:line references into the clones listed
at the end; product facts link the page they came from.

## Contents
1. Simulators and hardware
2. Isaac Lab and rsl_rl: the default robotics PPO
3. FastTD3: the recipe that made off-policy fast
4. FastSAC (holosoma): humanoids in minutes
5. FlashSAC: scaling the network
6. PufferLib 5.0: native CUDA PPO
7. Rules for keeping the loop on the GPU
8. Off-policy scaling results to apply

## 1. Simulators and hardware

| Simulator | Runs on | Notes |
| --- | --- | --- |
| Isaac Lab 3.0.0 | Ubuntu 22.04+ (x86_64, aarch64) or Windows 11; NVIDIA GPU, Turing or newer (CC 7.5+) for the default CUDA 13 stack; driver 595.58.03 on Linux x86_64 | Python 3.12; Newton is the default physics engine (MuJoCo Warp solver by default for velocity tasks), Isaac Sim 6.2 optional; pins torch 2.11.0 and rsl-rl-lib 5.5.1 (IsaacLab pyproject.toml:19-87; docs/source/isaaclab/installation.rst:83-118; [install docs](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)). Headless Newton needs no RTX GPU. macOS is not listed. |
| mjlab 1.6.0 | training: Linux + NVIDIA GPU (Turing+, driver 580+); macOS evaluation only, CPU | Isaac Lab's manager-based API on MuJoCo Warp 3.16, trains with rsl_rl (mjlab README.md:13-19; docs/source/installation.rst:17-20; faq.rst:13-22) |
| [MuJoCo Warp (MJWarp)](https://mujoco.readthedocs.io/en/latest/mjwarp/) | tuned for NVIDIA GPUs; CPU mode for determinism | docs say it scales better than MJX on nearly every workload but is not differentiable; MJX exposes part of it with `impl='warp'` |
| MJX / MuJoCo Playground | JAX accelerators | see `jax-stack.md`; `wrap_for_torch_training` exposes `info["truncation"]` as `time_outs` (Playground _src/wrapper_torch.py:176-179) |
| Isaac Gym (legacy) | NVIDIA GPU, Linux | [Makoviychuk et al. 2021](https://arxiv.org/abs/2108.10470): physics and policy both on GPU, 2-3 orders of magnitude faster than a CPU simulator with a GPU learner. SAPG's code still uses it; holosoma dropped it after v0.0.3. |

On a Mac: PyTorch's MPS backend runs networks on the Apple GPU, but none of the GPU
simulators above train there. Apple's [jax-metal](https://developer.apple.com/metal/jax/)
plugin is labelled experimental, does not pass all JAX tests, lacks float64, and its
compatibility table stops at jaxlib 0.4.26. Treat a Mac as a place to write and test
code at small N (gymnax and Craftax run on JAX's CPU backend; mjlab can evaluate on CPU)
and rent an NVIDIA GPU for anything that is measured. `scripts/check_env.py` reports
what the current machine can do.

## 2. Isaac Lab and rsl_rl: the default robotics PPO

rsl_rl (ETH Zurich and NVIDIA, BSD-3) is the PPO library Isaac Lab, mjlab, Legged Gym
and MuJoCo Playground list as users (rsl_rl README.md:24-27).

**Velocity-task defaults**, shared by Isaac Lab's Anymal-D, Go2, H1 and G1 rough configs
(core/velocity/config/<robot>/agents/rsl_rl_ppo_cfg.py) and mjlab's G1/Go1 velocity and G1
tracking configs (mjlab tasks/*/config/*/rl_cfg.py:10-46):

| Setting | Value |
| --- | --- |
| num_envs | 4096 (Isaac Lab velocity_env_cfg.py:341; mjlab passes `--env.scene.num-envs 4096`) |
| steps per env per update | 24 |
| batch per update | 4096 x 24 = 98,304 |
| epochs / minibatches | 5 / 4, so minibatch 24,576 |
| learning rate | 1e-3 with the adaptive-KL schedule, desired_kl 0.01 |
| gamma / lambda / clip | 0.99 / 0.95 / 0.2, clipped value loss, value coef 1.0, grad norm 1.0 |
| entropy | 0.005 (Anymal-D), 0.01 (Go2, H1), 0.008 (G1); mjlab 0.01 velocity, 0.005 tracking |
| networks | [512, 256, 128] ELU for rough terrain, smaller for flat |
| obs normalization | off in Isaac Lab, on in mjlab |
| iterations | Isaac Lab rough: 2000 / 1500 / 2000 / 5000; mjlab: 30k G1 velocity, 10k Go1, 30k G1 tracking |
| control rate | sim dt 0.005 with decimation 4 = 50 Hz; 20 s episodes = 1000 steps |

Mechanisms worth knowing (rsl_rl/algorithms/ppo.py):
- **Adaptive-KL learning rate** (:240-266), checked every minibatch: KL > 2 x desired_kl
  divides lr by 1.5 (floor 1e-5); KL < desired_kl / 2 multiplies it by 1.5 (cap 1e-2).
  With multiple GPUs the KL is averaged across ranks and rank 0 broadcasts the new lr.
  This is a trust-region-by-learning-rate, the outer step-size lever Beukman et al.
  discuss, applied automatically.
- **Time-out bootstrap is approximate** (:153-158). It adds gamma x V(s_t) to the reward
  on time-outs, where s_t is the state before the last step, because the env returns
  the post-reset observation. Then GAE treats the step as done.
- Isaac Lab now has an opt-in `compute_final_obs` flag that puts the pre-reset
  observation in `extras["final_obs"]` (manager_based_rl_env_cfg.py:47-61;
  manager_based_rl_env.py:248-249). The SB3 and TorchRL wrappers read it; the rsl_rl
  wrapper does not (isaaclab_rl/rsl_rl/vecenv_wrapper.py:155-168). mjlab's
  `auto_reset=False` returns the true terminal observation for custom loops
  (mjlab envs/manager_based_rl_env.py:144-155).
- Episode lengths are randomized once at the start of training
  (`init_at_random_ep_len`, on_policy_runner.py:58-62) so resets are spread out; not on
  every reset.
- Feedforward minibatches reuse one permutation across all epochs
  (rollout_storage.py:229-248); recurrent minibatches split by environment with padded
  trajectories.
- `EmpiricalNormalization` updates once per iteration, after the gradient steps, and
  all-reduces across GPUs (ppo.py:327-332; modules/normalization.py:62-71).
- Multi-GPU is data-parallel over NCCL with bucketed gradient all-reduce; each rank runs
  its own envs, so experience per iteration multiplies by the GPU count and
  `max_iterations` does not shrink on its own (mjlab docs/source/training/distributed_training.rst:41-58).
- Also included: LSTM/GRU policies, symmetry augmentation (data or mirror loss; not with
  recurrent policies), RND curiosity, distillation.

Batch-size note: 4 minibatches of 24,576 is a large minibatch by the standard of
Beukman et al. (who found 16k better than 96k on Isaac Gym tasks). If a user scales
num_envs past 4096 with `num_mini_batches=4` fixed, the minibatch grows with it; the
`large-batch-on-policy.md` recipe applies.

## 3. FastTD3 ([paper](https://arxiv.org/abs/2505.22642), repo MIT)

TD3 (twin critics, delayed actor updates, target policy smoothing) made fast by parallel
envs and very large batches.

Loop (fast_td3/train.py:578-676): step all N envs once, write the transition into a GPU
replay buffer, then run `num_updates` critic updates, an actor update every
`policy_frequency`, and a fused foreach soft target update
(`_foreach_mul_`, `_foreach_add_`, train.py:527-533).

- Replay buffer: preallocated device tensors `(n_env, buffer_size, dim)`, so
  `buffer_size` is per env; sampling draws `batch_size // num_envs` per env
  (fast_td3_utils.py:49-155). README: keep the whole buffer on the GPU; for OOM shrink
  buffer_size, then batch_size, then num_envs (README.md:270).
- Truncation: `bootstrap = (truncations | ~dones)`; the pre-reset obs replaces the reset
  obs where the env provides it (train.py:420-425, :607-616). FastTD3's Isaac Lab wrapper
  says Isaac Lab cannot return it (isaaclab_env.py:71-77) and disables bootstrapping on
  Lift-Cube; that predates Isaac Lab's `compute_final_obs` flag (section 2).
- Observations stored raw and normalized at sample time with the current running
  statistics (train.py:620, :646-649).
- Critic: distributional (C51-style, 101 atoms by default, 251 for Isaac Lab), clipped
  double Q picks the critic with the lower mean (train.py:449-470).

| Setting | HumanoidBench base | MuJoCo Playground | Isaac Lab |
| --- | --- | --- | --- |
| num_envs | 128 | 1024 | 4096 |
| batch_size | 32768 | 32768 | 32768 |
| buffer per env | 51,200 | 10,240 | 10,240 (5,120 rough terrain) |
| num_updates per env step | 2 | 2 (8 for Go1) | 2 (4 velocity, 8 Franka) |
| gamma | 0.99 | 0.97 | 0.99 |
| v_min / v_max | ±250 (per task in HumanoidBench) | ±10 | ±10 |

Other defaults (hyperparams.py): tau 0.1, AdamW lr 3e-4 with weight decay 0.1, critic
width 1024 and actor 512 (each halved twice per layer), exploration noise std per env
drawn from [0.001, 0.4], bf16 autocast, `torch.compile(mode="reduce-overhead")` on the
update functions with `torch.compiler.cudagraph_mark_step_begin()` once per iteration
(fast_td3_utils.py:810-813). The README reports all experiments on one A100 80 GB;
larger num_envs, num_updates or batch improve sample efficiency but cost wall-clock
(README.md:263-268). If learning stalls early, raise num_updates, then try n-step 3 or
turn off clipped double Q.

The paper (2505.22642 v3) reports solving a range of HumanoidBench tasks in under 3
hours on one A100, and that rewards tuned for PPO did not transfer to FastTD3 or the
reverse.

**Simplicial embeddings** ([Obando-Ceron et al., ICLR 2026](https://arxiv.org/abs/2510.13704))
are now in upstream FastTD3 (commit 229ed59, PR #35): `SimNormLinear` = Linear -> LayerNorm
-> softmax within each of L = 8 groups of V = 64 units, no temperature
(fast_td3/fast_td3.py:20-55). It replaces the critic's third hidden layer and/or the
actor head; enable with `--sim_type {sim_actor,sim_critic,sim_both}`, `--sim_dimension 64`
(hyperparams.py:128-135). Use upstream: the paper's linked fork
(waltermayor/FastTD3_SEm, 749a20b) has the layer but its hyperparams.py lacks the four
`sim_*` flags its own train.py reads, and its multi-GPU script does not pass them.

## 4. FastSAC (holosoma, Apache-2.0)

The open-source code for [Seo et al. 2025](https://arxiv.org/abs/2512.01996) is
[amazon-far/holosoma](https://github.com/amazon-far/holosoma) (952e418), which ships PPO and
FastSAC, not FastTD3 (README.md:8). Simulators: Isaac Sim and MuJoCo Warp (beta) for
training, MuJoCo for inference; 200 Hz physics with decimation 4. Robots: Unitree G1 and
Booster T1, 29 DoF. The repo itself makes no "15 minutes on an RTX 4090" claim; that is
the paper's.

| Setting (G1 locomotion preset, config_values/algo.py:77-117) | Value |
| --- | --- |
| num_envs | 4096 |
| global batch | 8192 (per env per update: batch // num_envs // world_size) |
| replay buffer | 1024 transitions per env, on the GPU |
| updates per env step / actor every | 8 / 4th (2 actor updates per 8 critic) |
| gamma / tau / n-step | 0.97 / 0.125 / 1 |
| optimizer | AdamW, lr 3e-4 for actor, critic and alpha; weight decay 1e-3; betas (0.9, 0.95); fused |
| entropy | alpha starts at 1e-3, autotuned; target entropy 0 for locomotion (-0.5 x n_act for whole-body tracking); log std in [-5, 0], so max std 1 |
| critic | 2 distributional Q nets, 101 atoms over ±20 (501 for tracking); LayerNorm + SiLU in every layer; the actor uses the **mean** of the two Qs; no clipped double Q anywhere |
| widths | critic 768, actor 512, each tapering by halves |
| other | obs normalization (stats updated on replay samples), symmetry augmentation doubling the batch, bf16 AMP, torch.compile, no grad clipping |

- Action bounds come from joint limits: per joint, the larger distance from the default
  pose to a limit, divided by the action scale (0.25), sets the tanh scale
  (agents/fast_sac/fast_sac_agent.py:116-154).
- Only timed-out envs store a true final observation, computed before reset
  (envs/base_task/base_task.py:503-558); the target uses `bootstrap = truncations | ~dones`
  (fast_sac_agent.py:439).
- The alpha loss uses next-state log-probs and steps with every critic update
  (fast_sac_agent.py:477-493).
- G1 locomotion reward has 10 terms (config_values/loco/g1/reward.py:99-191).
- CUDA graphs ("reduce-overhead") only for multi-GPU; single GPU uses default compile.
  The distributional projection is excluded from compilation because the fused kernel
  gave wrong projections in multi-GPU runs (fast_sac.py:213-229).
- Multi-GPU: `torchrun --nproc_per_node=4 ... --training.num-envs 16384`, num_envs and
  batch divided by world size, one flattened gradient all-reduce (src/holosoma/README.md:216-221).
- Whole-body tracking preset: gamma 0.99, 4 updates per step, actor every 2nd, tau 0.05,
  400k iterations, G1 on Isaac Sim only.
- Domain randomization (G1 locomotion): pushes every 5-10 s, 0-1 step action delay, PD
  gains x0.9-1.1, link mass x0.9-1.2, friction 0.5-1.25, COM ±5 cm
  (config_values/loco/g1/randomization.py).

## 5. FlashSAC ([paper](https://arxiv.org/abs/2604.04539), repo MIT)

SAC made stable at about 10x the usual network size.

- Blocks: inverted bottleneck residual (Linear h to 4h, BatchNorm, ReLU, Linear 4h to h,
  BatchNorm, ReLU, add) (flash_rl/agents/flashSAC/layer.py:91-108). Input BatchNorm stands
  in for an observation normalizer (:79-88).
- After every optimizer step, weight rows are projected to unit L2 norm and norm scales
  to sqrt(d) (layer.py:10-76; update.py:153-155).
- Critic: two critics in one ensemble module, 101-bin categorical value, min-Q selection
  per sample (update.py:14-78). Truncation bootstraps; only `terminated` cuts it
  (update.py:229-237).
- Entropy target `0.5 * act_dim * log(2*pi*e*0.15^2)`, the entropy of a Gaussian with std
  0.15 per dimension (agent.py:355).
- Exploration noise is held for n steps, n from a truncated zeta distribution (mu 2,
  max 16) (agent.py:200-257).
- GPU-simulator runs: 1024 envs, 10M-transition buffer on the GPU, batch 2048, 2 updates
  per env step, fp16 AMP with GradScaler (scripts/run_isaaclab.sh:31-48). CPU simulators:
  1 env, CPU buffer, batch 512, no AMP. README: AMP adds a host sync that hurts small
  batches (README.md:160-162); on PyTorch 2.8+ `reduce-overhead` caused 5-10x slowdowns
  for them, so `compile_mode: auto` picks max-autotune on torch 2.9+ (README.md:151-154;
  agent.py:326-333).
- Paper: sim-to-real on a Unitree G1 walked in about 20 minutes of training vs about 3
  hours for PPO on flat ground.
- Caveat for copying its loop: it moves env I/O to NumPy every step and calls `.item()`
  on every metric every update (isaaclab.py:171-177; agent.py:507-513). Those syncs are
  avoidable (section 7).

FastSAC vs FlashSAC on the same question, so the choice is explicit: FastSAC averages
its two critics and keeps small LayerNorm MLPs (768/512); FlashSAC takes the per-sample
minimum and uses 2.5M-parameter residual blocks with weight projection.

## 6. PufferLib 5.0 (repo MIT)

The current branch is a native C/CUDA trainer, not a Python package: environments are C
or CUDA compiled into the binary (`./build.sh ENV [--cu]`, build.sh:4-23).

- CPU envs run under OpenMP inside per-buffer threads with pinned host arrays and async
  copies; GPU envs capture the whole horizon (network and env) in one CUDA graph; the
  whole training epoch is also graph-captured (src/pufferl.cu:882-1289, :1606-1626).
- Defaults (config/default.ini): 4096 agents, horizon 64, minibatch 8192 steps,
  `replay_ratio` 1.0 in place of epochs, lr 0.015 cosine-annealed, gamma 0.995,
  gae_lambda 0.90, clip 0.2, vf_coef 2.0, ent 0.001, grad norm 1.5.
- Muon optimizer (Nesterov momentum 0.95, 5 Newton-Schulz steps, no weight decay) in
  place of Adam; a code comment calls it "a major upgrade over Adam" (src/algo.cu:1122-1229).
- GAE with optional V-trace clipping, recomputed per minibatch with current values
  (algo.cu:1687-1765). An async mode trains on one buffer while collecting into the other
  with a one-epoch-stale actor.
- No throughput figure in the README; the code prints live steps/s.

## 7. Rules for keeping the loop on the GPU

1. Keep obs, actions, rewards, dones and the replay buffer as device tensors end to end.
   One `.cpu()`, `.numpy()` or `.item()` per step forces the CPU to wait for the GPU.
2. Log from device tensors every K steps (FastTD3 reads means every 100 steps,
   train.py:678-691).
3. Make schedule values (learning rate, entropy coefficient) device tensors when using
   CUDA graphs: a Python float is baked into the captured graph (FastTD3 passes
   `lr=torch.tensor(..., device=)`, train.py:291-312; PufferLib keeps them in device
   memory).
4. With `torch.compile(mode="reduce-overhead")` (CUDA graphs) call
   `torch.compiler.cudagraph_mark_step_begin()` once per iteration before any compiled
   call; holosoma also clones graph outputs before reuse (fast_sac_agent.py:50-56).
5. Mixed precision: bf16 autocast needs no GradScaler; fp16 does.
6. Sample replay indices on the device (`torch.randint(..., device=)`) and gather.
7. Cap CPU threads in the learner process (FastTD3 sets `OMP_NUM_THREADS=1`).

## 8. Off-policy scaling results to apply

- Batch size depends on model size: small critics overfit large batches because their
  bootstrap targets are poor ("TD-overfitting"); the best batch grows with model size
  and shrinks as the update-to-data ratio (UTD) rises ([Fu et al. 2025](https://arxiv.org/abs/2508.14881)).
  Predicted batch sizes had about 49% error, but performance was flat near the optimum.
- Data and compute trade off through UTD along a predictable frontier
  ([Rybkin et al. 2025](https://proceedings.mlr.press/v267/rybkin25a.html)).
- Bigger networks need normalization to stay stable: LayerNorm everywhere (FastSAC),
  weight projection plus BatchNorm (FlashSAC), SimbaV2-style hyperspherical layers
  (FastTD3 `--agent fasttd3_simbav2`), or simplicial embeddings (section 3).

Commits read (2026-10-27): FastTD3 229ed59, FastTD3_SEm 749a20b, holosoma 952e418,
FlashSAC 87edc90, PufferLib ea7f989 (branch 5.0), Isaac Lab d67292f, rsl_rl f26f3e5,
mjlab 8bb2bc9, SAPG (rl_games) default branch.
