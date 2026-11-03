# Simulators and RL tooling (state as of 2026-11-02)

What each tool is, its latest version, backend, install route, license, and when to pick it. The case-study stack is mjlab 1.6.0 + rsl-rl-lib 5.4.2, task `Mjlab-Velocity-Flat-Unitree-G1`, one A100; those two get full config detail below.

How this was checked (all on 2026-11-02):
- Versions and upload dates: PyPI JSON API, `https://pypi.org/pypi/<name>/json` (same data as `https://pypi.org/project/<name>/`). Dates are the upload time of the first file of that version, UTC.
- Release notes: GitHub release pages opened with WebFetch. Several GitHub release pages showed an older "Latest" than PyPI (MuJoCo Warp, Gymnasium); where they disagree, PyPI is the newer and is used for "latest".
- mjlab and rsl_rl source: `git clone --branch v1.6.0` of mujocolab/mjlab (commit 0fb8a68, 2026-08-08) and `--branch v5.4.2` of leggedrobotics/rsl_rl (commit aabd3eb, 2026-07-15). Source URLs below point at those tags.
- Docs: Context7 MCP (mjlab docs, Gymnasium docs, SB3 source, MuJoCo Warp README) and direct fetches.
- Lines marked [derived] are arithmetic I did on verified numbers. Lines marked "Not verified" say what could not be checked.

Terms: PPO (proximal policy optimization) is an on-policy actor-critic algorithm that limits each update with a clipped probability ratio. GAE (generalized advantage estimation) blends n-step returns with weight λ (`lam`). An "iteration" in rsl_rl = collect `num_steps_per_env` steps from every parallel env, then run `num_learning_epochs` passes over that batch split into `num_mini_batches` minibatches. Decimation = number of physics steps per policy step.

---

## Quick pick table

| Tool | Latest seen | Upload date | Backend | License | Pick it when |
|---|---|---|---|---|---|
| mjlab | 1.6.0 | 2026-08-09 | MuJoCo Warp, NVIDIA GPU, PyTorch | Apache-2.0 | Isaac Lab-style manager API on MuJoCo physics; used in the G1 case study |
| MuJoCo Warp (`mujoco-warp`) | 3.17.0 | 2026-10-22 | NVIDIA Warp on NVIDIA GPU (CPU for debugging) | Apache-2.0 | raw batched MuJoCo on GPU without a framework |
| MJX (`mujoco-mjx`) | 3.17.0 | 2026-10-22 | JAX (GPU, TPU, Apple Silicon), or Warp via `impl='warp'` | Apache-2.0 | JAX training stacks; differentiable sim (JAX impl only) |
| MuJoCo Playground (`playground`) | 0.2.0 | 2026-03-16 | MJX / MuJoCo Warp, JAX + Brax training | Apache-2.0 (classifier) | ready JAX envs for quadrupeds, humanoids, hands, arms |
| Isaac Lab (`isaaclab`) | 3.0.0 | 2026-10-19 | Isaac Sim / PhysX; Newton (MuJoCo Warp solver) | BSD-3-Clause | NVIDIA stack, rendering, sensors, largest task zoo |
| Isaac Gym | Preview 4 (last) | not dated on page | PhysX GPU | NVIDIA | do not start new work; deprecated |
| rsl_rl (`rsl-rl-lib`) | 5.5.2 | 2026-10-28 | PyTorch | BSD-3-Clause | GPU-batched PPO for locomotion; mjlab pins 5.4.2 |
| Brax | 0.14.2 | 2026-03-15 | JAX | Apache-2.0 | JAX PPO/SAC training code (`brax/training`) |
| ManiSkill3 (`mani-skill`) | 3.0.1 | 2026-04-21 | SAPIEN (PhysX), GPU or CPU | Apache-2.0 (repo LICENSE) | GPU-parallel manipulation with rendering |
| Genesis (`genesis-world`) | 1.4.5 | 2026-10-22 | Quadrants compiler: CUDA, ROCm, Metal, Vulkan, x86, ARM64 | Apache-2.0 | multi-physics (rigid, FEM, MPM, SPH, PBD) or non-NVIDIA GPUs |
| Gymnasium | 1.4.0 | 2026-10-05 | CPU env API | MIT | single-env API standard, CPU baselines |
| Stable-Baselines3 | 2.9.2 | 2026-10-26 | PyTorch, CPU-oriented | MIT | reference PPO on Gymnasium envs, small experiments |
| CleanRL | 1.2.0 on PyPI (stale) | 2023-05-22 | PyTorch / JAX single files | MIT | readable single-file PPO to compare against |

Sources for every row: the PyPI JSON for that name (https://pypi.org/pypi/<name>/json, checked 2026-11-02); licenses as noted in each section below.

---

## mjlab

- What it is: Isaac Lab's manager-based environment API (observation, reward, event, termination, curriculum managers) running on MuJoCo Warp. "Isaac Lab API, powered by MuJoCo-Warp, for RL and robotics research." (source: https://pypi.org/pypi/mjlab/json, checked 2026-11-02; https://github.com/mujocolab/mjlab/blob/v1.6.0/README.md, checked 2026-11-02)
- Latest: 1.6.0, uploaded 2026-08-09. Recent: 1.5.3 (2026-07-22), 1.5.0 (2026-06-28), 1.4.0 (2026-05-26). (source: https://pypi.org/pypi/mjlab/json, checked 2026-11-02)
- 1.6.0 release notes: MuJoCo and MuJoCo Warp bumped to 3.11; visual domain randomization of lights and textures; a `GeomCfg` editor; fixes for partial resets leaking state between envs; breaking changes to `CollisionCfg`, `CommandTerm._update_command`, `ViewerConfig`. (source: https://github.com/mujocolab/mjlab/releases, checked 2026-11-02)
- Pinned dependencies in 1.6.0: `rsl-rl-lib==5.4.2` (exact), `mujoco~=3.11.0`, `mujoco-warp~=3.11.0`, `warp-lang>=1.14.0`, `torch>=2.7.0`; Python `>=3.10,<3.14`. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/pyproject.toml, checked 2026-11-02)
- Backend: training needs Linux, an NVIDIA GPU and CUDA 12.4 or newer (for conditional execution in CUDA graphs); evaluation also runs on macOS or WSL. (source: https://mujocolab.github.io/mjlab/main/source/installation.html via Context7, checked 2026-11-02; https://mujocolab.github.io/mjlab/main/source/faq.html via Context7, checked 2026-11-02)
- GPU guidance: FAQ recommends "RTX 40-series GPUs (or newer)" and "L40s, H100"; A100 is not in that list, but the cloud guide shows `A100:1` and `A100:8` launches. Multi-GPU uses torchrunx with `--gpu-ids`. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/docs/source/faq.rst, checked 2026-11-02; https://github.com/mujocolab/mjlab/blob/v1.6.0/docs/source/training/cloud.rst, checked 2026-11-02)
- Install: `pip install mjlab`; or manage it as a uv dependency (from PyPI, from GitHub, or a local editable install); or clone and `uv run demo`. (source: https://mujocolab.github.io/mjlab/main/source/installation.html via Context7, checked 2026-11-02; https://github.com/mujocolab/mjlab/blob/v1.6.0/README.md, checked 2026-11-02)
- License: Apache-2.0. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/pyproject.toml, checked 2026-11-02)
- Citation metadata: "mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning", Zakka, Yi, Liao, Le Lay, Sreenath, Abbeel. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/CITATION.cff, checked 2026-11-02)
- When to pick: you want Isaac Lab-style task configs, MuJoCo's contact model and MJCF assets, and a small dependency set.

### Training command and the `num_envs` flag

- README: `uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096`. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/README.md, checked 2026-11-02)
- The training docs in the same tag write `--num-envs 4096`. In v1.6.0 `TrainConfig` has fields `env`, `agent`, `gpu_ids`, video and logging options, and no top-level `num_envs`; `num_envs` lives at `env.scene.num_envs` (default 1). So `--env.scene.num-envs` matches the source; whether `--num-envs` is accepted was not run, so it is not verified. Run `uv run train Mjlab-Velocity-Flat-Unitree-G1 --help` to confirm. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/docs/source/training/rsl_rl.rst, checked 2026-11-02; https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/scripts/train.py, checked 2026-11-02; https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/velocity/velocity_env_cfg.py, checked 2026-11-02)
- Any config field can be overridden from the CLI with dot paths and hyphens, e.g. `--agent.algorithm.learning-rate 3e-4`, `--env.decimation 2`; booleans need explicit `True`/`False`. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/docs/source/training/rsl_rl.rst, checked 2026-11-02)
- The default logger is wandb (`logger: "wandb"`, project `mjlab`); set `--agent.logger tensorboard` to avoid it. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/rl/config.py, checked 2026-11-02)
- Training starts with `init_at_random_ep_len=True`, so envs begin at random points in their episodes and resets are spread out. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/scripts/train.py, checked 2026-11-02)

### PPO config for `Mjlab-Velocity-Flat-Unitree-G1`

Both `Mjlab-Velocity-Flat-Unitree-G1` and `Mjlab-Velocity-Rough-Unitree-G1` use the same `unitree_g1_ppo_runner_cfg()` and `VelocityOnPolicyRunner`. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/velocity/config/g1/__init__.py, checked 2026-11-02)

Values below are from https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/velocity/config/g1/rl_cfg.py (checked 2026-11-02), compared with mjlab's dataclass defaults in https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/rl/config.py (checked 2026-11-02) and rsl_rl 5.4.2's own defaults in https://github.com/leggedrobotics/rsl_rl/blob/v5.4.2/rsl_rl/algorithms/ppo.py and https://github.com/leggedrobotics/rsl_rl/blob/v5.4.2/docs/guide/configuration.rst (both checked 2026-11-02).

| Field | G1 velocity task | mjlab dataclass default | rsl_rl 5.4.2 default |
|---|---|---|---|
| num_steps_per_env | 24 | 24 | required (no default) |
| max_iterations | 30,000 | 300 | n/a (runner arg) |
| num_learning_epochs | 5 | 5 | 5 |
| num_mini_batches | 4 | 4 | 4 |
| clip_param | 0.2 | 0.2 | 0.2 |
| gamma | 0.99 | 0.99 | 0.99 |
| lam (GAE λ) | 0.95 | 0.95 | 0.95 |
| learning_rate | 1.0e-3 | 1e-3 | 0.001 |
| schedule | "adaptive" | "adaptive" | "adaptive" |
| desired_kl | 0.01 | 0.01 | 0.01 |
| entropy_coef | 0.01 | 0.005 | 0.01 |
| value_loss_coef | 1.0 | 1.0 | 1.0 |
| use_clipped_value_loss | True | True | True |
| max_grad_norm | 1.0 | 1.0 | 1.0 |
| optimizer | adam (default) | "adam" | "adam" |
| normalize_advantage_per_mini_batch | False (default) | False | False |
| actor hidden dims | (512, 256, 128), ELU | (128, 128, 128), ELU | [256, 256, 256], ELU |
| critic hidden dims | (512, 256, 128), ELU | (128, 128, 128), ELU | [256, 256, 256], ELU |
| obs_normalization | True (actor and critic) | False | False |
| init noise std | GaussianDistribution, init_std 1.0, std_type "scalar" | same | init_std 1.0, std_type "scalar", learn_std True |
| save_interval | 50 | 50 | required |
| seed | 42 (default) | 42 | n/a |

Notes:
- There is no field literally named `init_noise_std` in this rsl_rl version; the initial action standard deviation is `actor.distribution_cfg["init_std"]`. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/rl/config.py, checked 2026-11-02; https://github.com/leggedrobotics/rsl_rl/blob/v5.4.2/docs/guide/configuration.rst, checked 2026-11-02)
- Adaptive schedule in rsl_rl 5.4.2: after each minibatch, if mean KL > 2 × desired_kl then lr = max(1e-5, lr / 1.5); if 0 < KL < desired_kl / 2 then lr = min(1e-2, lr × 1.5). (source: https://github.com/leggedrobotics/rsl_rl/blob/v5.4.2/rsl_rl/algorithms/ppo.py, checked 2026-11-02)
- Time-out bootstrapping: when the env passes `extras["time_outs"]`, rsl_rl adds γ × V(s) to the reward of timed-out transitions, so a time limit is not treated as a failure. mjlab's wrapper sets `extras["time_outs"] = truncated`. (source: https://github.com/leggedrobotics/rsl_rl/blob/v5.4.2/rsl_rl/algorithms/ppo.py, checked 2026-11-02; https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/rl/vecenv_wrapper.py, checked 2026-11-02)
- Batch sizes at 4096 envs [derived]: 4096 × 24 = 98,304 transitions per iteration; minibatch 24,576; 5 epochs × 4 minibatches = 20 gradient steps per iteration; 30,000 iterations ≈ 2.95 billion env steps. These match legged_gym's Table 3 values (see `../papers/legged-gym.md`).

### G1 velocity environment (what PPO is trained on)

From https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/velocity/velocity_env_cfg.py (checked 2026-11-02) and https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/velocity/config/g1/env_cfgs.py (checked 2026-11-02) unless noted.

- Timing: physics timestep 0.005 s, decimation 4, so the policy acts every 0.02 s (50 Hz) [derived]; episode length 20 s (1,000 policy steps [derived]). Solver iterations 10, line-search iterations 20.
- Action: joint position targets for all actuators (`JointPositionActionCfg`, `use_default_offset=True`, so the action is an offset from the default pose), scaled per joint by `G1_ACTION_SCALE`.
- Actor observations (with uniform noise): base linear velocity (±0.5), base angular velocity (±0.2), projected gravity (±0.05), joint positions relative to default with encoder bias (±0.01), joint velocities (±1.5), last action, velocity command. Flat task removes `height_scan`.
- Critic observations (no noise, asymmetric critic): the actor terms with unbiased joint positions, plus foot height, foot air time, foot contact and foot contact forces.
- Commands: (vx, vy, ωz) resampled every 3 to 8 s; initial ranges vx, vy ∈ [−1, 1] m/s, ωz ∈ [−0.5, 0.5] rad/s; 10% standing envs, 30% heading-command envs. A command curriculum widens vx to [−1.5, 2.0] and ωz to [−0.7, 0.7] at env step 5000 × 24 (iteration 5,000), then vx to [−2.0, 3.0] at iteration 10,000. The flat task keeps this curriculum and drops only the terrain curriculum.
- Domain randomization ("events"): foot friction U(0.3, 1.2) at startup, shared across foot geoms; encoder bias U(−0.015, 0.015) rad; torso COM offset ±0.025 m (x, y) and ±0.03 m (z); pushes every 1 to 3 s that set base velocity within ±0.5 m/s (x, y), ±0.4 (z), ±0.52 rad/s (roll, pitch), ±0.78 rad/s (yaw); reset pose x, y ±0.5 m, yaw ±3.14. No link-mass or PD-gain randomization in this config.
- Rewards (weights): linear velocity tracking 2.0 (std √0.25), angular velocity tracking 2.0 (std √0.5), upright 1.0, variable posture 1.0 (per-joint std for standing, walking, running), joint position limits −1.0, action rate −0.1, foot clearance −2.0 (target 0.1 m), foot swing height −0.25, foot slip −0.1, soft landing −1e−5; G1 overrides: body angular velocity −0.05, angular momentum −0.02, air time 0.0, self-collisions −1.0.
- Terminations: time-out (marked as time-out) and fell over (tilt beyond 70°).

### G1 actuator model (PD gains and action scale)

From https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py (checked 2026-11-02):
- Armature (reflected rotor inertia) is computed from Unitree motor rotor inertias and two-stage planetary gear ratios. Stiffness Kp = armature × ω², damping Kd = 2 × ζ × armature × ω, with natural frequency ω = 2π × 10 Hz and damping ratio ζ = 2.0. Ankle and waist pitch/roll (parallel linkages driven by two 5020 motors) use 2 × the 5020 values.
- Effort limits: 5020 = 25 N·m, 7520-14 = 88 N·m, 7520-22 = 139 N·m, 4010 = 5 N·m.
- Action scale per joint = 0.25 × effort_limit / Kp.
- Resulting values [derived from the constants above]: 5020 (shoulders, elbows, wrist roll) Kp ≈ 14.3, Kd ≈ 0.91, scale ≈ 0.44; 7520-14 (hip pitch, hip yaw, waist yaw) Kp ≈ 40.2, Kd ≈ 2.56, scale ≈ 0.55; 7520-22 (hip roll, knee) Kp ≈ 99.1, Kd ≈ 6.31, scale ≈ 0.35; 4010 (wrist pitch, yaw) Kp ≈ 16.8, Kd ≈ 1.07, scale ≈ 0.075; ankle and waist pitch/roll Kp ≈ 28.5, Kd ≈ 1.81, scale ≈ 0.44.
- Initial state: knees-bent keyframe at base height 0.76 m. Soft joint limit factor 0.9.

---

## MuJoCo Warp (MJWarp)

- What it is: a GPU-accelerated version of MuJoCo built on NVIDIA Warp, maintained by Google DeepMind and NVIDIA as part of the Newton project. (source: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/README.md, checked 2026-11-02)
- Latest: 3.17.0, uploaded 2026-10-22; versions track MuJoCo's numbering (3.11.0 on 2026-07-28, 3.16.0 on 2026-10-14). The GitHub releases page still showed v3.11.0 as "Latest". (source: https://pypi.org/pypi/mujoco-warp/json, checked 2026-11-02; https://github.com/google-deepmind/mujoco_warp/releases, checked 2026-11-02)
- Backend: needs an NVIDIA GPU for fast simulation; CPU is supported for development and debugging. Requires `warp-lang>=1.15` and `mujoco>=3.13.0` (3.17.0). (source: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/README.md, checked 2026-11-02; https://pypi.org/pypi/mujoco-warp/json, checked 2026-11-02)
- Not differentiable; supports nearly all MuJoCo features except e.g. PGS and noslip solvers, the PLUGIN sensor type and SDF geoms (as listed in the MJX docs). (source: https://mujoco.readthedocs.io/en/stable/mjx.html, checked 2026-11-02)
- Install: `pip install mujoco-warp`. (source: https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/README.md, checked 2026-11-02)
- License: Apache-2.0. (source: https://pypi.org/pypi/mujoco-warp/json, checked 2026-11-02)
- Version note: mjlab 1.6.0 pins `mujoco-warp~=3.11.0`, six minor versions behind PyPI. Do not upgrade it separately. (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/pyproject.toml, checked 2026-11-02)
- When to pick: you want batched MuJoCo on GPU directly and will write your own env and training loop.

## MJX (MuJoCo XLA)

- What it is: a JAX API for MuJoCo physics, PyPI package `mujoco-mjx`. Two implementations: MJX-JAX (pure JAX; NVIDIA and AMD GPUs, Apple Silicon, TPUs) and MJX-Warp (`impl='warp'`, NVIDIA GPUs, uses MuJoCo Warp). (source: https://mujoco.readthedocs.io/en/stable/mjx.html, checked 2026-11-02)
- Latest: 3.17.0, uploaded 2026-10-22. (source: https://pypi.org/pypi/mujoco-mjx/json, checked 2026-11-02)
- Limits: MJX-JAX supports only FREE, BALL, SLIDE and HINGE joints, CG and Newton solvers, dense Jacobians, no Flex. MJX-Warp is not differentiable. (source: https://mujoco.readthedocs.io/en/stable/mjx.html, checked 2026-11-02)
- Install: `pip install mujoco-mjx`, or `pip install mujoco-mjx[warp]` for the Warp implementation. (source: https://mujoco.readthedocs.io/en/stable/mjx.html, checked 2026-11-02)
- License: Apache-2.0. (source: https://pypi.org/pypi/mujoco-mjx/json, checked 2026-11-02)
- When to pick: your training code is JAX, or you need gradients through the simulator (JAX implementation only).

## MuJoCo Playground

- What it is: open-source robot learning framework built on MJX, with environments for quadrupeds, humanoids, dexterous hands and arms, a batch renderer, and zero-shot sim-to-real from state and pixels. Paper: "MuJoCo Playground", Kevin Zakka, Baruch Tabanpour, Qiayuan Liao, Mustafa Haiderbhai, Samuel Holt, Jing Yuan Luo and 7 others (13 authors), arXiv 2502.08844 (12 Feb 2025). (source: https://arxiv.org/abs/2502.08844, checked 2026-11-02)
- Latest: `playground` 0.2.0, uploaded 2026-03-16. In 0.2.0 MuJoCo Warp became the default implementation for all environments; 0.1.0 (2026-01-08) added Warp support. Depends on `brax>=0.14.2`, `mujoco-mjx>=3.6.0`, `warp-lang>=1.11`, JAX, Flax. (source: https://pypi.org/pypi/playground/json, checked 2026-11-02; https://github.com/google-deepmind/mujoco_playground/releases, checked 2026-11-02)
- Install: `pip install playground`. (source: https://arxiv.org/abs/2502.08844, checked 2026-11-02)
- License: Apache classifier on PyPI; repo LICENSE is Apache 2.0. (source: https://pypi.org/pypi/playground/json, checked 2026-11-02; https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/LICENSE, checked 2026-11-02)
- When to pick: JAX + Brax PPO on ready-made environments. The releases page mentions a Booster T1 joystick task added in v0.0.4 and a G1 foot-friction randomization fix in the same release (source: https://github.com/google-deepmind/mujoco_playground/releases, checked 2026-11-02); the full environment list was not read.

## Isaac Lab

- What it is: NVIDIA's open-source robot learning framework built on Isaac Sim. (source: https://developer.nvidia.com/isaac-gym, checked 2026-11-02, which points to Isaac Lab as the successor)
- Latest: `isaaclab` 3.0.0, uploaded 2026-10-19; previous 2.3.2.post1 (2026-02-11). Python `>=3.12,<3.13`; pins `torch==2.11.0`, `warp-lang==1.17.0`, `gymnasium>=1.2.0`. (source: https://pypi.org/pypi/isaaclab/json, checked 2026-11-02)
- GitHub releases page showed v3.0.0-EA (16 Sep) as "Latest", requiring Isaac Sim 6.1, Python 3.12, PyTorch 2.11, Warp 1.16, and saying General Availability was targeted for end of October 2026; the docs release-notes page listed only v3.0.0-beta2.patch1 (Isaac Sim 6.0.1). Both pages lag PyPI. (source: https://github.com/isaac-sim/IsaacLab/releases, checked 2026-11-02; https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html, checked 2026-11-02)
- Backend: PhysX (default); Newton physics with multiple solvers including MuJoCo-Warp (quickstart uses `physics=newton_mjwarp`), VBD, MPM; OvPhysX experimental. (source: https://github.com/isaac-sim/IsaacLab/releases, checked 2026-11-02; https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html, checked 2026-11-02)
- Install: `pip install isaaclab`, with extras such as `isaacsim`, `rsl-rl`, `skrl`. Isaac Sim's PyPI metapackage `isaacsim` is 6.2.0.0 (2026-10-13), Python 3.12 only. (source: https://pypi.org/project/isaaclab/, checked 2026-11-02; https://pypi.org/pypi/isaacsim/json, checked 2026-11-02)
- License: Isaac Lab BSD-3-Clause; the `isaacsim` PyPI package declares "NVIDIA Proprietary Software". (source: https://pypi.org/pypi/isaaclab/json, checked 2026-11-02; https://pypi.org/pypi/isaacsim/json, checked 2026-11-02)
- When to pick: you need photoreal rendering, many sensor types, or Isaac Lab's existing task zoo. mjlab copies its manager API, so configs port with modest changes (mjlab has a migration guide at docs/source/migration_isaac_lab.rst; not read here).

## Isaac Gym (deprecated)

- Status: NVIDIA's page is titled "Isaac Gym - Now Deprecated" and says "This is legacy software. Developers may download and continue to use it, but it is no longer supported." It recommends Isaac Lab. Last release listed: Preview 4. (source: https://developer.nvidia.com/isaac-gym, checked 2026-11-02)
- IsaacGymEnvs (the companion task repo) was archived by its owner on 14 Apr 2026 and is read-only. (source: https://github.com/isaac-sim/IsaacGymEnvs, checked 2026-11-02)
- When to pick: only to reproduce older papers (legged_gym, Radosavovic et al. 2024, ASAP pre-training all used it).

## rsl_rl (`rsl-rl-lib`)

- What it is: "Fast and simple RL algorithms implemented in PyTorch" from ETH Zurich's Robotic Systems Lab; PPO and a teacher-student Distillation algorithm, with RND (random network distillation, a curiosity bonus) and symmetry extensions. (source: https://pypi.org/pypi/rsl-rl-lib/json, checked 2026-11-02; https://github.com/leggedrobotics/rsl_rl/blob/v5.4.2/docs/guide/configuration.rst, checked 2026-11-02)
- Latest: 5.5.2, uploaded 2026-10-28. mjlab 1.6.0's pinned 5.4.2 was uploaded 2026-07-15. Others: 5.5.0 (2026-08-27), 5.5.1 (2026-09-09). (source: https://pypi.org/pypi/rsl-rl-lib/json, checked 2026-11-02)
- 5.4.2 notes: fixes the loaded learning rate being ignored when resuming with the adaptive schedule. 5.5.0: bfloat16 mixed precision for PPO and Distillation, normalizer statistics synced across GPUs and updated after optimization; breaking: `logger_type` removed from `Logger`. 5.5.1: multi-GPU and wandb fixes. (source: https://github.com/leggedrobotics/rsl_rl/releases, checked 2026-11-02)
- PPO defaults are unchanged in 5.5.2; it adds `use_mixed_precision=False` and `grad_reduce_bucket_mb=25`. (source: https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v5.5.2/rsl_rl/algorithms/ppo.py, checked 2026-11-02)
- PPO defaults (5.4.2), from the docs table and the `PPO.__init__` signature: learning_rate 0.001, num_learning_epochs 5, num_mini_batches 4, schedule "adaptive", value_loss_coef 1.0, clip_param 0.2, use_clipped_value_loss True, desired_kl 0.01, entropy_coef 0.01, gamma 0.99, lam 0.95, max_grad_norm 1.0, normalize_advantage_per_mini_batch False, optimizer "adam". MLP model default hidden_dims [256, 256, 256], ELU, obs_normalization False. Gaussian distribution init_std 1.0, std_type "scalar", learn_std True. OnPolicyRunner: `num_steps_per_env` and `save_interval` are required; logger default "tensorboard" (plain "wandb" string is deprecated). (source: https://github.com/leggedrobotics/rsl_rl/blob/v5.4.2/docs/guide/configuration.rst, checked 2026-11-02; https://github.com/leggedrobotics/rsl_rl/blob/v5.4.2/rsl_rl/algorithms/ppo.py, checked 2026-11-02)
- Backend: PyTorch (`torch>=2.6.0`, `tensordict>=0.7.0`); Python `>=3.9`. (source: https://pypi.org/pypi/rsl-rl-lib/json, checked 2026-11-02)
- Install: `pip install rsl-rl-lib` (import name `rsl_rl`). mjlab installs it for you at the pinned version. (source: https://pypi.org/pypi/rsl-rl-lib/json, checked 2026-11-02)
- License: BSD-3-Clause. (source: https://pypi.org/pypi/rsl-rl-lib/json, checked 2026-11-02)
- When to pick: GPU-batched locomotion PPO where the env already returns torch tensors. Stay on 5.4.2 with mjlab 1.6.0 (exact pin).

## Brax

- What it is: "A differentiable physics engine written in JAX" plus JAX RL training code (PPO, SAC, ARS, evolution strategies, analytic policy gradients). (source: https://pypi.org/pypi/brax/json, checked 2026-11-02; https://raw.githubusercontent.com/google/brax/main/README.md, checked 2026-11-02)
- Status: README warning says only `brax/training` is actively maintained as of 0.13.0; use MuJoCo Playground instead of `brax/envs`, and MJX or MuJoCo Warp for physics. (source: https://raw.githubusercontent.com/google/brax/main/README.md, checked 2026-11-02)
- Latest: 0.14.2, uploaded 2026-03-15; it added a distributional critic option to PPO. 0.14.0 added adaptive learning rate, clipped value loss and value bootstrapping on time-outs to PPO. (source: https://pypi.org/pypi/brax/json, checked 2026-11-02; https://github.com/google/brax/releases, checked 2026-11-02)
- Install: `pip install brax` (Python `>=3.11`). (source: https://pypi.org/pypi/brax/json, checked 2026-11-02)
- License: Apache 2.0. (source: https://pypi.org/pypi/brax/json, checked 2026-11-02)
- When to pick: JAX PPO for Playground environments. Not useful with mjlab (PyTorch).

## ManiSkill3

- What it is: GPU-parallel robot simulation and benchmark powered by SAPIEN, focused on manipulation; GPU-parallel rendering (RGBD plus segmentation at 30,000+ FPS on a 4090 claimed), heterogeneous scenes per env, RL and IL baselines. Paper arXiv 2410.00425, stated as published at RSS 2025. (source: https://raw.githubusercontent.com/haosulab/ManiSkill/main/README.md, checked 2026-11-02)
- Latest: `mani-skill` 3.0.1, uploaded 2026-04-21 (previous upload 3.0.0b22 on 2025-12-05; 3.0.1 added Gymnasium >= 1.1 support). (source: https://pypi.org/pypi/mani-skill/json, checked 2026-11-02; https://github.com/haosulab/ManiSkill/releases, checked 2026-11-02)
- Backend: SAPIEN with PhysX, GPU or CPU simulation; best supported on Linux with NVIDIA GPU; rendering needs Vulkan. (source: https://raw.githubusercontent.com/haosulab/ManiSkill/main/README.md, checked 2026-11-02)
- Install: `pip install --upgrade mani_skill`, then `pip install torch`, then set up Vulkan. (source: https://raw.githubusercontent.com/haosulab/ManiSkill/main/README.md, checked 2026-11-02)
- License: Apache 2.0 (repo LICENSE; PyPI metadata has no license field). (source: https://raw.githubusercontent.com/haosulab/ManiSkill/main/LICENSE, checked 2026-11-02)
- When to pick: manipulation with camera observations at scale; not a locomotion tool.

## Genesis

- What it is: "A universal and generative physics engine"; Genesis World combines a multi-physics engine, a renderer (Nyx) and a compiler (Quadrants). Solvers: rigid body, FEM, MPM, particles (PBD and SPH), IPC, SAP, an explicit coupler. (source: https://pypi.org/project/genesis-world/, checked 2026-11-02)
- Latest: `genesis-world` 1.4.5, uploaded 2026-10-22; releases roughly every 1 to 3 weeks (1.4.0 on 2026-09-06). (source: https://pypi.org/pypi/genesis-world/json, checked 2026-11-02)
- Backend: Quadrants compiles to CUDA, AMD ROCm, Apple Metal, Vulkan, x86 and ARM64. (source: https://pypi.org/project/genesis-world/, checked 2026-11-02)
- Install: install PyTorch first, then `pip install genesis-world`; Python `>=3.10,<3.14`. (source: https://pypi.org/project/genesis-world/, checked 2026-11-02)
- License: Apache 2.0. (source: https://pypi.org/project/genesis-world/, checked 2026-11-02)
- When to pick: deformables or fluids, or a non-NVIDIA GPU. ASAP used it as a stand-in "real world" for sim-to-sim tests (see `../papers/canon-robotics.md`). The GitHub releases page failed to load, so release notes were not read.

## Gymnasium

- What it is: "A standard API for reinforcement learning and a diverse set of reference environments (formerly Gym)", Farama Foundation. (source: https://pypi.org/pypi/gymnasium/json, checked 2026-11-02)
- Latest: 1.4.0, uploaded 2026-10-05. Changes: `max_concurrency` for `AsyncVectorEnv`, generic vector env types, env checker verifies distinct obs and info objects per step, MuJoCo rendering camera fixes; no step-API or autoreset changes. The GitHub releases page still showed v1.3.0 (22 Apr) as "Latest". (source: https://pypi.org/pypi/gymnasium/json, checked 2026-11-02; https://gymnasium.farama.org/gymnasium_release_notes/, checked 2026-11-02; https://github.com/Farama-Foundation/Gymnasium/releases, checked 2026-11-02)
- terminated vs truncated: since v0.26, `env.step(action)` returns `obs, reward, terminated, truncated, info`. `terminated` = reached a terminal state of the MDP (e.g. fell over): do not bootstrap, the future value is 0. `truncated` = ended by an outside condition such as a time limit: bootstrap with V(next_obs). Both require `reset()`. (source: https://gymnasium.farama.org/introduction/migration_guide via Context7, checked 2026-11-02; https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits via Context7, checked 2026-11-02)
- Vector autoreset: since 1.0 the default is next-step autoreset (a sub-env that ends at step t returns its reset observation at step t + 1, with reward 0 and both flags False on that step). 1.1 added `AutoresetMode.NEXT_STEP`, `SAME_STEP` (final obs in `info["final_obs"]`) and `DISABLED`. (source: https://gymnasium.farama.org/gymnasium_release_notes via Context7, checked 2026-11-02; https://gymnasium.farama.org/_modules/gymnasium/vector/sync_vector_env via Context7, checked 2026-11-02)
- Install: `pip install gymnasium`; Python `>=3.10`. (source: https://pypi.org/pypi/gymnasium/json, checked 2026-11-02)
- License: MIT. (source: https://pypi.org/pypi/gymnasium/json, checked 2026-11-02)
- Relation to mjlab: mjlab's env returns `terminated` and `truncated` the same way, and its rsl_rl wrapper passes `truncated` as `time_outs` (see mjlab section). (source: https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/rl/vecenv_wrapper.py, checked 2026-11-02)

## PPO references: CleanRL and Stable-Baselines3

### Stable-Baselines3

- Latest: 2.9.2, uploaded 2026-10-26; Python `>=3.10`; MIT. (source: https://pypi.org/pypi/stable-baselines3/json, checked 2026-11-02)
- PPO defaults (master branch source): learning_rate 3e-4, n_steps 2048 (per env), batch_size 64, n_epochs 10, gamma 0.99, gae_lambda 0.95, clip_range 0.2, clip_range_vf None (no value clipping), normalize_advantage True, ent_coef 0.0, vf_coef 0.5, max_grad_norm 0.5, target_kl None, use_sde False. (source: https://github.com/dlr-rm/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py via Context7, checked 2026-11-02)
- Install: `pip install stable-baselines3`. (source: https://pypi.org/pypi/stable-baselines3/json, checked 2026-11-02)
- When to pick: a trusted CPU baseline on Gymnasium tasks. Its defaults target few envs and long rollouts, the opposite of rsl_rl's thousands of envs and 24-step rollouts.

### CleanRL

- PyPI `cleanrl` is 1.2.0 from 2023-05-22 with Python `>=3.7.1,<3.11`, so the pip package is stale; use the single-file scripts from the repo. License MIT. (source: https://pypi.org/pypi/cleanrl/json, checked 2026-11-02)
- `ppo_continuous_action.py` defaults: total_timesteps 1,000,000, learning_rate 3e-4, num_envs 1, num_steps 2048, anneal_lr True, gamma 0.99, gae_lambda 0.95, num_minibatches 32, update_epochs 10, norm_adv True, clip_coef 0.2, clip_vloss True, ent_coef 0.0, vf_coef 0.5, max_grad_norm 0.5, target_kl None. (source: https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_continuous_action.py, checked 2026-11-02)
- `ppo_continuous_action_isaacgym.py` defaults (massively parallel variant): num_envs 4096, num_steps 16, learning_rate 0.0026, anneal_lr False, num_minibatches 2, update_epochs 4, clip_vloss False, vf_coef 2, max_grad_norm 1, total_timesteps 30,000,000. (source: https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/ppo_continuous_action_isaacgym/ppo_continuous_action_isaacgym.py, checked 2026-11-02)
- The docs page says the continuous-action script follows "The 37 Implementation Details of Proximal Policy Optimization" (9 continuous-action details: Gaussian actions, state-independent log std, independent action components, separate policy and value MLPs, action clipping, observation normalization and clipping, reward scaling and clipping) and now recommends `rpo_continuous_action.py` over it. (source: https://docs.cleanrl.dev/rl-algorithms/ppo/, checked 2026-11-02)
- When to pick: reading a complete PPO in one file to check what rsl_rl does differently.

---

## Things that could not be verified

- Whether mjlab 1.6.0 accepts `--num-envs` (docs) in addition to `--env.scene.num-envs` (README, matches source). Not run.
- Isaac Lab 3.0.0 release notes on the docs site and GitHub: both pages lagged PyPI, so the GA notes themselves were not read.
- Genesis GitHub releases page: fetch failed.
- Isaac Gym Preview 4 release date: not shown on the NVIDIA page.
- MuJoCo Warp and Gymnasium: GitHub release pages disagreed with PyPI (older "Latest"); PyPI used.
- mjlab throughput on an A100: no number found in docs; the FAQ only claims mjlab is "on par or faster" than Isaac Lab.
