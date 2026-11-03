# FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control

Younggyo Seo, Carmelo Sferrazza, Haoran Geng, Michal Nauman, Zhao-Heng Yin, Pieter Abbeel (UC Berkeley; Nauman also University of Warsaw). Technical report, arXiv:2505.22642 (v3, June 2025, cs.RO); the PDF names no conference. Code and project page: https://younggyo.me/fast_td3

## The idea

FastTD3 is plain TD3 (an off-policy actor-critic with a deterministic actor and twin critics) run with massively parallel simulation, very large batches (32,768), a distributional (C51) critic, clipped double Q-learning, and tuned exploration noise, all in one synchronous process with the replay buffer on the GPU [FastTD3 Abstract, §2.1-2.2]. The authors do not claim novelty over Parallel Q-Learning (PQL), which found the same ingredients; the contribution is a simpler recipe without PQL's asynchronous processes, ablations of each choice, and an easy codebase [§1, §4]. It solves many HumanoidBench tasks in under 3 hours on one A100 and trains faster than PPO on IsaacLab and MuJoCo Playground humanoid locomotion, especially rough terrain with domain randomization [§1].

## Notation

FastTD3 does not restate TD3; it refers readers to Fujimoto et al. (2018) [§2.1]. The TD3 symbols below are standard background, marked as such.

| Symbol | Meaning |
|---|---|
| s, a, r, s′ | State (observation vector), action vector (joint targets), reward, next state |
| γ | Discount factor (value not given in the paper) |
| μ_φ(s) | Deterministic actor with parameters φ (TD3 background). Output shape: action dimension |
| Q_{θ₁}, Q_{θ₂} | The two critics of TD3, with slowly updated target copies (TD3 background) |
| CDQ | Clipped Double Q-learning: the TD target bootstraps from the minimum of the two target critics, to curb overestimation (TD3 background; the min is what the paper ablates) [§2.1] |
| C51 critic | Distributional critic of Bellemare et al. (2017): the critic outputs a categorical distribution over a fixed grid of return values ("atoms") spanning [v_min, v_max], trained by cross-entropy to the projected Bellman target [§2.1]. Number of atoms not given in the paper |
| v_min, v_max | Support bounds of the C51 return distribution; extra hyperparameters, tuned values only in the code [§2.1, footnote 1] |
| num_envs | Number of parallel simulated environments |
| N | Replay capacity per environment; total buffer = N × num_envs, stored on GPU [§2.1, §2.2] |
| B | Batch size per gradient update (32,768) |
| UTD | Update-to-data ratio: gradient updates per environment step (here, per batched step of all parallel envs) [§2.1] |
| σ_min, σ_max | Range from which each environment's Gaussian exploration-noise std is sampled ("mixed noise", from PQL) [§2.1] |

## The method

### 1. Parallel environments, synchronous loop

IsaacLab and MuJoCo Playground run natively in parallel on the GPU. HumanoidBench has no GPU parallelism, so it uses Stable Baselines3's SubprocVecEnv; its default per-environment GPU renderer made more than about 100 environments hard, so the authors added (and got merged) an option to disable it [§2.2]. The hypothesis for why parallelism helps a deterministic-policy method: randomness across many environments diversifies the data, covering TD3's weak exploration while keeping its strong value exploitation [§2.1]. All suites are wrapped to the RSL-RL API (Rudin et al.); for MuJoCo Playground the authors forked the wrapper to save final observations before resets [§2.2].

### 2. Large-batch, low-UTD updates from a per-environment replay buffer

Batch size 32,768 per update. Larger batches cost more per update but usually cut total training time, because with many environments and no large batch most data would never be sampled [§2.1]. FastTD3 runs at very low UTD, typically 2, 4, or 8 updates per batched step of 128 to 4096 parallel environments; raising UTD improves sample efficiency at the cost of wall time, with no extra stabilizers needed [§2.1, Fig. 6c]. The buffer holds N transitions per environment (total N × num_envs) so that adding environments does not cut how much of each trajectory is stored; larger N helped, at the cost of GPU memory [§2.1, §2.2].

### 3. Critic: distributional C51 plus clipped double Q

The C51 distributional critic helped in most cases, at the cost of tuning v_min and v_max, which the authors found easy [§2.1, Fig. 5c]. Using the minimum of the two critics (CDQ) beat not using it; the authors note that Nauman et al. found averaging better with layer normalization, but without layer normalization CDQ remained important and should be tuned per task [§2.1, Fig. 5d]. The paper does not say exactly how the min is applied to two return distributions; check the code.

### 4. Networks

Critic MLP 1024-512-256, actor MLP 512-256-128, no layer norm and no residual connections. Smaller models hurt both time and sample efficiency; residual paths and layer norm (as in BRO or Simba) slowed training without clear gains. The authors' hypothesis is that parallel data plus big batches reduce off-policyness and so the instability of the "deadly triad" (bootstrapping, function approximation, off-policy learning) [§2.1, Fig. 6a]. When IsaacLab or MuJoCo Playground provide privileged state, the critic can use it (asymmetric actor-critic, Pinto et al. 2017) [§2.2].

### 5. Exploration noise

Each environment gets its own Gaussian noise std sampled from [σ_min, σ_max] (PQL's mixed noise). The mixing itself gave no significant gain but was kept for flexible scheduling; σ_max = 0.4 was helpful [§2.1, Fig. 6b]. σ_min is not given in the paper.

### 6. Speed engineering

PyTorch with AMP in bfloat16 gave up to 40% faster training on one A100 with no instability; torch.compile (codebase built on LeanRL) gave up to 35%; together up to 70% [§2.2].

## Hyperparameters

The paper has no hyperparameter table; it says tuned values are in the open-source implementation [§2.1, footnote 1]. Values stated in the text:

| Setting | Value in paper | Source |
|---|---|---|
| Base algorithm | TD3 (deterministic actor, twin critics) | §2 |
| Batch size | 32,768 | §2.1 |
| Critic MLP | 1024, 512, 256 | §2.1 |
| Actor MLP | 512, 256, 128 | §2.1 |
| Normalization / residuals | None | §2.1 |
| Critic type | Distributional C51 with [v_min, v_max] | §2.1, Fig. 5c |
| Clipped double Q | On (min of two critics) | §2.1, Fig. 5d |
| Exploration | Mixed Gaussian, per-env std in [σ_min, σ_max], σ_max = 0.4 | §2.1 |
| UTD | Typically 2, 4, or 8 updates per 128-4096 parallel env steps | §2.1 |
| Replay buffer | N × num_envs, all on GPU | §2.1-2.2 |
| Precision / compile | AMP bfloat16, torch.compile | §2.2 |
| Hardware | 1 NVIDIA A100, 16 CPU cores | §3 |
| Learning rates, γ, target update rate, policy delay, target smoothing noise, atoms, v_min/v_max, σ_min, num_envs per suite, N | Not given in the paper (see code) | |

Ablation grids, run on HumanoidBench h1hand_hurdle and h1hand_slide, 3 seeds each [Figs. 5-6]:

| Factor | Values tested | Outcome in the plots |
|---|---|---|
| Parallel envs | 1, 4, 32, 64, 128 | More is better; 128 best |
| Batch size | 2K, 8K, 32K | 32K best |
| Distributional critic | off vs C51 | C51 better |
| CDQ | off vs on | On better |
| Model size (labels as printed; the paper does not say which network D refers to) | D256 (0.18M params), D512 (0.55M), D1024 (1.83M) | D1024 best |
| Max noise σ_max | 0.001, 0.1, 0.4 | 0.4 best on hurdle, similar on slide |
| UTD | 1, 2, 4, 8, 16 | Higher UTD more sample-efficient, slower in wall time |
| Buffer N per env | 1K, 10K, 50K | 50K best; 1K fails |

## Key results

Baselines [§3]: DreamerV3, SAC and TD-MPC2 curves come from the HumanoidBench repository (3 runs each, 48 hours, plotted with interpolated wall-clock timestamps); SimbaV2 is run with its official code on a single seed with action repeat 1 (its paper used 2); PPO is the comparison on IsaacLab and MuJoCo Playground. FastTD3 uses 3 seeds. One A100 and 16 CPU cores.

- Aggregate curves [Fig. 3] (normalized return, which the paper does not define further; values below are approximate readings off the plot, not reported numbers): HumanoidBench, 39 tasks, at 10,800 s (3 h): FastTD3 about 68, SimbaV2 about 35, PPO and SAC about 10, TD-MPC2 and DreamerV3 lower. MuJoCo Playground, 4 tasks, at 3,600 s: FastTD3 about 88, PPO about 58. IsaacLab, 6 tasks, at 3,600 s: FastTD3 about 75, PPO about 55, with PPO slightly ahead in the first few hundred seconds.
- Per-task curves for all 39 HumanoidBench tasks, 6 IsaacLab tasks (G1 and H1 velocity flat/rough, Allegro and Shadow cube repose), and 4 MuJoCo Playground tasks (G1 and T1 joystick flat/rough) are in App. A (wall-clock) and App. B (environment steps). The sample-efficiency plots run to roughly 1.5-3×10⁸ environment steps for IsaacLab and about 2×10⁸ for MuJoCo Playground, while the matching wall-clock plots stop at 3,600 s [Figs. 10-11, 13-14].
- Sim-to-real: a FastTD3 policy trained in MuJoCo Playground (Booster Gym's 12-DoF Booster T1 config and rewards, ported) was deployed on a real full-size Booster T1 humanoid, which the authors believe is the first documented real-world deployment of an off-policy RL policy on a full-size humanoid [Fig. 1, §3].
- Reward functions do not transfer across algorithms: FastTD3 on a PPO-tuned G1 reward gave jerky, undeployable arm motion; a FastTD3-tuned reward with stronger penalty terms gave a smooth gait, while PPO on that reward walked too slowly. Episode return did not capture deployability [§3, Fig. 7].
- FastSAC (the same recipe applied to SAC) trains much faster than vanilla SAC but slower than FastTD3 and is less stable, which the authors attribute to entropy maximization in high-dimensional action spaces [§3, Fig. 8].

## Gotchas when reproducing

- The batch size of 32,768 and the large per-environment buffer are load-bearing (2K batch and 1K buffer clearly fail in Figs. 5b and 6d).
- Keep CDQ on unless you also add layer normalization [§2.1].
- σ_max = 0.4 exploration noise; small noise (0.001 or 0.1) learns hurdle much slower [Fig. 6b].
- C51 needs v_min and v_max set to match the task's return scale; use the repo's values [§2.1].
- Do not shrink the networks to save time; smaller critics were slower in wall clock too [§2.1, Fig. 6a].
- Store final observations before auto-reset; the stock MuJoCo Playground wrapper did not, so the authors forked it [§2.2]. A wrong next-state at episode boundaries corrupts TD targets.
- Reward functions tuned for PPO may give bad gaits under FastTD3; budget for re-tuning penalties [§3, Fig. 7].
- Most hyperparameters live only in the code, so reproduce from the official configs rather than the paper.

## Connections

Q-learning (deep) and actor-critic units: FastTD3 is a deep Q-learning-style critic (TD targets, target networks, clipped double Q, distributional value) driving a deterministic policy gradient actor. It is the off-policy counterpart to the PPO papers (legged-gym, SAPG) in the same massively parallel setting, and a good contrast for "on-policy vs off-policy at scale".
