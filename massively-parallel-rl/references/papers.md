# Paper map

Each entry was opened and checked on 2026-10-27 (arXiv record, proceedings page, or
DOI). "Venue" comes from the proceedings or conference page; where only the arXiv page
was checked, the venue column says so. Cite these links, not memory.

## Massively parallel RL, 2024 to 2026 (newest first)

| Date | Work | Venue | One-line takeaway | Where used in this skill |
| --- | --- | --- | --- | --- |
| Oct 2026 | [Zheng, Lyle, Khetarpal, Dabney: An Empirical Study of RL Scaling Laws](https://arxiv.org/abs/2610.33420) | arXiv | Hyperparameter, model-size and Elo laws for A2C predict 1e18-1e19 runs from 1e14-1e17 sweeps; Craftax SOTA 54.0% | `scaling-laws.md` |
| Oct 2026 | [Zhang et al.: A Balanced Data Diet (Success Guided Sampling)](https://arxiv.org/abs/2610.12465) | arXiv | Uniform resets waste the batch at 1M envs; sample moderate-success configurations | `large-batch-on-policy.md` |
| May 2026 | [Park: Scalable On-Policy RL via Adaptive Batch Scaling](https://arxiv.org/abs/2605.21557) | arXiv | Grow the batch as the policy stops changing | `large-batch-on-policy.md` |
| Apr 2026 | [Kim et al.: FlashSAC](https://arxiv.org/abs/2604.04539) | arXiv | Normalized 2.5M-param SAC beats FastTD3 and PPO on high-dim control | `pytorch-gpu-sim.md` |
| Mar 2026 | [Beukman et al.: Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments](https://arxiv.org/abs/2603.06009) | RLJ 2026 ([PDF](https://rlj.cs.umass.edu/2026/papers/Paper19.pdf)) | PPO plateaus are an outer step too large for the noise; add minibatches as N grows | `large-batch-on-policy.md` |
| Dec 2025 | [Seo et al.: Learning Sim-to-Real Humanoid Locomotion in 15 Minutes](https://arxiv.org/abs/2512.01996) | arXiv | FastSAC/FastTD3 recipe, one RTX 4090 | `pytorch-gpu-sim.md` |
| Oct 2025 | [Obando-Ceron et al.: Simplicial Embeddings Improve Sample Efficiency in Actor-Critic Agents](https://arxiv.org/abs/2510.13704) | ICLR 2026 | Grouped-softmax layer helps FastTD3, FastSAC, PPO | `pytorch-gpu-sim.md` |
| Aug 2025 | [Fu et al.: Compute-Optimal Scaling for Value-Based Deep RL](https://arxiv.org/abs/2508.14881) | NeurIPS 2025 | Best batch grows with model size, shrinks with UTD; TD-overfitting | `scaling-laws.md`, `pytorch-gpu-sim.md` |
| May 2025 | [Seo et al.: FastTD3](https://arxiv.org/abs/2505.22642) | arXiv | Parallel envs + batch 32,768 + distributional critic | `pytorch-gpu-sim.md` |
| Mar 2025 | [Wang et al.: 1000 Layer Networks for Self-Supervised RL](https://arxiv.org/abs/2503.14858) | NeurIPS 2025, Best Paper | Depth to 1024 layers; deeper nets use bigger batches | `scaling-laws.md`, `jax-stack.md` |
| Mar 2025 | [Wang, Su, Gupta, Pathak: Evolutionary Policy Optimization](https://arxiv.org/abs/2503.19037) | arXiv | Evolved latent genes across a population beat SAPG | `large-batch-on-policy.md` |
| Feb 2025 | [Rybkin et al.: Value-Based Deep RL Scales Predictably](https://proceedings.mlr.press/v267/rybkin25a.html) | ICML 2025 | Data-compute frontier set by UTD | `scaling-laws.md` |
| Oct 2024 | [Matthews et al.: Kinetix](https://arxiv.org/abs/2410.23208) | ICLR 2025 (conference page) | Tens of millions of procedurally generated 2D physics tasks, billions of steps | `jax-stack.md` |
| Jul 2024 | [Singla et al.: SAPG](https://arxiv.org/abs/2407.20230) | ICML 2024 | PPO saturates on duplicate data; leader-follower split | `large-batch-on-policy.md` |
| Jul 2024 | [Gallici et al.: Simplifying Deep Temporal Difference Learning (PQN)](https://arxiv.org/abs/2407.04811) | ICLR 2025 (conference page) | Q-learning without replay or target net; LayerNorm; 50x faster than DQN on Atari | `jax-stack.md`, `scripts/pqn_jax.py` |

## Foundations the scaling work rests on

| Work | Takeaway |
| --- | --- |
| [Makoviychuk et al. 2021, Isaac Gym](https://arxiv.org/abs/2108.10470) | Physics and policy both on GPU: 2-3 orders of magnitude faster training |
| [Matthews et al. 2024, Craftax](https://proceedings.mlr.press/v235/matthews24a.html) | JAX Crafter: PPO for 1B steps in under an hour on one GPU |
| [Cobbe et al. 2020, Procgen](https://arxiv.org/abs/1912.01588) | Starpilot and the other procedurally generated benchmarks |
| [Mnih et al. 2016, A3C/A2C](https://proceedings.mlr.press/v48/mniha16.html) | The actor-critic Zheng et al. scale |
| [Espeholt et al. 2018, IMPALA](https://arxiv.org/abs/1802.01561) | ResNet torso; V-trace off-policy correction |
| [Hilton et al. 2023](https://arxiv.org/abs/2301.13442) | Single-agent RL scaling; return is not smooth, intrinsic performance |
| [Jones 2021](https://arxiv.org/abs/2104.03113), [Neumann & Gros 2023](https://arxiv.org/abs/2210.00849) | Elo scales with compute in two-player games |
| [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361), [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556), [Bi et al. 2024](https://arxiv.org/abs/2401.02954) | LM scaling laws, isoFLOPs profiles, hyperparameter laws |
| [Malladi et al. 2022](https://arxiv.org/abs/2205.10287) | Square-root learning-rate rule for Adam when batch size changes |
| [Schaul et al. 2019, Ray interference](https://arxiv.org/abs/1904.11455) | One reward source suppressing another causes plateaus |
| [Henderson et al. 2018](https://arxiv.org/abs/1709.06560) | RL results need many seeds |

## Code availability for the 2024-2026 papers (checked 2026-10-27)

| Work | Code |
| --- | --- |
| Zheng et al. 2026 | none linked in the paper |
| Success Guided Sampling | project page [sgs-rl.github.io](https://sgs-rl.github.io/) says "Code (coming soon)" |
| Adaptive Batch Scaling | [daisophila/ABS](https://github.com/daisophila/ABS), built on CleanRL; no license file; PQN path has a constructor bug (see `large-batch-on-policy.md`) |
| FlashSAC | [Holiday-Robot/FlashSAC](https://github.com/Holiday-Robot/FlashSAC) |
| PPO at 1M envs (Beukman et al.) | none released; the paper says its Isaac Gym runs start from SAPG's code |
| Humanoid sim-to-real in 15 min (FastSAC) | [amazon-far/holosoma](https://github.com/amazon-far/holosoma), linked from [the project page](https://younggyo.me/fastsac-humanoid) |
| Simplicial embeddings | [waltermayor/FastTD3_SEm](https://github.com/waltermayor/FastTD3_SEm), linked in the paper, is missing its own CLI flags; the working version is upstream [FastTD3](https://github.com/younggyoseo/FastTD3) (PR #35) |
| Fu et al. 2025 | none linked; built on BRO and SimbaV2 |
| Rybkin et al. 2025 | none found in the paper text checked |
| EPO | none; project page has videos only |
| Previous Craftax best (Hamon 2024) | [Reytuag/transformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX) |

## Code (licenses checked in each repo's LICENSE file)

| Repo | License | What to take from it |
| --- | --- | --- |
| [purejaxrl](https://github.com/luchris429/purejaxrl) | Apache-2.0 | Reference jitted PPO, PPO-RNN, DQN |
| [purejaxql](https://github.com/mttga/purejaxql) | Apache-2.0 | PQN for MinAtar, Atari, Craftax, Playground |
| [rejax](https://github.com/keraJLi/rejax) | Apache-2.0 | vmappable algorithm objects |
| [Craftax](https://github.com/MichaelTMatthews/Craftax) | MIT | Environment; baselines are in a separate repo |
| [Kinetix](https://github.com/FLAIROx/Kinetix) | MIT | Multi-GPU PPO with shard_map; SFL curriculum |
| [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) | Apache-2.0 | On-device replay buffer, contrastive RL |
| [scaling-crl](https://github.com/wang-kevin3290/scaling-crl) | Apache-2.0 | Deep residual critics |
| [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) | Apache-2.0 | MJX tasks and Brax PPO/SAC configs |
| [brax](https://github.com/google/brax) | Apache-2.0 | PPO with truncation masking |
| [FastTD3](https://github.com/younggyoseo/FastTD3) | MIT | GPU replay, compiled updates |
| [FlashSAC](https://github.com/Holiday-Robot/FlashSAC) | MIT | Normalized large SAC |
| [SAPG](https://github.com/jayeshs999/sapg) | MIT (rl_games MIT, isaacgymenvs BSD-3) | Leader-follower PPO |
| [PufferLib](https://github.com/PufferAI/PufferLib) | MIT | Native CUDA PPO with Muon |
| [Craftax_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines) | MIT | Official Craftax PPO, PPO-RNN, RND, ICM, E3B; optimistic reset wrapper |
| [transformerXL_PPO_JAX](https://github.com/Reytuag/transformerXL_PPO_JAX) | MIT | Transformer-XL PPO; prior Craftax best |
| [holosoma](https://github.com/amazon-far/holosoma) | Apache-2.0 | FastSAC and PPO for G1/T1 humanoids, sim-to-real |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | BSD-3 | Default robotics PPO: adaptive-KL lr, symmetry, RND, multi-GPU |
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | BSD-3 (Isaac Sim parts proprietary) | GPU robot tasks; Newton physics by default |
| [mjlab](https://github.com/mujocolab/mjlab) | Apache-2.0 | Isaac Lab-style API on MuJoCo Warp |
| [ABS](https://github.com/daisophila/ABS) | none stated | Adaptive batch scaling for PQN and PPO |
