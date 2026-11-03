# Massively parallel RL survey: index

Source: Owen Beckett Jones, "RL Scaling Laws: References and Massively Parallel Training", Oct 27, 2026, 11 pages. Every link below is copied from the doc's hyperlinks and was not re-verified for this note; the doc says it opened each one [Survey p. 1, p. 10]. Page tags are [Survey p. N].

## What the doc covers

Two things for Zheng et al. 2026 (`rl-scaling-laws.md`): all 35 references resolved to links, and a survey of 2024-2026 massively parallel RL work [Survey p. 1]. "Massively parallel" means thousands of environment copies on one accelerator, N envs × T rollout steps = the batch per update; Zheng et al. fix T = 16 and vary N [Survey p. 1, p. 4]. Headline: plain PPO stops improving as envs are added, and fixes come from three directions: change how extra data is used (smaller policy steps, more diverse behavior, smarter resets), switch to off-policy learners, or scale the network [Survey p. 5]. Its figure assigns each work to one stage of the loop: reset sampler, N envs, batch N × T, learner update, policy network [Survey p. 5 figure].

## Index of surveyed works

| Work | Year | Link (from the doc) | Stage | Contribution, as the doc states it |
|---|---|---|---|---|
| Success Guided Sampling (Zhang et al.), "A Balanced Data Diet" | Oct 2026 | https://arxiv.org/abs/2610.12465 | Resets | Uniform resets waste a growing share of the batch at up to 2^20 envs; weighting toward tasks with moderate success gives 73% vs 54% (PLR) on a 13-terrain quadruped, 70% vs 6% on Franka nut-and-bolt [p. 6, p. 8] |
| Adaptive Batch Scaling (Park) | May 2026 | https://arxiv.org/abs/2605.21557 | Batch | Grow the batch (2,048 to 16,384) as the policy stops changing; a fixed 16,384 batch hurts larger PQN models, the schedule helps [p. 6] |
| FlashSAC (Kim et al.) | Apr 2026 | https://arxiv.org/abs/2604.04539 | Network | Heavy normalization lets SAC use a 10× larger network than FastTD3 (2.5M params, 1,024 envs); G1 walks in about 20 min vs about 3 h for PPO [p. 6, p. 9] |
| PPO at 1M environments (Beukman et al.), RLJ 2026 | Mar 2026 | https://arxiv.org/abs/2603.06009 (RLJ PDF: https://rlj.cs.umass.edu/2026/papers/Paper19.pdf) | Batch | PPO plateaus come from an outer step too large relative to update noise; hold minibatch size and LR fixed and add minibatches as envs grow; monotonic gains to 10^12 steps on >1M envs, 128 GPUs; cutting Isaac Gym minibatches from 96k to 16k improved PPO everywhere and narrowed the gap to SAPG [p. 6, p. 8] |
| Humanoid sim-to-real in 15 minutes (Seo et al.) | Dec 2025 | https://arxiv.org/abs/2512.01996 | Learner | FastSAC and FastTD3 recipe trains deployable G1 and T1 locomotion in about 15 min on one RTX 4090, up to 16,384 envs [p. 7] |
| Simplicial embeddings (Obando-Ceron et al.), ICLR 2026 | Oct 2025 | https://arxiv.org/abs/2510.13704 | Network | Grouped-softmax feature layer improves sample efficiency of FastTD3, FastSAC and PPO at no wall-clock cost [p. 7] |
| Compute-optimal value-based RL (Fu et al.), NeurIPS 2025 | Aug 2025 | https://arxiv.org/abs/2508.14881 | Learner | Big batches hurt small critics ("TD-overfitting": a small critic gives bad targets that a big batch fits tightly); optimal batch grows with model size, shrinks with UTD [p. 7, p. 9] |
| FastTD3 (Seo et al.) | May 2025 | https://arxiv.org/abs/2505.22642 | Learner | Parallel envs + batch 32,768 + distributional critic solve HumanoidBench tasks in under 3 h on one A100 [p. 7]; notes in `fasttd3.md` |
| 1000-layer networks (Wang et al.), NeurIPS 2025 best paper | Mar 2025 | https://arxiv.org/abs/2503.14858 | Network | Depth up to 1,024 layers gives 2× to 50× gains in self-supervised goal-conditioned RL; bigger batches pay off only for deeper nets (jumps at about 8 layers on Ant Big Maze, 64 on Humanoid U-Maze) [p. 7, p. 9] |
| EPO, Evolutionary Policy Optimization (Wang, Su, Gupta, Pathak) | Mar 2025 | https://arxiv.org/abs/2503.19037 | N envs | Evolves per-policy latent "genes" with a genetic algorithm (6,144 to 49,152 envs); beats SAPG on two-arm tasks, 35.8 vs 2.2 successes [p. 7, p. 8] |
| Value-based RL scales predictably (Rybkin et al.), ICML 2025 | Feb 2025 | https://proceedings.mlr.press/v267/rybkin25a.html | (not in figure) | Data and compute for a target return lie on a frontier set by UTD, so budgets can be extrapolated [p. 8] |
| SAPG (Singla et al.), ICML 2024 | Jul 2024 | https://proceedings.mlr.press/v235/singla24a.html (also https://arxiv.org/abs/2407.20230) | N envs | PPO saturates because envs take near-identical actions; 6 policies with importance-weighted data sharing beat DexPBT by 12% to 66% on three hard manipulation tasks [p. 8]; notes in `sapg.md` |
| PQN (Gallici et al.), ICLR 2025 | Jul 2024 | https://arxiv.org/abs/2407.04811 | Learner | Q-learning without replay buffer or target network, using LayerNorm and λ-returns over vectorized envs; 50× faster than standard DQN on Atari [p. 8, p. 9] |

Background tools the doc opened [p. 4, p. 11]: Isaac Gym (Makoviychuk et al. 2021, https://arxiv.org/abs/2108.10470), 2 to 3 orders of magnitude faster than CPU sim + GPU learner; Craftax (Matthews et al. 2024, https://proceedings.mlr.press/v235/matthews24a.html), PPO for 1B env steps in under an hour on one GPU; Kinetix (Matthews et al. 2025, https://arxiv.org/abs/2410.23208), listed as opened but not summarized.

## Which ones answer: how to use the extra data and big batches

Ranked by how directly they apply to PPO on mjlab + rsl_rl with num_envs 64..16384 on one A100 [ranking is inference]:

1. PPO at 1M envs (Beukman): rsl_rl splits each rollout into a fixed number of minibatches, so minibatch size grows with num_envs (verified: mjlab's G1 config sets num_mini_batches=4, and the G1 sweep's README shows minibatch 384 at 64 envs up to 98,304 at 16,384); Beukman's recipe says hold minibatch size fixed and raise the minibatch count instead [p. 8, p. 9]. Cheapest change to test in the sweep.
2. Adaptive Batch Scaling (Park): grow num_envs or batch during training instead of fixing it [p. 6].
3. SAPG and EPO: spend extra envs on behavioral diversity (several policies) instead of more near-duplicate samples [p. 8].
4. Success Guided Sampling: spend extra envs on informative start states; for locomotion, a terrain or command curriculum weighted toward moderate success [p. 6, p. 8; locomotion reading is inference]. Also ties to the RVFS/reset project.
5. Network size coupling (1000-layer, Fu, FlashSAC, simplicial embeddings): bigger batches help mainly if the network is big enough; consider widening the actor-critic when num_envs is large [p. 9].
6. Off-policy alternatives (FastTD3, Seo et al. 15-min, FlashSAC, PQN): use throughput for replay-based learners; G1-specific wall-clock numbers make them a baseline for any PPO-at-scale comparison [p. 6, p. 7].
7. Zheng et al. and Rybkin et al.: how to pick batch and model size as a function of budget [p. 5, p. 8].

## Doc notes worth keeping

- Zheng et al.'s laws per 10× compute: batch ×2.45 / ×1.82, LR ×0.95 / ×0.83, FPF ×2.75 / ×2.69 (Starpilot / Craftax) [p. 9]; matches Tab. 1 arithmetic in `rl-scaling-laws.md`.
- The doc flags the Zheng §4.1 learning-rate sign error [p. 10] and two reference fixes: Fu et al. is NeurIPS 2025 (vol. 38), not 2026; DeepSeek-R1 also has a Nature version [p. 1].
- Claims the PPO-at-1M paper shares four authors with Zheng et al. (Khetarpal, Zheng, Dabney, Lyle) and that both find LR should barely move with scale [p. 9].
- Open questions it raises for Zheng et al.: refit the batch law with fixed minibatch size; test whether diversity methods remove the Craftax Ray-interference dip; other algorithms; depth scaling; data use (135B frames) [p. 10].
- Venue sources: PQN, SAPG and Kinetix venues come from conference pages in search results; RLJ venue from its PDF; FlashSAC, FastTD3 and simplicial-embedding numbers quote text only, not curves [p. 10].

## Inconsistencies in the doc

- The loop figure's caption says "5 stages, 13 methods", but the figure places 12 (Success Guided Sampling, SAPG, EPO, PPO at 1M, Adaptive Batch Scaling, PQN, FastTD3, FastSAC, Fu et al., 1000 layers, SEM layer, FlashSAC); Rybkin et al. is in the table but not the figure [p. 5, p. 8].
- The figure lists "FastSAC (2025)" as a method; the table entry is the Seo et al. 15-minute paper, which uses FastSAC and FastTD3 [p. 5, p. 7].
- Kinetix is in the "papers opened" list but has no table row or summary [p. 11].
