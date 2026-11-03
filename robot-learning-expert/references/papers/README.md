# Paper notes

One file per paper (plus two canon lists verified against arXiv and proceedings pages), written from full reads of the PDFs (algorithm boxes checked against page images). Claims carry location tags like [PoR Thm 4.1]; [inference] marks synthesis that is not in the paper. Each file ends with the paper's inconsistencies or ambiguities where any were found.

| File | Paper | Read it when |
|---|---|---|
| `power-of-resets.md` | Mhammedi, Foster, Rakhlin, The Power of Resets in Online RL (NeurIPS 2024) | reset-based exploration: RVFS pseudocode (§5), implementation (§7), noisy observations (§8), PDF ambiguities (§9) |
| `ugts.md` | Mhammedi, Cohan, Uncertainty Guided Tree Search / GowU (arXiv 2026) | a practical reset-based explorer with experiments; baseline or testbed for reset experiments |
| `theory-map.md` | synthesis of the six theory papers | how coverability, pushforward coverability, BE dimension and access models relate |
| `coverage.md` | Xie, Foster, Bai, Jiang, Kakade, The Role of Coverage in Online RL (ICLR 2023) | coverability C_cov, GOLF under coverability |
| `bellman-eluder.md` | Jin, Liu, Miryoosefi, Bellman Eluder Dimension (NeurIPS 2021) | BE dimension, GOLF, how it contains Bellman rank and Eluder dimension |
| `coordinated-exploration.md` | Dimakopoulou, Van Roy, Coordinated Exploration in Concurrent RL (2018) | seed sampling, why optimism gives no diversity across many parallel agents |
| `mcts-spectral.md` | Riviere, Lathrop, Chung, MCTS with Spectral Expansion (Science Robotics 2024) | MCTS for continuous deterministic dynamical systems (SETS) |
| `icrl-transformers.md` | Liang, Lai, Transformers Provably Implement In-Context RL (arXiv 2026) | in-context RL constructions (SARSA / actor-critic step in linear attention) |
| `trpo.md` | Schulman et al., TRPO (ICML 2015) | trust regions, the policy gradient theory and PPO units |
| `sapg.md` | Singla, Agarwal, Pathak, SAPG (ICML 2024) | PPO with very many parallel envs; leader/follower off-policy aggregation |
| `fasttd3.md` | Seo et al., FastTD3 (arXiv 2025) | off-policy TD3 at scale, deep Q-learning and actor-critic units |
| `legged-gym.md` | Rudin et al., Learning to Walk in Minutes (CoRL 2021) | massively parallel PPO recipe, time-out bootstrapping, curriculum, sim-to-real |
| `canon-robotics.md` | 18 canonical sim-to-real, locomotion, imitation and VLA papers (Tobin 2017 to ASAP 2025), verified against arXiv PDFs 2026-11-02 | DR ranges, PD/action spaces, control rates, teacher-student designs; ends with a sim-to-real locomotion recipe |
| `supermimic.md` | Liao et al., SuperMimic: humanoid whole-body control with one differentiable world model (arXiv 2610.28480, Oct 2026) | humanoid motion tracking, world-model policy gradients vs PPO, G1 noise/DR/reward tables |
| `rl-scaling-laws.md` | Zheng, Lyle, Khetarpal, Dabney, An Empirical Study of RL Scaling Laws (arXiv 2610.33420, Oct 2026) | how batch size (num_envs), learning rate and model size should grow with compute |
| `massively-parallel-survey.md` | survey doc "RL Scaling Laws: References and Massively Parallel Training" (Oct 27, 2026) | index of 2024 to 2026 massively parallel work, ranked for a PPO-at-scale study |
| `canon-algorithms.md` | verified list: DQN, Double DQN, PER, DDPG, TD3, SAC, PPO, GAE, PPO implementation studies, DAgger, GAIL, DreamerV3, TD-MPC2, RLPD, PQL, PQN, CrossQ, KL estimators | equations and default hyperparameters you can cite |

Known caveats from the distillation pass:
- Venues not printed on an arXiv PDF were taken from citations in the other papers; the Dimakopoulou and Van Roy venue (ICML 2018) is unverified.
- FastTD3 has no hyperparameter table in the paper (values live in its code); its summary numbers are plot readings.
- The coverability coefficient is defined over greedy policies of the class in Coverage but over all policies in Power of Resets, so the two C_cov values are not interchangeable.
