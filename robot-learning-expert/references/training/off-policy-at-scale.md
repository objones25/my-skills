# Off-policy RL with massively parallel simulation

Replay-based learners (TD3, SAC, Q-learning variants) reuse each sample many times, so they win when samples are expensive. With GPU simulators samples are cheap, and the 2024 to 2026 line of work shows off-policy methods can also win on wall-clock if they use very large batches, big critics, and parallel envs. For G1 locomotion specifically, the survey doc reports FlashSAC walking in about 20 min versus about 3 h for PPO, and the Seo et al. FastSAC/FastTD3 recipe training deployable G1 and T1 policies in about 15 min on one RTX 4090 (`papers/massively-parallel-survey.md`, figures as the doc states them, not re-verified).

Sources: `papers/fasttd3.md` (read in full), `papers/massively-parallel-survey.md` (index of 2024 to 2026 work), `papers/canon-algorithms.md` (TD3, SAC, PQL, PQN, CrossQ, RLPD), `concepts/actor-critic.md` and `concepts/q-learning.md` for the algorithms themselves.

## The ingredients that recur

| Ingredient | What it does | Where shown |
|---|---|---|
| Many parallel envs feeding one replay buffer | diverse, fresher data, so the buffer is less off-policy | FastTD3 §2.1, PQL |
| Very large batches (FastTD3: 32,768) | most collected data would never be sampled otherwise | FastTD3 §2.1, Fig. 5b |
| Low update-to-data ratio (UTD 2 to 8 per batched step) | wall-clock friendly; raise UTD for sample efficiency | FastTD3 §2.1, Fig. 6c |
| Per-env buffer capacity N × num_envs | adding envs does not shorten how much of each trajectory is stored | FastTD3 §2.2, Fig. 6d |
| Clipped double Q; distributional critic (C51) | CDQ controls overestimation; C51 helped in FastTD3's ablations | FastTD3 §2.1 |
| Bigger critics than actors (1024-512-256 vs 512-256-128) | smaller models hurt both time and sample efficiency in FastTD3; small critics overfit big batches ("TD-overfitting", Fu et al.) | FastTD3 §2.1; survey p. 7 |
| Normalization (LayerNorm, batch renorm in CrossQ, heavy normalization in FlashSAC) | lets larger networks and no-target-network training stay stable | PQN, CrossQ, FlashSAC (survey) |
| Asymmetric critic with privileged state | same as on-policy | FastTD3 §2.2 |

## Qualifiers on the FlashSAC wall-clock claim

The "about 20 min vs about 3 h on G1" figure is one task in a larger comparison. FlashSAC reports parity with PPO on low-DoF state tasks and clear gains on high-DoF ones (humanoid, dexterous) [2604.04539 abstract, line 53]. Budgets differ: off-policy methods were trained for 50M environment steps and PPO for 200M, at about 3x FlashSAC's compute [2604.04539 lines 392-394]; FlashSAC uses one unified hyperparameter set while baselines were tuned per task [line 425]. Sabatini et al. find SAC closes the return gap to PPO on all tasks and exceeds it on humanoids, yet stays slower in wall-clock (stated for the whole study, not quadrupeds only), from larger networks, more loss terms, a higher UTD ratio and a learning rate about 10x smaller than PPO's [2605.24975 lines 426-445]; they also report higher seed variance than PPO.

## SAC in Isaac Lab / rsl_rl checklist (Sabatini et al., arXiv 2605.24975)

Setting: legged locomotion in Isaac Lab, 8192 envs, 24 rollout steps, 800 iterations [Table C9]. Values below are that paper's, not general defaults:
- Action bounds from the soft joint limits (distance from the default pose to the soft limits, divided by the action scale), not the hard limits [lines 247-257].
- Initial action noise std σ0 = 0.15, which concentrates exploration near the default pose; large σ0 gives near-uniform exploration [lines 277, Table C9].
- Store the pre-reset next observation and handle time-outs by bootstrapping from it; the n-step target is masked over the window [lines 318-352]. n = 5.
- Table C9: replay 5e6, 200 minibatches of 8192 per iteration, actor and critic lr 2e-4, temperature lr 2e-5, γ = 0.97, τ = 0.003, networks [1024, 512, 256] with SiLU, no layer norm.
- Expect: SAC needs larger networks and higher UTD, is slower than PPO per second, and varies more across seeds; on G1 high reward did not always mean natural movement [lines 426-445].

## On-policy or off-policy for this kind of project

- Throughput-bound, single GPU, a working PPO config exists (as for G1 walking on mjlab): PPO is the baseline; an off-policy run (FastTD3 or FastSAC on the same task) is the most informative second arm, because the surveyed G1 wall-clock claims are exactly that comparison.
- PQN is the Q-learning analogue of PPO's recipe: vectorized envs, no replay, no target network, LayerNorm and λ-returns (survey p. 8). Notes 7's FQI view explains why it can work: each update is still an approximate Bellman backup, just with fresher data.
- Off-policy needs truncation handled the same way: bootstrap on time-outs (y = r + γ(1 − terminated) Q'(s', a')), which requires storing the true final observation before auto-reset.

## Practical pitfalls `[practice]`

- Replay buffers on the GPU are memory-bound: N × num_envs × (obs + act + reward + done) floats; compute it before a sweep.
- Exploration noise for deterministic actors matters: FastTD3 samples a per-env noise std from [σ_min, σ_max] (PQL's mixed noise); the mixing gave no significant gain, but σ_max = 0.4 learned much faster than 0.1 or 0.001 on the hurdle task (`papers/fasttd3.md`, Fig. 6b).
- Hyperparameters in FastTD3 are mostly in its code, not the paper; cite the code commit if you reproduce it.
