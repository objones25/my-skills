# Actor-critic and off-policy continuous control

An actor-critic learns two functions: a critic (V or Q) that scores actions, and an actor (the policy) trained to pick actions the critic likes. On-policy actor-critics (PPO with a value baseline) use the critic only to reduce variance; off-policy ones (DDPG, TD3, SAC) use a Q critic as a differentiable stand-in for the return, which is how Q-learning extends to continuous actions (Notes 7 §4: "learn a function S ↦ A (or S ↦ △(A)) that tries to approximate argmax_a Q(s,a)").

Sources: SB §13.5 and the papers below. Equations and default hyperparameters are verified in `papers/canon-algorithms.md`; the scale-up story is in `papers/fasttd3.md` and `training/off-policy-at-scale.md`.

## Notation

| Symbol | Meaning |
|---|---|
| Q_φ, Q_φ' | critic and its slowly updated target copy |
| π_θ, π_θ' | actor and target actor (deterministic μ_θ(s) in DDPG/TD3, stochastic in SAC) |
| τ | Polyak rate: φ' ← τφ + (1−τ)φ' after each update |
| α | SAC entropy temperature |
| d | 1 if the transition ended in a true termination (not a time-out), else 0 |

## The family in one table

| Method | Actor | Critic target y | Key trick |
|---|---|---|---|
| One-step AC (SB §13.5) | stochastic, on-policy | r + γV(s') | bootstrapped advantage δ |
| A2C / PPO | stochastic, on-policy | GAE return | batch from many envs, clipped surrogate (`policy-gradient.md`) |
| DDPG | deterministic μ_θ | r + γ(1−d) Q_φ'(s', μ_θ'(s')) | deterministic policy gradient ∇_a Q · ∇_θ μ |
| TD3 | deterministic | r + γ(1−d) min_{i=1,2} Q_φ'_i(s', μ_θ'(s') + clipped noise) | clipped double Q, target policy smoothing, delayed actor |
| SAC | stochastic, tanh-Gaussian | r + γ(1−d)(min_i Q_φ'_i(s', a') − α log π(a'\|s')), a' ~ π | maximum entropy, learned α |

Why two critics and the min: Q-learning's max over noisy estimates is biased upward (SB §6.7, maximization bias); with an actor doing the max by gradient ascent, the actor exploits critic errors. Taking the min of two critics is a pessimistic correction (TD3).

## Choosing on-policy versus off-policy for a robot `[practice]`

| Situation | Usually pick | Reason |
|---|---|---|
| Thousands of GPU envs, cheap samples, wall-clock matters | PPO | stable with huge batches; legged_gym recipe (`training/on-policy-at-scale.md`) |
| Few envs or real hardware, samples expensive | SAC / TD3 / RLPD / CrossQ | replay reuses each sample many times |
| Massively parallel and want sample efficiency too | FastTD3, PQN, PQL, SAPG | large-batch off-policy or off-policy-corrected PPO (`training/off-policy-at-scale.md`) |
| Demonstrations available | BC pretraining, then RL fine-tune, or RLPD with demos in the buffer | `imitation.md` |

## Asymmetric actor-critic

The critic is only used during training, so it may see privileged simulator state the deployed actor cannot (true base velocity, contact forces, terrain, friction). Pinto et al. 2018 (`papers/canon-robotics.md`). mjlab's G1 velocity task does this: the critic group adds foot height, air time, contact flags and contact forces and has no observation noise, while the actor's observations are corrupted with uniform noise (`velocity_env_cfg.py` critic_terms vs actor_terms, enable_corruption). Caveat [inference]: the gradient is exactly unbiased only when the critic acts as a pure baseline (λ = 1 in GAE). With λ < 1 (0.95 in rsl_rl) the critic also bootstraps the advantage, so a critic conditioned on information the actor lacks can bias the actor's gradient under partial observability; Baisero and Amato (2022) analyze this (not yet verified in this skill). In practice asymmetric critics work well for locomotion.

## Failure signatures

| Symptom | Likely cause |
|---|---|
| Q values climb without bound | overestimation feedback; check the min over critics, target network τ, reward scale vs 1/(1−γ) |
| SAC entropy collapses immediately | target entropy too low or α learning rate too high; target is usually −dim(A) (SAC v2) |
| Actor improves then crashes | actor updating faster than the critic tracks; TD3's delayed actor updates exist for this |
| Explained variance near 0 (PPO critic) | critic not learning: reward scale, observation normalization off, value lr too low |
