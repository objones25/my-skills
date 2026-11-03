# Policy gradient: foundations, PPO, theory

Optimize a parameterized policy π_θ by gradient ascent on its return, using only samples: the gradient needs ∇log π_θ, never a derivative of the dynamics. That is why it works for contact-rich robots where the simulator is not differentiable, and why PPO trains most legged-robot controllers today.

Sources: AJKS ch. 9 to 12 and SB ch. 13 (section numbers checked against the books; keys in `../sources.md`). Implementation facts tagged [rsl_rl 5.4.2] were read from an installed rsl-rl-lib 5.4.2 (`rsl_rl/algorithms/ppo.py`). Paper facts and default hyperparameters: `papers/canon-algorithms.md` (verified) and `papers/trpo.md`.

## Notation

| Symbol | Meaning |
|---|---|
| π_θ(a\|s) | policy with parameters θ; for robots usually Gaussian N(m_θ(s), diag σ²) with mean network m_θ, over joint-position targets |
| τ, p_θ(τ), R(τ) | a trajectory (s_0, a_0, s_1, ...), its probability under π_θ, and its discounted return Σ_t γ^t r_t |
| G_t | reward-to-go from step t, Σ_{k≥t} γ^{k−t} r_k |
| η | step size |
| J(θ) = V^{π_θ}(μ) | expected discounted return from start distribution μ |
| d^π_μ(s,a) | discounted visitation distribution, (1−γ) Σ_t γ^t P(s_t=s, a_t=a \| π, μ) (Notes 7 Def. 2) |
| A^π(s,a) | advantage Q^π(s,a) − V^π(s): how much better a is than the policy's average action |
| ρ_t(θ) | probability ratio π_θ(a_t\|s_t) / π_old(a_t\|s_t) (PPO writes r_t; renamed to avoid clashing with reward) |
| δ_t | TD error r_t + γV(s_{t+1}) − V(s_t) |
| λ | GAE mixing parameter in [0, 1] |
| ε | PPO clip range |

## 1. The policy gradient theorem

∇J(θ) = (1/(1−γ)) E_{(s,a)~d^{π_θ}_μ} [ Q^{π_θ}(s,a) ∇_θ log π_θ(a|s) ]   (AJKS §9.1; SB §13.2)

Mechanism in three steps:
1. Likelihood ratio: ∇ E_{τ~p_θ}[R(τ)] = E[R(τ) ∇log p_θ(τ)].
2. The dynamics cancel: log p_θ(τ) = log μ(s_0) + Σ_t [log π_θ(a_t|s_t) + log P(s_{t+1}|s_t,a_t)], and only the π terms depend on θ.
3. Causality: an action cannot affect earlier rewards, so a_t's score multiplies only the return from t on, which in expectation is γ^t Q^π(s_t, a_t); summing the γ^t weights over t is what turns the state distribution into d^π_μ and produces the 1/(1−γ).

Any b(s) can be subtracted from Q without bias, because E_{a~π}[∇log π(a|s)] = ∇Σ_a π(a|s) = ∇1 = 0. Choosing b = V^π gives the advantage form, a standard and good variance reducer (SB §13.4), though not the exactly variance-minimizing baseline.

## 2. Estimators, from high variance to biased

| Multiplier on ∇log π | Bias | Variance | Name |
|---|---|---|---|
| full return R(τ) | none | highest | likelihood-ratio gradient without the causality step |
| reward-to-go G_t | none | high | REINFORCE |
| G_t − V(s_t) | none if V is any function of s | lower | REINFORCE with baseline |
| δ_t | yes (V is wrong) | lowest | one-step actor-critic (SB §13.5) |
| Â_t^{GAE(γ,λ)} = Σ_l (γλ)^l δ_{t+l} | tunable | tunable | GAE (λ=0 is δ_t, λ=1 is G_t − V) |

GAE is computed backward over the rollout: Â_t = δ_t + γλ(1 − done_t) Â_{t+1}, value target R_t = Â_t + V(s_t). `scripts/rl_math.py` has a reference implementation with separate terminated and truncated flags.

## 3. Trust regions: from NPG to TRPO to PPO

A raw gradient step can change the policy's action distribution a lot for a small change in θ. Three answers, each cheaper than the last:
- Natural policy gradient: precondition with the Fisher information F(θ) = E[∇log π ∇log πᵀ], θ ← θ + η F⁻¹∇J (AJKS §10.4). Invariant to how π is parameterized.
- TRPO: maximize the surrogate E[ρ_t Â_t] subject to average KL(π_old ‖ π_θ) ≤ δ_KL (a trust-region radius; 0.01 in TRPO), solved with conjugate gradient on Fisher-vector products plus a line search (`papers/trpo.md`; AJKS §12.2). The monotone-improvement story comes from conservative policy iteration (AJKS §12.1) and the performance difference lemma below.
- PPO: drop the constraint and clip the ratio instead,
  L^CLIP(θ) = E_t[ min( ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t ) ].
  The min makes the objective a pessimistic bound: once ρ_t leaves [1−ε, 1+ε] in the direction that would help, the gradient through that sample is zero. Several epochs of minibatch SGD reuse each batch.

## 4. PPO as actually run on robots [rsl_rl 5.4.2]

What the shipped G1 velocity config in mjlab 1.6.0 sets (`mjlab/tasks/velocity/config/g1/rl_cfg.py`): actor and critic MLPs (512, 256, 128) with ELU, observation normalization on both, Gaussian policy with scalar std initialized to 1.0; value_loss_coef 1.0, clipped value loss, clip_param 0.2, entropy_coef 0.01, 5 epochs, 4 minibatches, learning_rate 1e-3 with schedule "adaptive", γ 0.99, λ 0.95, desired_kl 0.01, max_grad_norm 1.0, 24 steps per env per iteration.

Mechanisms worth knowing because each one changes results:
- Time-out bootstrapping: when an env times out, rsl_rl adds γ·V(s_t)·1[time_out] to that step's reward (`ppo.py` process_env_step). V(s_t) stands in for V(s_{t+1}) because the env has already auto-reset and the true final observation is gone. Falls get no bootstrap.
- GAE then treats every done (fall or time-out) as a cut: δ uses next_is_not_terminal (`compute_returns`).
- Advantages are normalized over the whole batch by default (normalize_advantage_per_mini_batch=False).
- Adaptive learning rate: after each minibatch, if the mean KL > 2·desired_kl, lr ← max(1e-5, lr/1.5); if KL < desired_kl/2, lr ← min(1e-2, lr·1.5). The KL here is the closed form KL(old ‖ new) between the two Gaussians (`torch.distributions.kl_divergence` in `modules/distribution.py`), evaluated on each minibatch before its gradient step, not a sample estimate.
- Clipped value loss: V_clipped = V_old + clamp(V − V_old, −ε, ε), loss is the max of the two squared errors. Andrychowicz et al. found value clipping hurts at every threshold they tried, and it is in the ppo2 code but not the PPO paper (`papers/canon-algorithms.md`); rsl_rl keeps it on.
- Gradient clipping on actor and critic separately, at max_grad_norm.
- Loss = surrogate + value_loss_coef·value_loss − entropy_coef·entropy.

## 5. Theory you will be asked about (AJKS ch. 9 to 12)

- Performance difference lemma (AJKS §1.5): V^{π'}(μ) − V^π(μ) = (1/(1−γ)) E_{s~d^{π'}_μ} E_{a~π'(·|s)}[A^π(s,a)]. The expectation is over the new policy's states, which is why every practical method needs the new policy to stay close to the old one (so d^{π'} ≈ d^π).
- Nonconvexity (AJKS §9.3): J is nonconvex in θ even for tabular softmax policies, yet softmax policy gradient converges to the global optimum (§10.2), possibly very slowly.
- Vanishing gradients and coverage (§10.1): if the start distribution μ rarely reaches the rewarding states, gradients are exponentially small; rates depend on a distribution-mismatch coefficient ‖d^{π⋆}_μ/μ‖_∞. Same lesson as the exploration lower bounds in Q-learning: policy gradient does not explore strategically.
- Log-barrier regularization (§10.3) and NPG (§10.4, ch. 11 with function approximation) fix the rate; Q-NPG and compatible function approximation are ch. 11.

## 6. Policy parameterizations for robots `[practice]`

- Gaussian over joint-position targets, fed to a PD controller (the action space nearly every locomotion paper uses; see `training/sim-to-real.md`). State-independent log std (rsl_rl "scalar" std) is the default; watch it in logs, since std collapsing early ends exploration and std growing means the critic is useless.
- Actions are usually scaled residuals around a default pose: q_target = q_default + scale·a. mjlab sets scale per joint to 0.25·effort_limit/stiffness (`g1_constants.py`).
- rsl_rl 5.4.2 also ships a Beta distribution (`modules/distribution.py`), which has bounded support by construction, so no clip or squash is needed.
- Bounding actions: clipping the sampled action changes the effective distribution but log π is computed before the clip (biased but common); tanh squashing needs the change-of-variables correction log π(a) = log N(u) − Σ log(1 − tanh²u) (SAC App. C).

## Common confusions

| Claim | Correction |
|---|---|
| "PPO is on-policy so data is used once" | each batch is reused for several epochs; the clip removes the incentive to push ρ further on samples already past 1 ± ε, which is what makes reuse workable |
| "The baseline must be V^π" | any function of s keeps the gradient unbiased; V^π is just a good variance reducer |
| "Clip range bounds the ratio or the KL" | it bounds neither: it only zeroes the gradient on samples past 1 ± ε in the helpful direction. Engstrom et al. found ratios routinely leave [1−ε, 1+ε] and that much of PPO's stability comes from code-level choices (`papers/canon-algorithms.md`); hence target-KL early stopping or rsl_rl's adaptive lr |
| "λ = 1 means no critic" | the critic is still the baseline in G_t − V(s_t); λ only controls bootstrapping |
