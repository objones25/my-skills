# Policy gradient: foundations, PPO, theory

Optimize a parameterized policy π_θ by gradient ascent on its return, using only samples: the gradient needs ∇log π_θ, never a derivative of the dynamics. That is why it works for contact-rich robots where the simulator is not differentiable, and why PPO trains most legged-robot controllers today.

Sources: AJKS ch. 9 to 12 and SB ch. 13 (section numbers checked against the books; keys in `../sources.md`); Notes 8 (policy gradient part I) and Notes 9 (advanced: TRPO, PPO, optimality, PC-PG), which use a finite horizon t = 1..H with no discount, so their Q_t^π is the undiscounted reward-to-go from step t (errata for both in `../sources.md`). Implementation facts tagged [rsl_rl 5.4.2] were read from an installed rsl-rl-lib 5.4.2 (`rsl_rl/algorithms/ppo.py`). Paper facts and default hyperparameters: `papers/canon-algorithms.md` (verified) and `papers/trpo.md`.

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

Why not differentiate the return directly: with deterministic P and π, ∇J already needs ∂r/∂a at step 1, and from step 2 on the chain rule pulls in the Jacobians of P, i.e. a model (Notes 8 §2, warm-up). Two model-free answers:
- Zeroth-order: smooth J by averaging over a ball, Ĵ(θ) = E_{v~Unif(ball)}[J(θ + σv)]; then ∇Ĵ(θ) = (n/σ) E_{ε~Unif(sphere)}[J(θ + σε) ε] for θ ∈ ℝⁿ (Notes 8 Thm 1). Needs only return evaluations, but J itself is a noisy expectation, so it needs care (the evolution-strategies family).
- Score function (log-derivative trick, Notes 8 Thm 2): ∇_θ E_{z~p_θ}[f(z)] = E_{z~p_θ}[∇_θ log p_θ(z) f(z)], under differentiability and regularity conditions. Needs: sampling from p_θ, ∇log p_θ, and the value f(z). Does not need ∇f or samples from any other distribution. With z a trajectory this is the policy gradient below.

∇J(θ) = (1/(1−γ)) E_{(s,a)~d^{π_θ}_μ} [ Q^{π_θ}(s,a) ∇_θ log π_θ(a|s) ]   (AJKS §9.1; SB §13.2)

Mechanism in three steps:
1. Likelihood ratio: ∇ E_{τ~p_θ}[R(τ)] = E[R(τ) ∇log p_θ(τ)].
2. The dynamics cancel: log p_θ(τ) = log μ(s_0) + Σ_t [log π_θ(a_t|s_t) + log P(s_{t+1}|s_t,a_t)], and only the π terms depend on θ.
3. Causality: an action cannot affect earlier rewards, so a_t's score multiplies only the return from t on, which in expectation is γ^t Q^π(s_t, a_t); summing the γ^t weights over t is what turns the state distribution into d^π_μ and produces the 1/(1−γ).

Finite-horizon, undiscounted form (Notes 8 §2.2 to 2.3): REINFORCE ĝ = (Σ_t r_t)(Σ_t ∇log π_θ(a_t|s_t)) is unbiased; dropping the rewards before t gives ĝ' = Σ_t ∇log π_θ(a_t|s_t) Σ_{s≥t} r_s with E[ĝ'] = E[Σ_t ∇log π_θ(a_t|s_t) Q_t^π(s_t,a_t)]. The causality step needs the return to be a sum of per-step rewards. The full-return REINFORCE works for any trajectory score R(s_1, a_1, ..., s_H, a_H), e.g. the best reward reached, max_t r(s_t, a_t).

Any b(s) can be subtracted from Q without bias, because E_{a~π}[∇log π(a|s)] = ∇Σ_a π(a|s) = ∇1 = 0. Choosing b = V^π gives the advantage form, a standard and good variance reducer (SB §13.4), though not the exactly variance-minimizing baseline. For a constant b and the full-return estimator, the variance-minimizing value is b⋆ = E[‖g‖²R] / E[‖g‖²] with g = Σ_t ∇log π_θ(a_t|s_t), the return weighted by the squared score norm; an exponential moving average of R approximates it (Notes 8 §4; Greensmith, Bartlett and Baxter 2004). (Derivation: E[g] = 0, so tr Cov[g(R − b)] = E[‖g‖²(R − b)²] − ‖E[gR]‖², a quadratic in b.)

Softmax bandit, the smallest worked example (Notes 8 §3): π_θ(a) = exp(θ_a)/Σ exp(θ_a'), ∇log π_θ(a) = e_a − π. A single sample gives ĝ = r(a)(e_a − π), so its component on a is r(a)(1 − π(a)) > 0 whenever r(a) > 0: every sampled action is made more likely, however small its reward, and only the average cancels this. That is variance; an advantage can be negative, which fixes the sign. Initializing θ = C(1, −1, ..., −1) with large C makes π ≈ e_1, so a_1 is nearly always sampled and e_{a_1} − π ≈ 0: tiny steps, no exploration.

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

A raw gradient step is steepest ascent over parameter steps of fixed Euclidean length (Notes 9 §2), but equal-length steps in θ do not give equal changes in the action distribution. Notes 9 §3's example: π(a_1) ∝ e^{θ_1}, π(a_2) ∝ e^{Cθ_2} with C ≫ 1 is far more sensitive to θ_2. The fix is steepest ascent measured by KL between distributions.

Derivation of the TRPO step (Notes 9 §3):
1. Objective: linearize at θ, giving ⟨E[Σ_t ∇log π_θ(a_t|s_t) Q_t^π(s_t,a_t)], θ' − θ⟩, an expectation under the old policy.
2. Constraint: KL(d^{π_θ} ‖ d^{π_θ'}) ≤ ε over trajectory distributions. Its gradient is zero at θ' = θ (a minimum), so keep the quadratic term.
3. The Hessian of KL(p ‖ q_θ) at q_θ = p is the Fisher matrix E_p[∇log q ∇log qᵀ], with no second derivatives needed (Notes 9 Lemma 2, App. A). For trajectories the dynamics drop out as in §1, leaving F = E[Σ_t ∇log π ∇log πᵀ]. The cross terms between steps vanish in expectation because each step's score has conditional mean zero.
4. Maximize aᵀx subject to xᵀFx ≤ ε: x⋆ = sqrt(ε / (aᵀF⁻¹a)) F⁻¹a (Notes 9 prints a spurious minus sign; see `../sources.md`). Solve F⁻¹a by conjugate gradient. The sample F is singular unless N·H ≥ d (N trajectories, d parameters), which fails for large networks. Finish with a line search because the quadratic model is only local.

Three answers, each cheaper than the last:
- Natural policy gradient: precondition with the Fisher information F(θ) = E[∇log π ∇log πᵀ], θ ← θ + η F⁻¹∇J (AJKS §10.4). Invariant to how π is parameterized.
- TRPO: maximize the surrogate E[ρ_t Â_t] subject to average KL(π_old ‖ π_θ) ≤ δ_KL (a trust-region radius; 0.01 in TRPO), solved with conjugate gradient on Fisher-vector products plus a line search (`papers/trpo.md`; AJKS §12.2). The monotone-improvement story comes from conservative policy iteration (AJKS §12.1) and the performance difference lemma below.
- PPO: drop the constraint and clip the ratio instead,
  L^CLIP(θ) = E_t[ min( ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t ) ].
  The min makes the objective a pessimistic bound: once ρ_t leaves [1−ε, 1+ε] in the direction that would help, the gradient through that sample is zero. Several epochs of minibatch SGD reuse each batch.
  Three facts from Notes 9 §4 that are easy to miss:
  - Reuse is importance sampling, E_{p_θ}[f] = E_p[(p_θ/p) f]. One rare sample with a large f can blow up the ratio.
  - On the first gradient step ρ_t = 1 everywhere, so the clip is invisible and the step is the plain policy gradient. Clipping only acts from the second step on the same batch.
  - The advantages Â_t are computed once and held fixed through all inner steps; only ρ_t changes. With Â_t < 0 the ratio is never clipped from above: making a bad action more likely is always penalized (Notes 9 Fig. 1).
- NPG versus TRPO: NPG uses the same direction F⁻¹∇J but a fixed step size, so it does not enforce the KL radius each iteration (Notes 9 §3; the notes add that TRPO "generally performs better").

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
- Stationary points (Notes 9 Thm 4, citing Zhang et al. 2020): under Lipschitz and smoothness assumptions, with step sizes satisfying Σ η_k = ∞ and Σ η_k² < ∞ (Robbins-Monro), policy gradient converges to a stationary point of J. That says nothing yet about optimality.
- Restricted policy classes can have bad local optima (Notes 9 §5, from Bhandari and Russo 2019). In a two-state cost MDP with the class "play R with probability θ in every state", the cost is nonconvex in θ, and a small initial θ descends to a bad local minimum at θ = 0. The missing property is *closure under policy improvement*: one policy-iteration step from any π_θ must land back in the class (Notes 9 Def. 5).
- Global optimality (Notes 9 Thm 6, Bhandari and Russo 2019): suppose the one-step improvement objective has no suboptimal stationary points, the class is closed under policy improvement, and further exploration and smoothness conditions hold. Then J has no suboptimal stationary points. If that objective is gradient-dominated (f(z) ≤ min f + (c/2)‖∇f‖² for a cost f), J is too, which gives convergence rates. Tabular MDPs satisfy these conditions. So does LQR, with care about unbounded cost: its cost is nonconvex and the set of stabilizing gains is nonconvex, yet the cost is gradient-dominated (Fazel et al. 2018; `continuous-trajopt.md`). These results assume exact gradients and a start distribution μ with full support, so they sidestep exploration.
- Provable exploration with policy gradient, PC-PG (Notes 9 §6.1, Agarwal, Henaff, Kakade and Sun 2020): in a linear MDP, keep a policy cover, i.e. the mixture of all previous policies, with feature covariance Σ̂_cov = Σ_i Σ̂^i + λI. Each episode, add the bonus 1{φ(s,a)ᵀ Σ̂_cov⁻¹ φ(s,a) ≥ β}/(1−γ) for rarely visited pairs, and optimize the next policy for r + bonus with starts drawn from the cover's distribution. The policy step is itself NPG on estimated Q. The bonus has the same shape as LSVI-UCB's (`q-learning.md`), and it is the first policy gradient result in these notes that does not rely on a good initial state distribution.

## 6. Exploration comes from the policy class

Policy gradient explores only through the randomness of π_θ (Notes 8 §5). A class that never takes some actions never learns about them, so check that π_θ(a|s) > 0 for all θ, s, a. Notes 9 §6 adds: initialize with high action entropy and do not let it collapse. Entropy regularization adds a bonus to the objective, J(θ) = E_s[E_{a~π}[r(s,a)] + α H(π(·|s))] with α > 0 (Notes 8 §5.1). For a scalar Gaussian, H = ½ log(2πeσ²), so the bonus directly rewards keeping σ large. Entropy regularization can also make policies more robust to changes in the dynamics (Eysenbach and Levine 2021, cited in Notes 8 §5.1). In rsl_rl this is entropy_coef (§4).

## 7. Policy parameterizations for robots `[practice]`

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
