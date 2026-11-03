# Trust Region Policy Optimization (TRPO)

John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, Pieter Abbeel. ICML 2015 (PMLR vol. 37). arXiv:1502.05477 (v5 used here).

## The idea

The return of a new policy can be written as the old return plus the new policy's expected advantage under the old policy's advantage function. Replacing the new policy's state distribution with the old one gives a surrogate objective L that is exact to first order, and the paper proves that L minus a KL penalty is a lower bound on the true return, so maximizing it never makes the policy worse [TRPO §3, Eq. 9-10]. TRPO turns this penalty into a hard constraint on the average KL divergence and solves the resulting problem approximately with conjugate gradient plus a line search [TRPO §4, §6, App. C].

## Notation

| Symbol | Meaning |
|---|---|
| (S, A, P, r, ρ₀, γ) | Infinite-horizon discounted MDP: state set, action set, transition probability P(s'∣s,a), reward r(s) (state-only in this paper), initial-state distribution ρ₀, discount γ ∈ (0,1) [§2] |
| π(a∣s) | Stochastic policy. π is the current policy, π̃ a candidate new policy |
| θ ∈ ℝᵈ | Policy parameter vector (d parameters); π_θ is the parameterized policy; θ_old is the current iterate |
| η(π) | Expected discounted return E[Σₜ γᵗ r(sₜ)] with s₀ ~ ρ₀ and actions from π |
| Q_π(s,a), V_π(s) | Action-value and state-value functions of π |
| A_π(s,a) | Advantage, Q_π(s,a) − V_π(s): how much better action a is than π's average action at s |
| ρ_π(s) | Unnormalized discounted state visitation: P(s₀=s) + γP(s₁=s) + γ²P(s₂=s) + ... under π |
| L_π(π̃) | Surrogate objective η(π) + Σ_s ρ_π(s) Σ_a π̃(a∣s) A_π(s,a) [Eq. 3] |
| D_TV(p‖q) | Total variation distance, ½ Σᵢ ∣pᵢ − qᵢ∣ |
| D_KL^max(π, π̃) | max over states of D_KL(π(·∣s) ‖ π̃(·∣s)) |
| D̄_KL^ρ(θ₁, θ₂) | Average KL, E_{s~ρ}[D_KL(π_θ₁(·∣s) ‖ π_θ₂(·∣s))] |
| α | D_TV^max(π_old, π_new) in Theorem 1; the mixing weight in Kakade-Langford's mixture policy (Eq. 5) |
| ε | max_{s,a} ∣A_π(s,a)∣ in Theorem 1 (Kakade-Langford use a slightly different ε, Eq. 6) |
| C | Penalty coefficient 4εγ/(1−γ)² |
| δ | Trust region size: the bound on average KL per update |
| q(a∣s) | Sampling distribution used for importance sampling (q = π_θold for single path) |
| A (matrix) | Fisher information matrix, d×d, the Hessian of average KL at θ_old. Not the advantage |
| g | Gradient of the surrogate L at θ_old, a d-vector |
| s (vector), β | Search direction ≈ A⁻¹g (d-vector) and step length |
| μ_θ(x) | Network output that parameterizes the action distribution (for Gaussians, mean and std) |
| J, M | J = ∂μ/∂θ, shape (dim μ × d); M = Fisher of the action distribution with respect to μ, shape (dim μ × dim μ) |

## The method

### 1. Start from an exact identity and a local approximation

For any two policies, η(π̃) = η(π) + E_{τ~π̃}[Σₜ γᵗ A_π(sₜ, aₜ)] (Kakade and Langford's identity) [§2, Eq. 1; proof in App. A, Lemma 1]. Rewriting with visitation frequencies gives η(π̃) = η(π) + Σ_s ρ_π̃(s) Σ_a π̃(a∣s) A_π(s,a) [Eq. 2]. The trouble is that ρ_π̃ depends on the new policy. Swapping in ρ_π gives the surrogate L_π(π̃) [Eq. 3]. For a differentiable π_θ, L matches η in value and gradient at θ_old [Eq. 4], so a small enough step that improves L improves η, but this alone says nothing about how big a step is safe.

### 2. The monotonic improvement bound

Kakade and Langford (2002) bounded improvement only for mixture policies π_new = (1−α)π_old + απ′ [Eq. 5]: η(π_new) ≥ L_π_old(π_new) − 2εγ/(1−γ)² · α² [Eq. 6]. TRPO's Theorem 1 extends this to any stochastic policy by setting α = D_TV^max(π_old, π_new):

η(π_new) ≥ L_π_old(π_new) − (4εγ/(1−γ)²) α², with ε = max_{s,a} ∣A_π(s,a)∣ [§3, Eq. 8].

Using D_TV(p‖q)² ≤ D_KL(p‖q) (Pollard) gives η(π̃) ≥ L_π(π̃) − C·D_KL^max(π, π̃), C = 4εγ/(1−γ)² [Eq. 9]. Call the right side M_i(π). Since M_i(π_i) = η(π_i), maximizing M_i each iteration gives η(π_{i+1}) − η(π_i) ≥ M_i(π_{i+1}) − M_i(π_i) ≥ 0, so returns never decrease [Eq. 10, Algorithm 1]. This is a minorization-maximization (MM) scheme: M_i is a lower bound on η that touches it at the current policy. Two proofs are given: a coupling argument (App. A) and a perturbation-theory argument (App. B).

### 3. Three approximations that make it practical

- Penalty to constraint. With the theoretical C the steps are tiny, and a penalty coefficient is hard to pick robustly, so TRPO solves: maximize L_θold(θ) subject to D_KL^max(θ_old, θ) ≤ δ [§4, Eq. 11].
- Max KL to average KL. A constraint at every state is impractical, so it uses D̄_KL^{ρ_θold}(θ_old, θ) ≤ δ [Eq. 12]. On cart-pole (the only task where max KL was tractable), max KL learned somewhat slower but similarly [§8.1].
- Sample-based form. Replace Σ_s ρ(s)[...] by (1/(1−γ)) E_{s~ρ}[...], replace A by Q (changes the objective by a constant), and importance-sample actions:

  maximize_θ E_{s~ρ_θold, a~q}[ (π_θ(a∣s)/q(a∣s)) Q_θold(s,a) ] subject to E_{s~ρ_θold}[D_KL(π_θold(·∣s) ‖ π_θ(·∣s))] ≤ δ [§5, Eq. 14].

  The theory ignores advantage estimation error [§6].

### 4. Estimating Q: single path vs vine

| | Single path [§5.1] | Vine [§5.2] |
|---|---|---|
| Data | Roll out π_θold from s₀ ~ ρ₀; use every (sₜ, aₜ) | Roll out "trunk" trajectories, pick N states (the rollout set), sample K actions per state from q, do a short rollout after each |
| q | π_θold | Any q covering π's support; π_θi worked for locomotion, uniform for Atari |
| Q estimate | Discounted sum of future rewards along the trajectory | Rollout return per (sₙ, a_{n,k}), using common random numbers across the K rollouts to cut variance |
| Estimator | Importance-sampled surrogate | Small discrete A: Lₙ(θ) = Σₖ π_θ(aₖ∣sₙ) Q̂(sₙ,aₖ) [Eq. 15]. Large/continuous A: self-normalized IS estimator [Eq. 16], which needs no baseline |
| Pros/cons | No state resets needed, works on a physical system | Much lower-variance advantage estimates, but many more simulator calls and requires resetting to arbitrary states (simulation only) |

### 5. Solve the constrained step: CG, step size, line search [App. C]

Linearize L and take a quadratic model of the KL: D_KL ≈ ½(θ−θ_old)ᵀA(θ−θ_old).

1. Search direction: solve A s = g approximately with conjugate gradient (CG), which only needs matrix-vector products y ↦ Ay, never A itself. k = 10 CG iterations worked well; more did not speed up improvement [App. C.1].
2. Maximal step: from δ = ½(βs)ᵀA(βs), β = √(2δ / sᵀAs). The quantity sᵀAs is one extra Fisher-vector product and also falls out of CG.
3. Line search: starting from β, shrink exponentially until the surrogate L improves and the actual (nonlinear) KL ≤ δ. The paper does not state the shrink factor. "Without this line search, the algorithm occasionally computes large steps that cause a catastrophic degradation of performance" [App. C].

### 6. Fisher-vector products [§6, App. C.1]

The Fisher matrix is built analytically as the Hessian of the KL, A_ij = (1/N) Σₙ ∂²/∂θᵢ∂θⱼ D_KL(π_θold(·∣sₙ) ‖ π_θ(·∣sₙ)), not as the empirical covariance of score-function gradients. It integrates over actions at each state, so it does not depend on the sampled action, and it avoids storing a dense Hessian or all per-sample gradients. Writing the KL as kl(μ_θ(x), μ_old), its Hessian is JᵀMJ (the term with second derivatives of μ vanishes at θ_old) [Eq. 57]. So Ay = Jᵀ(M(Jy)): a forward-mode product, a closed-form multiply by M for the distribution, and a backprop. A generic Hessian-vector product of the KL also works and is easier to code, slightly slower. A Fisher-vector product costs about one gradient; with k = 10 a naive implementation spends over 90% of compute on them, so compute them on a 10% subsample of the data, which brings the natural-gradient step to roughly the cost of the gradient [App. C.1].

Policy parameterization [App. D]: continuous actions use a Gaussian with an MLP mean and a state-independent log-std vector r (std = exp(r)); Atari uses a factored categorical action space with a softmax per factor.

### Relation to other updates [§7]

Linear L plus quadratic KL constraint gives the natural policy gradient θ_new = θ_old + (1/λ) A⁻¹∇L [Eq. 17]; TRPO differs by enforcing the constraint each update (step size from δ, plus line search) rather than using a fixed step size λ. An ℓ² constraint gives the vanilla policy gradient [Eq. 18]. Unconstrained maximization of L gives policy iteration. REPS constrains state-action marginals p(s,a); TRPO constrains conditionals p(a∣s).

## Key results

Setup [§8.1, App. E, Table 2]: MuJoCo swimmer, hopper, walker; δ = 0.01 everywhere; γ = 0.99; 200 policy iterations; 5 random seeds per algorithm.

| | Swimmer | Hopper | Walker |
|---|---|---|---|
| State dim / control dim | 10 / 2 | 12 / 3 | 20 / 6 (text in §8.1 says 18) |
| Policy params | 364 | 4806 | 8206 |
| Hidden layer size | 30 | 50 | 50 |
| Sim steps per iteration | 50K | 1M | 1M |
| Vine: rollout length / rollouts per state / Q-values per batch | 50 / 4 / 500 | 100 / 4 / 2500 | 100 / 4 / 2500 |
| Vine compute time (min) | 2 | 14 | 40 |
| Single path: paths / path length | 50 / 1000 | 1000 / 1000 | 10000 / 1000 |
| Single path compute time (min) | 5 | 35 | 100 |

Rewards: swimmer r = vₓ − 10⁻⁵‖u‖²; hopper adds +1 per non-terminal step and terminates on falling; walker adds a foot-impact penalty [§8.1].

- Single path and vine TRPO solved all locomotion tasks and gave the best solutions. Natural gradient (fixed step size, swept in factors of three) did well on cart-pole and swimmer but did not produce forward-moving hopping or walking gaits. CEM and CMA (derivative-free) did poorly on larger problems. A score of −1 on hopper/walker means standing still [Fig. 4]. Exact final returns are only plotted, not tabulated.
- Empirical FIM (covariance of gradients) improved at a rate similar to the analytic FIM [§6, Fig. 4].
- Atari, 7 games, raw pixels, CNN with two conv layers (16 channels, stride 2) and one 20-unit FC layer, 33,500 params; 500 iterations took about 30 hours on a 16-core computer; one run per game, no error bars [§8.2, Table 3]:

| | Beam Rider | Breakout | Enduro | Pong | Q*bert | Seaquest | Space Invaders |
|---|---|---|---|---|---|---|---|
| Random | 354 | 1.2 | 0 | −20.4 | 157 | 110 | 179 |
| Human | 7456 | 31.0 | 368 | −3.0 | 18900 | 28010 | 3690 |
| DQN (Mnih 2013) | 4092 | 168.0 | 470 | 20.0 | 1952 | 1705 | 581 |
| UCC-I (Guo 2014) | 5702 | 380 | 741 | 21 | 20025 | 2995 | 692 |
| TRPO single path | 1425.2 | 10.8 | 534.6 | 20.9 | 1973.5 | 1908.6 | 568.4 |
| TRPO vine | 859.5 | 34.2 | 430.8 | 20.9 | 7732.5 | 788.4 | 450.2 |

Atari settings: single path 100K sim steps per iteration, vine 400K, about 4 rollouts per state for vine [Table 3].

## Gotchas when reproducing

- δ = 0.01 (average KL) for every experiment [§8.1, Tables 2-3].
- Keep the line search. It checks the true KL and the true surrogate, not the quadratic model [App. C].
- Use the analytic KL Hessian for the Fisher, computed on about 10% of the batch, with about 10 CG iterations [App. C.1].
- Batch sizes are large: up to 10,000 paths × 1000 steps per iteration for single-path walker [Table 2].
- Single path uses plain Monte Carlo Q (discounted reward-to-go), not a learned value baseline; the paper notes that replacing A by Q only shifts the objective by a constant [§5]. The vine self-normalized estimator removes the need for a baseline [Eq. 16].
- Vine requires resetting the simulator to arbitrary states and uses common random numbers across branch rollouts [§5.2].
- The Gaussian log-std is a free state-independent parameter vector [App. D].
- The paper's walker state dimension is inconsistent (18 in §8.1, 20 in Table 2).

## Connections

Policy gradient foundations and theory: the performance-difference identity, the surrogate objective, and natural gradient all show up here, and the bound in Eq. 9 is the theory behind "trust region" methods. PPO (a later paper by the same first author, not covered here) replaces this second-order constrained step with a clipped first-order objective, so TRPO is the background for the "PPO and friends" lecture.
