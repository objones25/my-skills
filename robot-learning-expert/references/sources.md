# Sources and errata

Where the concept files' citations point, and the known errors in those sources. When two sources disagree, the concept file says so; this file records the specific errors.

## Citation keys

| Key | Source | Notes |
|---|---|---|
| AJKS | Agarwal, Jiang, Kakade, Sun, *Reinforcement Learning: Theory and Algorithms*, free at rltheorybook.github.io | theory and the V^π, Q^π, T⋆ notation used here; section numbers were checked against a June 2026 draft and may drift between drafts, so search by title if one looks off |
| SB | Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed., free at incompleteideas.net/book | value-function methods, TD, Q-learning; little on modern policy gradient |
| BBC | Bagnell, Boots, Choudhury, *Modern Adaptive Control and Reinforcement Learning* (draft) | trajectory optimization, imitation, control-flavored RL |
| Åström and Murray | *Feedback Systems* | control intuition (stabilization, tracking) |
| Notes N | J. A. Preiss, *Introduction to Robot Learning* lecture notes (UC Santa Barbara, 2026): 0 intro, 1 MDPs, 2 continuous states (LQR, trajectory optimization), 3 bandits, 4 model-based RL, 5 tabular Q-learning, 6 linear function approximation, 7 deep Q-learning, 8 policy gradient part I (score function, REINFORCE, baselines, entropy), 9 policy gradient advanced (TRPO, NPG, PPO, optimality, PC-PG) | concise derivations in AJKS notation; drafts, so check the errata below. Notes 8 and 9 (Winter 2026, revision A) use a finite horizon, t = 1..H, no discount, and write Q_t^π, V_t^π with a step index |
| `[Paper §x]` | the paper digests in `papers/` | located claims; `papers/README.md` indexes them |

## Errata in the lecture notes

| Where | What the PDF says | What it should be |
|---|---|---|
| Notes 1 §6.2 | to act from V⋆, solve min_a r(s,a) + E[V⋆(s')] | max, and the expectation should carry γ |
| Notes 4 §5.1 (MPC) | min over a_{t:t+K−1} of E[Σ r] | max (they are rewards), or min of costs |
| Notes 3 §3.1 | gap Δ_a = μ_a − μ⋆ | Δ_a = μ⋆ − μ_a ≥ 0 (otherwise Reg_n = Σ Δ_a E[T_a(n)] would be negative) |
| Notes 3 §3 | Hoeffding gives z = sqrt(log(1/δ)/(2n)) but the UCB index uses sqrt(2 log(1/δ)/T_a) | Not wrong: the index is a looser bound (the 1-subgaussian form from Lattimore and Szepesvári); just do not expect the two constants to match. |
| Notes 3 App. A | Hoeffding's lemma says X is "(b−a)/2-subgaussian" while Def. 15 uses "σ²-subgaussian" | It means σ = (b−a)/2, i.e. ((b−a)²/4)-subgaussian |
| Notes 1 Def. 14 | greedy w.r.t. V: argmax_a r(s,a) + E[V(s')] | needs γ: r(s,a) + γ E[V(s')] (Thm 22 assumes it) |
| Notes 1 Alg. 3 / Prop. 21 | "Initialize V_0 arbitrarily", bound γ^k/(1−γ) | the bound needs V_0 = 0 (or ‖V_0 − V⋆‖ ≤ 1/(1−γ)) |
| Notes 1 eq. (15) | starts from T^{π_{i+1}} Q̂^{π_i}, then drops the hat | π_{i+1} is greedy w.r.t. the estimate Q̂^{π_i}, so the identity is T^{π_{i+1}} Q̂^{π_i} = T⋆ Q̂^{π_i} |
| Notes 2 p. 7 (Fig. 1 example) | "no fixed action sequence has value ≥ 1/2" | every fixed sequence gets exactly 1/2; read "> 1/2" |
| Notes 4 Alg. 2 l. 5 | s^k_t ~ P(·\|s^k_t, a^k_t) | s^k_t ~ P(·\|s^k_{t−1}, a^k_{t−1}) |
| Notes 3 Thm 16, 17, 20 | "(σ_1² + σ²)", "a²σ_1²", "X_i is σ_i-subgaussian" | σ_1² + σ_2²; a²σ²; σ_i²-subgaussian |
| Notes 3 App. A | Markov proof writes 1{Z ≥ t}; eq. (6) labels the second moment m_1 | 1{X ≥ t}; m_2 |
| Notes 3 p. 10 | ellipsoid axis lengths are "the reciprocal of the eigenvalues" | proportional to 1/sqrt(eigenvalue) |
| Notes 5 Alg. 1 | a_t = max_a Q_t(s_t, a) | argmax_a Q_t(s_t, a) (we need the action, not the value) |
| Notes 5 Alg. 2 l. 6 to 7 | b_i, α_i | b_t, α_t with t = N_h(s_h, a_h), the per-pair count (the text below the algorithm says so) |
| Notes 5 §3.2 | "T is the number of episodes" | Jin et al. define T = KH, the total number of steps |
| Notes 6 eq. (2) | r(s,a) E[max Q⋆_{h+1}] | r(s,a) + E[...] (missing plus; finite horizon, so no γ) |
| Notes 6 §4 example | "this magnifies the noise ε_1" | ε_2 (θ̂_2 = θ_2 + ε_2/δ) |
| Notes 6 §4 | ellipsoid {x : ‖x‖²_Σ ≤ d} | ‖x‖²_{Σ⁻¹} ≤ d, matching the design condition |
| Notes 6 §5 eigen view | b = β‖Σ^{-1/2}Vϕ‖ with Λ = VΣVᵀ | β‖Σ^{-1/2}Vᵀϕ‖ (missing transpose) |
| Notes 6 Alg. 2 l. 8 | Q_h = min{θ_hᵀφ + β b_h, H} with b_h already containing β | apply β once: min{θ_hᵀφ + b_h, H} (Jin et al. Alg. 1 l. 6) |
| Notes 6 Alg. 2 l. 14 | v_h accumulates targets computed with the Q_{h+1} of the episode they were collected in | Jin et al. recompute every past target with the current Q_{h+1} each episode; the incremental version is cheaper but its old targets go stale |
| Notes 6 §5 | β = C d sqrt(log(2dH³T/δ)) | Jin et al. Thm 3.1: β = c·d·H·sqrt(ι), ι = log(2dT/p). The "extra |S||A|" the notes puzzle over comes from the uniform-concentration (covering) argument over the value class (Jin et al. §4, Lemma B.3) |
| Notes 6 §5 | lower bound Õ(√(dH²T)) | Ω̃ (a lower bound) |
| Notes 8 Thm 1 | "An unbiased estimate of the gradient of f̂" | of Ĵ (the smoothed objective defined just above) |
| Notes 8 §3, Challenge 1 | "Since π ∈ △(d)" | △(K) (the bandit has K arms) |
| Notes 8 §3, Challenge 2 | "The e_i − π term will be very close to 0" | e_{a_1} − π: only the near-always sampled action a_1 makes it small; a rarely sampled a_i gives e_i − π ≈ e_i − e_1 |
| Notes 8 §4 | b⋆ = min_b tr(Cov[ĝ_b]) | argmin (b⋆ is the minimizer); the closed form E[‖g‖²R]/E[‖g‖²] is for a constant b and the full-return estimator |
| Notes 9 §3 | x⋆ = −sqrt(ε/(aᵀΣ⁻¹a)) Σ⁻¹a | no minus sign: maximizing aᵀx subject to xᵀΣx ≤ ε gives x⋆ = +sqrt(ε/(aᵀΣ⁻¹a)) Σ⁻¹a (then aᵀx⋆ = sqrt(ε aᵀΣ⁻¹a) > 0) |
| Notes 9 §3 | LineSearch takes one shrink step x + αδ | a simplification: TRPO backtracks repeatedly (α^j δ, j = 0, 1, ...) and accepts the first step that improves the surrogate and satisfies the KL constraint (`papers/trpo.md`) |
| Notes 9 Alg. 1 l. 7 | θ ← θ − ∇_θ Σ_t min{...} | θ ← θ + η ∇_θ Σ_t min{...}: the clipped objective is maximized, and the learning rate η from the Require line is missing |
| Notes 9 Fig. 2b caption | "γ = 1.8" | a discount factor must be below 1 (the quoted text says the optimal policy needs γ > 1/2); likely a typo, not checked against Bhandari and Russo (2019) |

## Errata found in papers

Collected in each digest's "gotchas" or "ambiguities" section and in `papers/canon-algorithms.md` ("Disagreements and errors found") and `papers/canon-robotics.md` ("Unverified or inconsistent items"). Examples: PPO Eq. 10-11 print the last GAE exponent as T−t+1 (should be T−t−1); SAC Algorithms and Applications Eq. 7 writes Q(s_t, s_t) for Q(s_t, a_t).
